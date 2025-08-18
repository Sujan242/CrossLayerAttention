from typing import Optional, Union, Callable, List

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM, GenerationConfig, LogitsProcessorList, StoppingCriteriaList, \
    DynamicCache
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from .DynamicCacheCrossLayer import DynamicCacheCrossLayer, DynamicCacheCrossLayerWithOldCache


class LlamaWithAllLayerCrossAttention(LlamaForCausalLM):

    def __init__(self, config: LlamaConfig,mode, num_layers_to_attend: int = 1):
        # Create configuration
        super().__init__(config)
        self.loss_type = "ForMaskedLM"
        self.num_layers_to_attend = num_layers_to_attend
        self.mode = mode

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs,
    ):
        batch_size, seq_len = input_ids.shape
        if past_key_values is None or len(past_key_values.key_cache) == 0:
            past_key_values = DynamicCacheCrossLayer(mode=self.mode, num_layers_to_attend=self.num_layers_to_attend)
        device = input_ids.device
        loss = 0 # Initialize loss
        logits = []
        for t in range(seq_len):
            current_input = input_ids[:, t].unsqueeze(1)
            # if t != 0:
            #     # add (config.num_hidden_layers -1 ) 1s to the begginning of the attention mask
            #     current_mask = torch.cat(
            #         [torch.ones(batch_size, (self.config.num_hidden_layers - 1) * t).to(device=device),
            #          attention_mask[:, :t + 1]], dim=1)
            # else:
            #     current_mask = attention_mask[:, :t + 1]

            current_labels = labels[:, t+1].unsqueeze(1) if (labels is not None and t!=seq_len-1) else None
            logits_to_keep = 1
            # setting attention mask to None because we are using causal attention and right padding
            outputs = super().forward(input_ids=current_input, attention_mask=None,
                            past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                            labels=current_labels, use_cache=use_cache, output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states, return_dict=return_dict,
                         logits_to_keep=logits_to_keep, **kwargs
            )
            if outputs.loss is not None:
                loss += outputs.loss
            logits.append(outputs.logits)
        logits = torch.cat(logits, dim=1)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values,
                                      hidden_states=output_hidden_states,
                                      attentions=output_attentions)


class LlamaWithPreviousLayerCrossAttention(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        # Create configuration
        super().__init__(config)
        self.model.layers = nn.ModuleList(
            [LlamaDecoderForPreviousLayerAttention(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCacheCrossLayer] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ):
        # Hack to check if prefilling or not. just check if the KV cache is empty
        if past_key_values is None or len(past_key_values.key_cache) == 0:
            past_key_values = DynamicCacheCrossLayer(mode='previous', equivalent_to_training=True)
        else:
            past_key_values.set_equivalent_to_training(False)

        return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, cache_position=cache_position, logits_to_keep=logits_to_keep, **kwargs)



class LlamaDecoderForPreviousLayerAttention(LlamaDecoderLayer):

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value= None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings = None,
            **kwargs):
        attention_mask = self._augment_attention_mask(attention_mask)
        return super().forward(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position, position_embeddings, **kwargs)

    def _augment_attention_mask(self, attention_mask):
        if attention_mask is None or self.self_attn.layer_idx == 0:
            return attention_mask
        batch_size, _, num_tokens, _ = attention_mask.shape
        # clone attention mask
        attention_mask_clone = attention_mask.clone()

        # set all the diagnol elements to -inf
        diag_mask = torch.eye(num_tokens, dtype=torch.bool, device=attention_mask.device)
        attention_mask_clone = attention_mask_clone.masked_fill(diag_mask[None, None, :, :], torch.finfo(attention_mask.dtype).min)

        # repeat attention mask clone for layer_idx times
        attention_mask_clone = attention_mask_clone.repeat(1, 1, 1, self.self_attn.layer_idx)

        # append the original attention mask to the repeated attention mask
        attention_mask = torch.cat([attention_mask_clone, attention_mask], dim=-1) # TODO validate memory and compute overhead

        return attention_mask

class LlamaCrossLayerAttentionTwoPass(LlamaForCausalLM):

    def __init__(self, config: LlamaConfig,mode, num_layers_to_attend: int = 1):
        super().__init__(config)
        self.loss_type = "ForMaskedLM"
        self.num_layers_to_attend = num_layers_to_attend
        self.first_pass_cache = None
        self.second_pass_cache = None
        self.model.layers = nn.ModuleList(
            [LlamaDecoderForPreviousLayerAttention(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values=None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs,
    ):

        # first pass
        if self.first_pass_cache is None or len(self.first_pass_cache.key_cache) == 0:
            self.first_pass_cache = DynamicCache()

        super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                               past_key_values=self.first_pass_cache, inputs_embeds=inputs_embeds, labels=labels,
                               use_cache=use_cache, output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states, return_dict=return_dict,
                               cache_position=cache_position, logits_to_keep=logits_to_keep, **kwargs)

        # second pass
        if self.second_pass_cache is None or len(self.second_pass_cache.key_cache) == 0:
            self.second_pass_cache = DynamicCacheCrossLayerWithOldCache(self.first_pass_cache, self.num_layers_to_attend)
        else:
            # setting the updated first pass cache
            self.second_pass_cache.previous_cache = self.first_pass_cache

        return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                               past_key_values=self.second_pass_cache, inputs_embeds=inputs_embeds, labels=labels,
                               use_cache=use_cache, output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states, return_dict=return_dict,
                               cache_position=cache_position, logits_to_keep=logits_to_keep, **kwargs)

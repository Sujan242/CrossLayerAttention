from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel, LlamaPreTrainedModel, LlamaForCausalLM, GenerationMixin
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from .CustomDynamicCache import CustomDynamicCache


class LlamaWithAllLayerCrossAttention(LlamaPreTrainedModel):
    def __init__(self, vocab_size, hidden_size=512, num_attention_heads=5,
                 num_hidden_layers=4,
                 attention_dropout=0.1,
                 hidden_dropout=0.1):

        # Create configuration
        self.config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            attention_dropout=attention_dropout,  # Add dropout
            hidden_dropout=hidden_dropout,
        )
        super().__init__(self.config)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        self.llama_model = LlamaModel(self.config)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape
        dynamic_cache = CustomDynamicCache()
        logits = []

        device = input_ids.device

        for t in range(seq_len):
            # Get current token for all batches
            current_input = input_ids[:, t].unsqueeze(1)

            current_mask = None
            # add (config.num_hidden_layers -1 ) 1s to the begginning of the attention mask
            if t != 0:
                current_mask = torch.cat([torch.ones(batch_size, (self.config.num_hidden_layers - 1)*t).to(device=device), attention_mask[:, :t + 1]], dim=1)
            else:
                current_mask = attention_mask[:, :t + 1]

            # current_position_ids = position_ids[:, t].unsqueeze(1) if position_ids is not None else None

            # Get embeddings
            embeddings = self.embed_tokens(current_input)

            # Forward pass with cache
            outputs = self.llama_model(
                inputs_embeds=embeddings,
                attention_mask=current_mask,
                # position_ids=current_position_ids,
                past_key_values=dynamic_cache,
                use_cache=True
            )

            # Get logits for next token prediction
            next_logits = self.lm_head(outputs.last_hidden_state)
            logits.append(next_logits)

        # Combine all logits
        logits = torch.cat(logits, dim=1)  # [batch_size, seq_len, vocab_size]

        loss = None
        if labels is not None:
            # Shift logits and labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )

        return {'loss': loss, 'logits': logits}

    def generate(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor = None,
            max_new_tokens: int = 20,
            temperature: float = 1.0,
            eos_token_id: int = None,
    ):
        generated_ids = []
        dynamic_cache = CustomDynamicCache()  # Initialize cache

        # input_embeds = self.embed_tokens(input_ids)
        # outputs = self.llama_model(
        #     inputs_embeds=input_embeds,
        #     attention_mask=attention_mask,
        #     past_key_values=dynamic_cache,
        #     use_cache=True
        # )
        # TODO parallel - prefilling
        for t in range(input_ids.shape[1]):
            current_input = input_ids[:, t].unsqueeze(1)

            embeddings = self.embed_tokens(current_input)

            outputs = self.llama_model(
                inputs_embeds=embeddings,
                past_key_values=dynamic_cache,
                use_cache=True
            )

        # Autoregressive generation loop
        for _ in range(max_new_tokens):
            # Get logits for next token (use last token in sequence)
            next_logits = self.lm_head(outputs.last_hidden_state[:, -1, :])

            # Apply temperature
            next_token = torch.argmax(next_logits / temperature, dim=-1)

            # Append generated token
            generated_ids.append(next_token)

            # Stop if EOS generated
            if eos_token_id is not None and (next_token == eos_token_id).any():
                break

            embeds = self.embed_tokens(next_token.unsqueeze(1))
            # Forward the new token through the model
            outputs = self.llama_model(
                inputs_embeds=embeds,
                past_key_values=dynamic_cache,  # Reuse updated cache
                use_cache=True
            )

        return generated_ids

class LlamaWithPreviousLayerCrossAttention(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        # Create configuration
        super().__init__(config)
        self.model.layers = nn.ModuleList(
            [CustomLlamaDecoder(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values= None,
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
        past_key_values = CustomDynamicCache() # TODO pass in mode here
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, cache_position=cache_position, logits_to_keep=logits_to_keep, **kwargs)



class CustomLlamaDecoder(LlamaDecoderLayer):

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values= None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **flash_attn_kwargs,
    ) :
        attention_mask = self._augment_attention_mask(attention_mask)
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, cache_position=cache_position, **flash_attn_kwargs)


    def _augment_attention_mask(self, attention_mask) -> torch.Tensor:
        return attention_mask





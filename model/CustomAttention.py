# from typing import Optional, Tuple, Dict, Callable
#
# import torch
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, eager_attention_forward
# from transformers.integrations.flex_attention import flex_attention_forward
# from transformers.integrations.sdpa_attention import sdpa_attention_forward
# from transformers.integrations.flash_attention import flash_attention_forward
# from transformers.utils import logging
#
# from transformers.cache_utils import Cache
#
# ALL_ATTENTION_FUNCTIONS: Dict[str, Dict[str, Callable]] = {}
#
# ALL_ATTENTION_FUNCTIONS.update(
#     {
#         "flash_attention_2": flash_attention_forward,
#         "flex_attention": flex_attention_forward,
#         "sdpa": sdpa_attention_forward,
#     }
# )
#
# logger = logging.get_logger(__name__)
# class CustomLlamaAttention(LlamaAttention):
#
#     def forward(
#             self,
#             hidden_states: torch.Tensor,
#             position_embeddings: Tuple[torch.Tensor, torch.Tensor],
#             attention_mask: Optional[torch.Tensor],
#             past_key_value: Optional[Cache] = None,
#             cache_position: Optional[torch.LongTensor] = None,
#             **kwargs,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#
#         input_shape = hidden_states.shape[:-1]
#         hidden_shape = (*input_shape, -1, self.head_dim)
#
#         # Original projections and rotary embeddings
#         query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
#         key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
#         value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
#
#         cos, sin = position_embeddings
#         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
#
#         # Collect previous token's cross-layer KV
#         cross_layer_kvs = []
#         if past_key_value is not None and len(past_key_value) > 0:
#             # Extract all layers' keys and values from the cache
#             prev_keys = past_key_value.key_cache  # Shape: [batch, num_layers, seq_len, num_heads, head_dim]
#             prev_values = past_key_value.value_cache  # Shape: [batch, num_layers, seq_len, num_heads, head_dim]
#
#             # Get the last token's keys/values from all layers
#             cross_layer_kvs = [
#                 prev_keys[:, :, -1:],  # [batch, num_layers, 1, num_heads, head_dim]
#                 prev_values[:, :, -1:]
#             ]
#
#         cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
#         key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
#
#         # Combine with current layer's KV
#         if cross_layer_kvs:
#             # current_layer_kv: [batch, seq_len, heads, dim]
#             # cross_layer_kvs: [batch, num_layers, 1, heads, dim]
#             key_states = torch.cat([
#                 key_states.unsqueeze(1),  # add layer dimension
#                 cross_layer_kvs[0]
#             ], dim=1).flatten(1, 2)  # [batch, seq_len + num_layers, heads, dim]
#
#             value_states = torch.cat([
#                 value_states.unsqueeze(1),
#                 cross_layer_kvs[1]
#             ], dim=1).flatten(1, 2)
#
#         attention_interface: Callable = eager_attention_forward
#         if self.config._attn_implementation != "eager":
#             if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
#                 logger.warning_once(
#                     "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
#                     'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
#                 )
#             else:
#                 attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
#
#         attn_output, attn_weights = attention_interface(
#             self,
#             query_states,
#             key_states,
#             value_states,
#             attention_mask,
#             dropout=0.0 if not self.training else self.attention_dropout,
#             scaling=self.scaling,
#             **kwargs,
#         )
#
#         attn_output = attn_output.reshape(*input_shape, -1).contiguous()
#         attn_output = self.o_proj(attn_output)
#
#
#         return attn_output, attn_weights
#
#

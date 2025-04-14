from transformers import DynamicCache
import torch
from typing import Dict, Any, Optional, Tuple


class DynamicCacheCrossLayer(DynamicCache):

    def __init__(self, mode='all', equivalent_to_training=False, num_hidden_layers=None, num_layers_to_attend=None):
        super().__init__(num_hidden_layers)
        self.mode = mode
        self.equivalent_to_training = equivalent_to_training
        self.num_layers_to_attend = num_layers_to_attend

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        super().update(key_states, value_states, layer_idx, cache_kwargs)

        # Gather token's cache from all other layers
        if self._seen_tokens > 1:
            if self.mode == 'next':
                if layer_idx == len(self.key_cache) - 1:
                    return self.key_cache[layer_idx], self.value_cache[layer_idx]
                prev_token_key_cache = torch.cat([self.key_cache[i][:, :, :self._seen_tokens - 1, :]
                                                  for i in range(layer_idx+1,
                                                                 len(self.key_cache))
                                                  if i != layer_idx],
                                                 dim=2)
                prev_token_value_cache = torch.cat([self.value_cache[i][:, :, :self._seen_tokens - 1, ]
                                                    for i in range(layer_idx+1,
                                                                   len(self.key_cache))
                                                    if i != layer_idx],
                                                   dim=2)
            elif self.mode.startswith('top_'):
                if self.num_layers_to_attend == 1 and layer_idx == len(self.key_cache)-1:
                    return self.key_cache[layer_idx], self.value_cache[layer_idx]

                prev_token_key_cache = torch.cat([self.key_cache[i][:, :, :self._seen_tokens - 1, :]
                                                  for i in range(len(self.key_cache)-self.num_layers_to_attend, len(self.key_cache))
                                                  if i != layer_idx],
                                                 dim=2)
                prev_token_value_cache = torch.cat([self.value_cache[i][:, :, :self._seen_tokens - 1, ]
                                                    for i in range(len(self.key_cache)-self.num_layers_to_attend, len(self.key_cache))
                                                    if i != layer_idx],
                                                   dim=2)
            elif self.mode == 'previous':
                if layer_idx == 0:
                    return self.key_cache[layer_idx], self.value_cache[layer_idx]

                upto_token = self._seen_tokens

                # if self.equivalent_to_training: # training or prefilling
                #     upto_token = self._seen_tokens
                # else:
                #     upto_token = self._seen_tokens - 1

                prev_token_key_cache = torch.cat([self.key_cache[i][:, :, :upto_token, :]
                                                  for i in range(layer_idx)],
                                                 dim=2)
                prev_token_value_cache = torch.cat([self.value_cache[i][:, :, :upto_token, ]
                                                    for i in range(layer_idx)],
                                                   dim=2)
            else:
                raise ValueError(f"Invalid mode {self.mode}")
            concatenated_keys = torch.cat([prev_token_key_cache, self.key_cache[layer_idx]], dim=2)
            concatenated_values = torch.cat([prev_token_value_cache, self.value_cache[layer_idx]], dim=2)
            return concatenated_keys, concatenated_values

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def set_equivalent_to_training(self, equivalent_to_training):
        self.equivalent_to_training = equivalent_to_training

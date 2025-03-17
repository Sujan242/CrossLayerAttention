from transformers import DynamicCache
import torch
from typing import Dict, Any, Optional, Tuple


class CustomDynamicCache(DynamicCache):

    def __init__(self, mode='all', training='true', num_hidden_layers=None):
        super().__init__(num_hidden_layers)
        self.mode = mode
        self.training = training

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
            if self.mode == 'all':
                prev_token_key_cache = torch.cat([self.key_cache[i][:, :, :self._seen_tokens - 1, :]
                                                  for i in range(len(self.key_cache))
                                                  if i != layer_idx],
                                                 dim=2)
                prev_token_value_cache = torch.cat([self.value_cache[i][:, :, :self._seen_tokens - 1, ]
                                                    for i in range(len(self.key_cache))
                                                    if i != layer_idx],
                                                   dim=2)
            elif self.mode == 'previous':
                if layer_idx == 0:
                    return self.key_cache[layer_idx], self.value_cache[layer_idx]

                if self.training:
                    upto_token = self._seen_tokens
                else:
                    upto_token = self._seen_tokens - 1

                prev_token_key_cache = torch.cat([self.key_cache[i][:, :, :upto_token, :]
                                                  for i in range(layer_idx)],
                                                 dim=2)
                prev_token_value_cache = torch.cat([self.value_cache[i][:, :, :upto_token, ]
                                                    for i in range(layer_idx)],
                                                   dim=2)
            else:
                raise ValueError(f"Invalid mode {self.mode}")
            concatenated_keys = torch.cat([prev_token_key_cache, self.key_cache[layer_idx]], dim=2)
            concatenated_values = torch.cat([prev_token_value_cache, self.value_cache[layer_idx]], dim=-2)
            return concatenated_keys, concatenated_values # TODO validate memory and compute overhead

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

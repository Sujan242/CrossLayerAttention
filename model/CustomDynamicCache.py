from transformers import DynamicCache
import torch
from typing import Dict, Any, Optional, Tuple


class CustomDynamicCache(DynamicCache):

    def __init__(self, mode='all', num_hidden_layers=None):
        super().__init__(num_hidden_layers)
        self.mode = mode

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
                concatenated_keys = torch.cat([self.key_cache[i][:, :, :self._seen_tokens-1,: ] for i in range(len(self.key_cache)) ], dim=2)
                concatenated_values = torch.cat([self.value_cache[i][:, :, :self._seen_tokens-1, ] for i in range(len(self.key_cache)) ], dim=-2)
            elif self.mode == 'previous':
                concatenated_keys = torch.cat([self.key_cache[i] for i in range(len(layer_idx))], dim=2)
                concatenated_values = torch.cat([self.value_cache[i] for i in range(len(layer_idx))], dim=-2)
            else:
                raise ValueError(f"Invalid mode {self.mode}")
            return concatenated_keys, concatenated_values # TODO validate memory and compute overhead

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

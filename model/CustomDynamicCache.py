from transformers import DynamicCache
import torch
from typing import Dict, Any, Optional, Tuple


class CustomDynamicCache(DynamicCache):

    def __init__(self, parallel_training: bool = False):
        super().__init__()
        self._parallel_training = parallel_training

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states for the current layer,
            appended with the cache of all other layers from the previous token.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # Fill skipped layers with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append([])
                    self.value_cache.append([])
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            elif len(self.key_cache[layer_idx]) == 0:
                # Handle previously skipped layers
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        # Gather previous token's cache from all other layers
        if self._seen_tokens > 1:
            previous_token_key_caches = []
            prev_token_value_caches = []
            if self._parallel_training:
                previous_token_key_caches = [self.key_cache[i][:, :, -1, ] for i in range(layer_idx)]
                prev_token_value_caches = [self.value_cache[i][:, :, -1, ] for i in range(layer_idx)]
            else:
                previous_token_key_caches = [self.key_cache[i][:, :, -1, ] for i in range(len(self.key_cache)) if i != layer_idx]
                prev_token_value_caches = [self.value_cache[i][:, :, -1, ] for i in range(len(self.key_cache)) if i != layer_idx]

            if len(previous_token_key_caches) == 0:
                return self.key_cache[layer_idx], self.value_cache[layer_idx]

            prev_token_key_cache = torch.stack(previous_token_key_caches, dim=2)
            prev_token_value_cache = torch.stack(prev_token_value_caches, dim=2)

            concatenated_keys = torch.cat([prev_token_key_cache,self.key_cache[layer_idx] ], dim=2)
            concatenated_values = torch.cat([prev_token_value_cache,self.value_cache[layer_idx]], dim=-2)
            return concatenated_keys, concatenated_values

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

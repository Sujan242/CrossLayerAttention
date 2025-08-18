import torch
from transformers import DynamicCache


def test_dynamic_cache():
    cache = DynamicCache()  # Create an empty cache

    # Dummy key-value tensors
    key1 = torch.rand(1, 5, 10)  # Shape: (batch, seq_len, dim)
    value1 = torch.rand(1, 5, 10)

    # Update the cache for layer 0
    cache.update(layer_idx=0, key_states=key1, value_states=value1)

    # Retrieve cache state
    cached_states = cache.key_cache

    # Assertions
    # assert 0 in cached_states, "Layer 0 should exist in cache"
    # assert "key" in cached_states[0] and "value" in cached_states[0], "Cache should store keys and values"
    # assert torch.equal(cached_states[0]["key"], key1), "Stored key does not match input"
    # assert torch.equal(cached_states[0]["value"], value1), "Stored value does not match input"
    #
    # print("DynamicCache update function test passed!")


# Run the test
test_dynamic_cache()

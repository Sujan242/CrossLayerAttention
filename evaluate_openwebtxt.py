import torch
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast, LlamaConfig
from datasets import load_dataset
import math
from safetensors.torch import load_file

from model.model import LlamaWithAllLayerCrossAttention, LlamaWithPreviousLayerCrossAttention
from train import load_config

def tokenize(example, tokenizer):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

def get_tokenized_dataset(tokenizer):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    split_dataset = dataset["train"].train_test_split(test_size=0.01, seed=42)
    test_valid_split = split_dataset["test"].train_test_split(test_size=0.5, seed=42)

    dataset_dict = {
        "train": split_dataset["train"],
        "validation": test_valid_split["train"],
        "test": test_valid_split["test"],
    }

    for split in ["train", "validation", "test"]:
        dataset_dict[split] = dataset_dict[split].map(lambda x: tokenize(x, tokenizer), batched=True, remove_columns=["text"])
        dataset_dict[split].set_format(type="torch", columns=["input_ids", "attention_mask"])

    return dataset_dict


def get_model(cfg, checkpoint_path):
    config = LlamaConfig.from_pretrained(checkpoint_path)

    if cfg.model_configs.mode == "traditional":
        from transformers import LlamaForCausalLM
        model = LlamaForCausalLM(config)
    elif cfg.model_configs.mode == "previous":
        model = LlamaWithPreviousLayerCrossAttention(config)
    elif cfg.model_configs.mode.startswith("top") or cfg.model_configs.mode == "next":
        model = LlamaWithAllLayerCrossAttention(config, mode=cfg.model_configs.mode)
    else:
        raise ValueError("Unsupported model mode")

    state_dict = load_file(f"{checkpoint_path}/model.safetensors")

    # If model was trained with DataParallel, remove "module." prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    return model

def evaluate_model(model, dataset, batch_size, device):
    model.eval()
    model.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size)
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / count
    perplexity = math.exp(avg_loss)
    print(f"Eval Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1]  # path to your YAML config
    checkpoint_path = sys.argv[2]  # path to checkpoint dir, e.g., results/checkpoint-25844

    cfg = load_config(config_path)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    tokenized = get_tokenized_dataset(tokenizer)
    model = get_model(cfg, checkpoint_path)
    device = torch.device(f"cuda:{cfg.gpu.ids[0]}" if torch.cuda.is_available() else "cpu")

    evaluate_model(model, tokenized["test"], batch_size=1, device=device)

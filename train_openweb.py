from transformers import LlamaConfig, LlamaForCausalLM
import torch

from model.model import LlamaWithAllLayerCrossAttention, LlamaWithPreviousLayerCrossAttention
from train import load_config
import sys
from torch.nn import DataParallel
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
import math

def compute_metrics(eval_pred):
    loss = eval_pred.loss if hasattr(eval_pred, "loss") else eval_pred[0]
    perplexity = math.exp(loss) if loss < 300 else float("inf")
    return {"perplexity": perplexity, "eval_loss": loss}

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM
)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

def get_dataset():
    # Load OpenWebText dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")  # ~2MB

    # Use a small % for testing, validation
    split_dataset = dataset["train"].train_test_split(test_size=0.01, seed=42)
    test_valid_split = split_dataset["test"].train_test_split(test_size=0.5, seed=42)


    dataset_dict = {
        "train": split_dataset["train"],
        "validation": test_valid_split["train"],
        "test": test_valid_split["test"],
    }

    # Tokenize
    tokenized = dataset_dict.copy()
    for split in ["train", "validation", "test"]:
        tokenized[split] = tokenized[split].map(tokenize, batched=True, remove_columns=["text"])
        tokenized[split].set_format(type="torch", columns=["input_ids", "attention_mask"])

    return tokenized

def train(tokenized, model, cfg):
    training_args = TrainingArguments(
        output_dir=cfg.training_configs.output_dir,
        per_device_train_batch_size=cfg.training_configs.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training_configs.gradient_accumulation_steps,
        learning_rate=float(cfg.training_configs.learning_rate),
        num_train_epochs=cfg.training_configs.num_train_epochs,
        logging_dir=cfg.training_configs.logging_dir,
        remove_unused_columns=cfg.training_configs.remove_unused_columns,
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        per_device_eval_batch_size=cfg.training_configs.per_device_eval_batch_size,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    results = trainer.evaluate(eval_dataset=tokenized["test"])

    print("Test Perplexity:", math.exp(results["eval_loss"]))

    trainer.save_model("./mini-llama-final_traditional")
    tokenizer.save_pretrained("./mini-llama-final_traditional")

def get_model(cfg):
    config = LlamaConfig(
        vocab_size=50257,
        hidden_size=cfg.model_configs.hidden_size,
        num_attention_heads=cfg.model_configs.num_attention_heads,
        num_hidden_layers=cfg.model_configs.num_hidden_layers,
        attention_dropout=cfg.model_configs.attention_dropout,
        hidden_dropout=cfg.model_configs.hidden_dropout
    )
    if cfg.model_configs.mode.startswith("top"):
        print(f"Using top {cfg.model_configs.mode.split('_')[1]} layer cross attention")
        model = LlamaWithAllLayerCrossAttention(config, mode=cfg.model_configs.mode,
                                                num_layers_to_attend=int(cfg.model_configs.mode.split("_")[1]))
    elif cfg.model_configs.mode == "next":
        model = LlamaWithAllLayerCrossAttention(config, mode=cfg.model_configs.mode)
    elif cfg.model_configs.mode == "previous":
        print("Using previous layer cross attention")
        model = LlamaWithPreviousLayerCrossAttention(config)
    elif cfg.model_configs.mode == "traditional":
        model = LlamaForCausalLM(config)
    else:
        raise ValueError("Invalid mode")

    if torch.cuda.is_available():
         gpu_ids = cfg.gpu.ids
         print(f"using DataParallel training on GPUs: {gpu_ids}")
         model = DataParallel(model, device_ids=gpu_ids)
    return model

if __name__ == "__main__":
    # get the config path from the script arguments
    config_path = sys.argv[1]

    cfg = load_config(config_path)

    # if torch.cuda.is_available():
    #     torch.set_default_device(f'cuda:{cfg.gpu.ids[0]}')

    model = get_model(cfg)
    train(get_dataset(), model, cfg)
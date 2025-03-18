import os
import sys
from types import SimpleNamespace

import torch
import yaml
from torch.nn import DataParallel
from transformers import Trainer, TrainingArguments, LlamaConfig, LlamaForCausalLM

from model.SmallScaleLlama import LlamaWithAllLayerCrossAttention, LlamaWithPreviousLayerCrossAttention
from utils.AdditionDataset import AdditionDataset
from utils.AdditionDatasetEval import EvalAdditionDataset
from utils.AdditionEvalCallbackActual import AdditionEvalCallbackActual


def load_config(config_path: str) -> SimpleNamespace:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert nested dictionaries to SimpleNamespace
    for section in config:
        config[section] = SimpleNamespace(**config[section])

    return SimpleNamespace(**config)


def train(train_dataset, model, eval_callback, training_args):

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[eval_callback]
    )

    trainer.train()

if __name__ == "__main__":
    # get the config path from the script arguments

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = 0
    gpu_ids = [0, 1, 2, 3]
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpus = min(4, torch.cuda.device_count())
        gpu_ids = gpu_ids[:num_gpus]


    print(f"running on device:{device} with {num_gpus} gpus")

    config_path = sys.argv[1]
    cfg = load_config(config_path)

    train_dataset = AdditionDataset(cfg.data_configs.train_data_path, max_sequence_length=cfg.data_configs.max_sequence_length)

    eval_dataset = EvalAdditionDataset(
        file_path=cfg.data_configs.test_data_path,
        token_to_id=train_dataset.token_to_id,
        id_to_token=train_dataset.id_to_token,
        pad_token_id=train_dataset.pad_token_id,
        eos_token_id=train_dataset.eos_token_id
    )
    config = LlamaConfig(
        vocab_size=train_dataset.vocab_size,
        hidden_size=cfg.model_configs.hidden_size,
        num_attention_heads=cfg.model_configs.num_attention_heads,
        num_hidden_layers=cfg.model_configs.num_hidden_layers,
        attention_dropout=cfg.model_configs.attention_dropout,
        hidden_dropout=cfg.model_configs.hidden_dropout,
        pad_token_id=train_dataset.pad_token_id,
        eos_token_id=train_dataset.eos_token_id
    )

    eval_callback = AdditionEvalCallbackActual(eval_dataset,
                                               max_answer_length=cfg.eval_configs.max_answer_length,
                                               eval_interval=cfg.eval_configs.eval_interval,
                                               save_path=cfg.eval_configs.save_path)

    if cfg.model_configs.mode == "full":
        model = LlamaWithAllLayerCrossAttention(config).to(device)
    elif cfg.model_configs.mode == "previous":
        model = LlamaWithPreviousLayerCrossAttention(config).to(device)
    elif cfg.model_configs.mode == "traditional":
        model = LlamaForCausalLM(config).to(device)
    else:
        raise ValueError("Invalid mode")

    if os.path.exists(cfg.eval_configs.save_path):
        print("loading previous weights")
        model.load_state_dict(torch.load(cfg.eval_configs.save_path))

    if num_gpus > 1:
        print("using DataParallel training")
        model = DataParallel(model, device_ids=gpu_ids)

    training_args = TrainingArguments(
        output_dir=cfg.training_configs.output_dir,
        per_device_train_batch_size=cfg.training_configs.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training_configs.gradient_accumulation_steps,
        learning_rate=float(cfg.training_configs.learning_rate),
        num_train_epochs=cfg.training_configs.num_train_epochs,
        logging_dir=cfg.training_configs.logging_dir,
        remove_unused_columns=cfg.training_configs.remove_unused_columns
    )

    train(train_dataset, model, eval_callback, training_args)
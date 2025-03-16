from transformers import Trainer, TrainingArguments

from utils.AdditionDataset import AdditionDataset
from utils.AdditionDatasetEval import EvalAdditionDataset
from utils.AdditionEvalCallback import AdditionEvalCallback
from model.SmallScaleLlama import LlamaWithAllLayerCrossAttention
from types import SimpleNamespace
import yaml
import sys
import torch
import os


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

    print("running on device:", device)

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


    model = LlamaWithAllLayerCrossAttention(vocab_size=train_dataset.vocab_size,
                                            hidden_size=cfg.model_configs.hidden_size,
                                            num_attention_heads=cfg.model_configs.num_attention_heads,
                                            num_hidden_layers=cfg.model_configs.num_hidden_layers,
                                            attention_dropout=cfg.model_configs.attention_dropout,
                                            hidden_dropout=cfg.model_configs.hidden_dropout,
                                            ).to(device)

    if os.path.exists(cfg.eval_configs.save_path):
        print("loading previous weights")
        model.load_state_dict(torch.load(cfg.eval_configs.save_path))

    eval_callback = AdditionEvalCallback(eval_dataset,
                                         max_answer_length=cfg.eval_configs.max_answer_length,
                                         eval_interval=cfg.eval_configs.eval_interval,
                                         save_path=cfg.eval_configs.save_path)

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

# python3 train.py configs/addition/addition_1000.yaml
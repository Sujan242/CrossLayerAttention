import sys
from types import SimpleNamespace

import torch
import yaml
from transformers import Trainer, TrainingArguments

from utils.train_utils import get_train_and_eval_callback, get_model

import warnings
warnings.filterwarnings('ignore')

def load_config(config_path: str) -> SimpleNamespace:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert nested dictionaries to SimpleNamespace
    for section in config:
        config[section] = SimpleNamespace(**config[section])

    return SimpleNamespace(**config)


def train(train_dataset, val_dataset, model, eval_callback):

    training_args = TrainingArguments(
        output_dir=cfg.training_configs.output_dir,
        per_device_train_batch_size=cfg.training_configs.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training_configs.gradient_accumulation_steps,
        learning_rate=float(cfg.training_configs.learning_rate),
        num_train_epochs=cfg.training_configs.num_train_epochs,
        logging_dir=cfg.training_configs.logging_dir,
        remove_unused_columns=cfg.training_configs.remove_unused_columns,
        dataloader_pin_memory=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[eval_callback],
        eval_dataset=val_dataset,
    )

    trainer.train()

if __name__ == "__main__":
    # get the config path from the script arguments
    config_path = sys.argv[1]

    cfg = load_config(config_path)

    if torch.cuda.is_available():
        torch.set_default_device(f'cuda:{cfg.gpu.ids[0]}')

    train_dataset, val_dataset, eval_callback = get_train_and_eval_callback(cfg)

    model = get_model(train_dataset, cfg)

    train(train_dataset, val_dataset, model, eval_callback)
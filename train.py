import sys
from types import SimpleNamespace

import torch
import yaml
from transformers import Trainer, TrainingArguments

from utils.AdditionDatasetEval import EvalAdditionDataset
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

def evaluate(test_dataset, model, cfg):
    model = model.module if hasattr(model, 'module') else model
    device = next(model.parameters()).device

    model.eval()
    total, correct = 0, 0

    model.eval()
    with torch.no_grad():
        for example in test_dataset:
            input_ids = example["input_ids"].unsqueeze(0).to(device)
            answer = example["answer"]

            generated_answer_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=cfg.eval_configs.max_answer_length,
            )

            generated_answer = []
            for token_id in generated_answer_ids[0].tolist()[input_ids.shape[1]:]:
                if token_id == test_dataset.eos_token_id:
                    break
                generated_answer.append(test_dataset.id_to_token[token_id])
            generated_answer = ''.join(generated_answer)
            correct += (generated_answer == answer)
            total += 1

    accuracy = correct / total if total else 0
    print(f"\nEvaluation Accuracy: {accuracy:.4f}")

def train(train_dataset, val_dataset, model, eval_callback):

    training_args = TrainingArguments(
        output_dir=cfg.training_configs.output_dir,
        per_device_train_batch_size=cfg.training_configs.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training_configs.gradient_accumulation_steps,
        learning_rate=float(cfg.training_configs.learning_rate),
        num_train_epochs=cfg.training_configs.num_train_epochs,
        logging_dir=cfg.training_configs.logging_dir,
        remove_unused_columns=cfg.training_configs.remove_unused_columns,
        dataloader_pin_memory=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,  # Keep only the best checkpoint
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        lr_scheduler_type="cosine",
        logging_strategy="steps",
        logging_steps=0.1,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[eval_callback],
        eval_dataset=val_dataset
    )

    trainer.train()

    return trainer.model

if __name__ == "__main__":
    # get the config path from the script arguments
    config_path = sys.argv[1]


    print(f"__________________________starting training for config: {config_path}________________________________________")

    cfg = load_config(config_path)

    if torch.cuda.is_available():
        torch.set_default_device(f'cuda:{cfg.gpu.ids[0]}')

    train_dataset, val_dataset, eval_callback = get_train_and_eval_callback(cfg)

    model = get_model(train_dataset, cfg)

    model = train(train_dataset, val_dataset, model, eval_callback)

    test_dataset_test = EvalAdditionDataset(
        file_path=cfg.data_configs.test_data_path,
        token_to_id=train_dataset.token_to_id,
        id_to_token=train_dataset.id_to_token,
        pad_token_id=train_dataset.pad_token_id,
        eos_token_id=train_dataset.eos_token_id,
        max_length=cfg.data_configs.max_sequence_length,
        pad=False
    )

    test_dataset_train = EvalAdditionDataset(
        file_path=cfg.data_configs.train_data_path,
        token_to_id=train_dataset.token_to_id,
        id_to_token=train_dataset.id_to_token,
        pad_token_id=train_dataset.pad_token_id,
        eos_token_id=train_dataset.eos_token_id,
        max_length=cfg.data_configs.max_sequence_length,
        pad=False
    )

    test_dataset_val = EvalAdditionDataset(
        file_path=cfg.data_configs.val_data_path,
        token_to_id=train_dataset.token_to_id,
        id_to_token=train_dataset.id_to_token,
        pad_token_id=train_dataset.pad_token_id,
        eos_token_id=train_dataset.eos_token_id,
        max_length=cfg.data_configs.max_sequence_length,
        pad=False
    )

    print(f"__________________________evaluating on train dataset________________________________________")
    evaluate(test_dataset_train, model, cfg)
    print(f"__________________________evaluating on val dataset________________________________________")
    evaluate(test_dataset_val, model, cfg)
    print(f"__________________________evaluating on test dataset________________________________________")
    evaluate(test_dataset_test, model, cfg)

    print(f"__________________________finished training and evaluation for config: {config_path}________________________________________")
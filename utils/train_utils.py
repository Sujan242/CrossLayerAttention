import os

import torch
from torch.nn import DataParallel
from transformers import LlamaConfig, LlamaForCausalLM

from model.model import LlamaWithAllLayerCrossAttention, LlamaWithPreviousLayerCrossAttention
from utils.AdditionDataset import AdditionDataset
from utils.AdditionDatasetEval import EvalAdditionDataset
from utils.AdditionEvalCallbackActual import AdditionEvalCallbackActual
from ast import literal_eval


def get_train_and_eval_callback(cfg):
    train_dataset = AdditionDataset(cfg.data_configs.train_data_path,
                                    max_sequence_length=cfg.data_configs.max_sequence_length)

    eval_dataset = EvalAdditionDataset(
        file_path=cfg.data_configs.test_data_path,
        token_to_id=train_dataset.token_to_id,
        id_to_token=train_dataset.id_to_token,
        pad_token_id=train_dataset.pad_token_id,
        eos_token_id=train_dataset.eos_token_id,
        max_length=cfg.data_configs.max_sequence_length
    )

    eval_callback = AdditionEvalCallbackActual(eval_dataset,
                                               max_answer_length=cfg.eval_configs.max_answer_length,
                                               eval_interval=cfg.eval_configs.eval_interval,
                                               save_path=cfg.eval_configs.save_path,
                                               batch_size=cfg.eval_configs.batch_size)

    return train_dataset, eval_callback

def get_model(train_dataset, cfg, device):
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

    if torch.cuda.is_available():
        gpu_ids = literal_eval(cfg.gpu_ids)
        print(f"using DataParallel training on GPUs: {gpu_ids}")
        model = DataParallel(model, device_ids=literal_eval(gpu_ids))

    return model

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
    # create set of all characters - 256 ASCII characters
    characters = set([chr(i) for i in range(256)])
    special_tokens = ['<PAD>', '<EOS>']
    vocab = special_tokens + characters
    token_to_id = {char: idx for idx, char in enumerate(vocab)}
    id_to_token = {idx: char for idx, char in enumerate(vocab)}

    train_dataset = AdditionDataset(cfg.data_configs.train_data_path,
                                    id_to_token=id_to_token,
                                    token_to_id=token_to_id,
                                    max_sequence_length=cfg.data_configs.max_sequence_length,
                                    vocab=vocab)

    val_dataset = AdditionDataset(cfg.data_configs.val_data_path,
                                  id_to_token=id_to_token,
                                  token_to_id=token_to_id,
                                  max_sequence_length=cfg.data_configs.max_sequence_length,
                                  vocab=vocab)

    eval_dataset = EvalAdditionDataset(
        file_path=cfg.data_configs.test_data_path,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        pad_token_id=train_dataset.pad_token_id,
        eos_token_id=train_dataset.eos_token_id,
        max_length=cfg.data_configs.max_sequence_length
    )

    eval_callback = AdditionEvalCallbackActual(eval_dataset,
                                               max_answer_length=cfg.eval_configs.max_answer_length,
                                               eval_interval=cfg.eval_configs.eval_interval,
                                               save_path=cfg.eval_configs.save_path,
                                               batch_size=cfg.eval_configs.batch_size)

    return train_dataset, val_dataset, eval_callback

def get_model(train_dataset, cfg, device):
    # device = torch.device("cpu")
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
        model = LlamaWithAllLayerCrossAttention(config)
    elif cfg.model_configs.mode == "previous":
        model = LlamaWithPreviousLayerCrossAttention(config)
    elif cfg.model_configs.mode == "traditional":
        model = LlamaForCausalLM(config)
    else:
        raise ValueError("Invalid mode")

    if os.path.exists(cfg.eval_configs.save_path):
        print("loading previous weights")
        model.load_state_dict(torch.load(cfg.eval_configs.save_path))

    if torch.cuda.is_available():
         gpu_ids = cfg.gpu.ids
         print(f"using DataParallel training on GPUs: {gpu_ids}")
         model = model.to(f'cuda:{gpu_ids[0]}')
         model = DataParallel(model, device_ids=gpu_ids)

    return model

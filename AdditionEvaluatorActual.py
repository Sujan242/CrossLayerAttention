import sys, os
from types import SimpleNamespace

import torch
import yaml

from utils.AdditionDataset import AdditionDataset
from utils.AdditionDatasetEval import EvalAdditionDataset
from utils.train_utils import get_model


def load_config(config_path: str) -> SimpleNamespace:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert nested dictionaries to SimpleNamespace
    for section in config:
        config[section] = SimpleNamespace(**config[section])

    return SimpleNamespace(**config)

def evaluate(eval_dataset, model_path, cfg, train_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(train_dataset, cfg)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    total, correct = 0, 0

    model.eval()
    with torch.no_grad():
        for example in eval_dataset:
            input_ids = example["input_ids"].unsqueeze(0).to(device)
            answer = example["answer"]

            generated_answer_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=cfg.eval_configs.max_answer_length,
            )

            generated_answer = []
            for token_id in generated_answer_ids[0].tolist()[input_ids.shape[1]:]:
                if token_id == eval_dataset.eos_token_id:
                    break
                generated_answer.append(eval_dataset.id_to_token[token_id])
            generated_answer = ''.join(generated_answer)

            correct += (generated_answer == answer)
            total += 1

    accuracy = correct / total if total else 0
    print(f"\nEvaluation Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    # get the config path from the script arguments
    config_path = sys.argv[1]

    cfg = load_config(config_path)

    cwd = os.getcwd()


    characters = set([chr(i) for i in range(256)])
    special_tokens = ['<PAD>', '<EOS>']
    vocab = special_tokens + sorted(characters)
    token_to_id = {char: idx for idx, char in enumerate(vocab)}
    id_to_token = {idx: char for idx, char in enumerate(vocab)}

    train_dataset = AdditionDataset(os.path.join(cwd,cfg.data_configs.train_data_path),
                                    id_to_token=id_to_token,
                                    token_to_id=token_to_id,
                                    max_sequence_length=cfg.data_configs.max_sequence_length,
                                    vocab=vocab)

    pad = False
    eval_dataset = EvalAdditionDataset(
        file_path=os.path.join(cfg.data_configs.test_data_path),
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        pad_token_id=train_dataset.pad_token_id,
        eos_token_id=train_dataset.eos_token_id,
        max_length=cfg.data_configs.max_sequence_length,
        pad=pad
    )

    model_path =os.path.join(cwd,cfg.eval_configs.save_path)

    evaluate(eval_dataset, model_path, cfg, train_dataset)
    print("Done.")
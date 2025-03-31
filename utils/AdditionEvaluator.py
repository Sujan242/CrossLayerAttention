import sys
from types import SimpleNamespace

import torch
import yaml

from model.SmallScaleLlama import LlamaWithAllLayerCrossAttention
from utils.AdditionDataset import AdditionDataset
from utils.AdditionDatasetEval import EvalAdditionDataset


def load_config(config_path: str) -> SimpleNamespace:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert nested dictionaries to SimpleNamespace
    for section in config:
        config[section] = SimpleNamespace(**config[section])

    return SimpleNamespace(**config)

def evaluate(eval_dataset, model_path, cfg, train_dataset):
    id_to_token = eval_dataset.id_to_token
    eos_token_id = eval_dataset.eos_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LlamaWithAllLayerCrossAttention(vocab_size=train_dataset.vocab_size,
                                            hidden_size=cfg.model_configs.hidden_size,
                                            num_attention_heads=cfg.model_configs.num_attention_heads,
                                            num_hidden_layers=cfg.model_configs.num_hidden_layers,
                                            attention_dropout=cfg.model_configs.attention_dropout,
                                            hidden_dropout=cfg.model_configs.hidden_dropout,
                                            ).to(device)
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
                eos_token_id=eval_dataset.eos_token_id,
            )

            generated_answer = []
            for token_id in generated_answer_ids:
                if token_id == eos_token_id:
                    break
                generated_answer.append(id_to_token[token_id.item()])
            generated_answer = ''.join(generated_answer)

            # write input,expected and generated into a csv file
            # if generated_answer!=answer:
            #     input_str = "".join([id_to_token[x] for x in input_ids.squeeze().tolist()])
            #     print(f"Input: {input_str}")
            #     print(f"Expected: {answer}")
            #     print(f"Generated: {generated_answer}")
            #     print()
            #     csv_list.append([input_str,answer,generated_answer])
            correct += (generated_answer == answer)
            total += 1

    accuracy = correct / total if total else 0
    print(f"\nEvaluation Accuracy: {accuracy:.4f}")

    import csv
    # write the csv file with headers
    with open('addition_eval_lg.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Input", "Expected", "Generated"])
        writer.writerows(csv_list)


if __name__ == "__main__":
    # get the config path from the script arguments
    config_path = sys.argv[1]

    cfg = load_config(config_path)

    train_dataset = AdditionDataset("/Users/Patron/PycharmProjects/CrossLayerAttention/data/addition/train_3digit_10000.txt",
                                    max_sequence_length=cfg.data_configs.max_sequence_length)
    eval_dataset = EvalAdditionDataset(
        file_path="/Users/Patron/PycharmProjects/CrossLayerAttention/data/addition/test_3digit_10000.txt",
        token_to_id=train_dataset.token_to_id,
        id_to_token=train_dataset.id_to_token,
        pad_token_id=train_dataset.pad_token_id,
        eos_token_id=train_dataset.eos_token_id
    )

    model_path =cfg.eval_configs.save_path

    evaluate(eval_dataset, "/Users/Patron/PycharmProjects/CrossLayerAttention/model_weights/addition_1000_small_full_reversed.pth", cfg, train_dataset)
    print("Done.")
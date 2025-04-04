from torch.utils.data import Dataset
import torch

class EvalAdditionDataset(Dataset):
    def __init__(self, file_path, token_to_id, id_to_token, pad_token_id, eos_token_id, max_length, pad=True):
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.max_length = max_length
        self.pad = pad

        self.lines = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                equation = line.strip()
                lhs = equation.split("=")[0]
                rhs = equation.split("=")[1]
                # the dataset already reverses the rhs
                # reversed_rhs = "".join(rhs[::-1])
                self.lines.append(f"{lhs}={rhs}")

        self.questions = []
        self.answers = []
        for line in self.lines:
            q, a = line.split('=', 1)
            self.questions.append(q + '=')
            self.answers.append(a)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]

        input_ids = [self.token_to_id[c] for c in question]

        # evaluating with batch size 1
        # attention_mask = [1] * len(input_ids)
        # padding_length = self.max_length - len(input_ids)
        # if self.pad:
        #     input_ids = [self.pad_token_id] * padding_length + input_ids
        # attention_mask = [0] * padding_length + attention_mask

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            # "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "answer": answer
        }
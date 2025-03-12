from torch.utils.data import Dataset
import torch

class EvalAdditionDataset(Dataset):
    def __init__(self, file_path, token_to_id, id_to_token, pad_token_id, eos_token_id):
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        self.lines = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                equation = line.strip()
                lhs = equation.split("=")[0]
                rhs = equation.split("=")[1]
                reversed_rhs = "".join(rhs[::-1])
                self.lines.append(f"{lhs}={reversed_rhs}")

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

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "answer": answer
        }
from torch.utils.data import Dataset
import torch
import os


class AdditionDataset(Dataset):
    def __init__(self, file_path, max_sequence_length=512, token_to_id=None, id_to_token=None):
        self.max_length = max_sequence_length
        self.lines = []
        # Read and process data
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                equation = line.strip()
                lhs = equation.split("=")[0]
                rhs = equation.split("=")[1]
                reversed_rhs = "".join(rhs[::-1])
                self.lines.append(f"{lhs}={reversed_rhs}")

        # Create vocabulary
        self.characters = set()
        for line in self.lines:
            self.characters.update(line)

        # Add special tokens
        self.special_tokens = ['<PAD>', '<EOS>']
        self.vocab = self.special_tokens + sorted(self.characters)
        if token_to_id is None:
            self.token_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        else:
            self.token_to_id = token_to_id

        if id_to_token is None:
            self.id_to_token = {idx: char for idx, char in enumerate(self.vocab)}
        else:
            self.id_to_token = id_to_token

        # Set token IDs
        self.pad_token_id = self.token_to_id['<PAD>']
        self.eos_token_id = self.token_to_id['<EOS>']
        self.vocab_size = len(self.vocab)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx]

        # Convert text to character IDs
        char_ids = [self.token_to_id[c] for c in text]

        # Add EOS token
        char_ids.append(self.eos_token_id)

        # Truncate if needed
        if len(char_ids) > self.max_length:
            char_ids = char_ids[:self.max_length]

        # Create attention mask before padding
        # attention_mask = [1] * (len(char_ids)+self.num_hidden_layers-1)
        attention_mask = [1] * len(char_ids)
        # Pad sequence
        padding_length = self.max_length - len(char_ids)
        input_ids = char_ids + [self.pad_token_id] * padding_length
        attention_mask += [0] * padding_length

        # For causal LM, labels are same as input_ids but shifted
        # We'll handle shifting in the model
        labels = torch.tensor(input_ids.copy(), dtype=torch.long)
        labels[labels == self.pad_token_id] = -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": labels
        }


# Example usage
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get directory of this script
    data_dir = os.path.join(script_dir, "..", "data")  # Go up and into data
    file_path = os.path.join(data_dir, "addition/train_3digit_10000.txt")

    dataset = AdditionDataset(file_path, max_sequence_length=20)

    # Show vocabulary information
    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"Sample token mapping: '7' -> {dataset.token_to_id.get('7', '<UNK>')}")

    # Show first sample
    sample = dataset[0]
    print("\nSample input IDs:", sample['input_ids'])
    print("Sample attention mask:", sample['attention_mask'])
    print("Sample labels:", sample['labels'])
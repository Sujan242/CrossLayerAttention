from transformers import Trainer, TrainingArguments

from utils.AdditionDataset import AdditionDataset
from model.SmallScaleLlama import CustomLlama

def train(dataset, model):

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        num_train_epochs=3,
        logging_dir="./logs",
        remove_unused_columns=False  # Preserve attention_mask
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()

if __name__ == "__main__":
    dataset = AdditionDataset("data/addition/train_3digit_10000.txt", max_sequence_length=20)

    model = CustomLlama(vocab_size=dataset.vocab_size, hidden_size=500, num_attention_heads=5, num_hidden_layers=4)


    train(dataset, model)
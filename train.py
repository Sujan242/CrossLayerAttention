from transformers import Trainer, TrainingArguments

from utils.AdditionDataset import AdditionDataset
from utils.AdditionDatasetEval import EvalAdditionDataset
from utils.AdditionEvalCallback import AdditionEvalCallback
from model.SmallScaleLlama import CustomLlama

def train(train_dataset, model, eval_callback):

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        num_train_epochs=3,
        logging_dir="./logs",
        remove_unused_columns=False  # Preserve attention_mask
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[eval_callback]
    )

    trainer.train()

if __name__ == "__main__":
    train_dataset = AdditionDataset("data/addition/train_3digit_10000.txt", max_sequence_length=20)
    eval_dataset = EvalAdditionDataset(
        file_path="data/addition/test_3digit_10000.txt",
        max_sequence_length=20,
        token_to_id=train_dataset.token_to_id,
        id_to_token=train_dataset.id_to_token,
        pad_token_id=train_dataset.pad_token_id,
        eos_token_id=train_dataset.eos_token_id
    )

    model = CustomLlama(vocab_size=train_dataset.vocab_size, hidden_size=500, num_attention_heads=5, num_hidden_layers=4)

    eval_callback = AdditionEvalCallback(eval_dataset, max_answer_length=5, num_hidden_layers=4)

    train(train_dataset, model, eval_callback)
from transformers import TrainerCallback
import torch
from torch.utils.data import DataLoader


class AdditionEvalCallbackActual(TrainerCallback):
    def __init__(self, eval_dataset, max_answer_length=5, eval_interval=100, save_path="model.pth", batch_size=32):
        self.eval_dataset = eval_dataset
        self.max_answer_length = max_answer_length
        self.token_to_id = eval_dataset.token_to_id
        self.id_to_token = eval_dataset.id_to_token
        self.pad_token_id = eval_dataset.pad_token_id
        self.eos_token_id = eval_dataset.eos_token_id
        self.eval_interval = eval_interval
        self.save_path = save_path
        self.best_accuracy = 0
        self.batch_size = batch_size

    def on_epoch_end(self, args, state, control, **kwargs):
        if (1 + state.epoch) % self.eval_interval != 0:
            return control
        model = kwargs['model'].module if hasattr(kwargs['model'], 'module') else kwargs['model']
        device = next(model.parameters()).device
        total, correct = 0, 0

        model.eval()
        dataloader = DataLoader(self.eval_dataset, batch_size=self.batch_size)

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                answers = batch["answer"]
                attention_mask = batch["attention_mask"].to(device)

                generated_answer_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_answer_length,
                )

                for i in range(len(answers)):
                    generated_answer = []
                    for token_id in generated_answer_ids[i].tolist()[input_ids.shape[1]:]:
                        if token_id == self.eos_token_id:
                            break
                        generated_answer.append(self.id_to_token[token_id])
                    generated_answer = ''.join(generated_answer)

                    correct += (generated_answer == answers[i])
                    total += 1

        accuracy = correct / total if total else 0
        print(f"\nEvaluation Accuracy: {accuracy:.4f}")

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            print(f"Bettered best accuracy with {accuracy}. Saving model.")
            torch.save(model.state_dict(), self.save_path.replace(".pth", "_best.pth"))

        torch.save(model.state_dict(), self.save_path)

        print("______________________________________________________________________________________")
        print("Best accuracy so far: ", self.best_accuracy)
        print("Current accuracy: ", accuracy)
        print("______________________________________________________________________________________")

        if state.log_history:
            state.log_history[-1]['eval_accuracy'] = accuracy

        return control
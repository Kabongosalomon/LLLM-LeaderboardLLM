from transformers import TrainerCallback, Trainer, TrainingArguments
import wandb

class CustomWandbCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Custom logs you want to add
        custom_logs = {
            "custom_metric": 123.45,
            # ... any other custom data
        }
        wandb.log(custom_logs)  # Log the custom data to wandb

training_args = TrainingArguments(
    # ... your usual TrainingArguments settings ...
    report_to='wandb',
)

trainer = Trainer(
    # ... your usual Trainer settings ...
    args=training_args,
    callbacks=[CustomWandbCallback()],
)

trainer.train()

wandb.log({"custom_metric": 123.45})

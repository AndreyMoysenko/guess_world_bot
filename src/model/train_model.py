from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_from_disk
import numpy as np
import wandb

# Initialize tokenizer and base model
model_name = "ai-forever/ruT5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
)

# Wrap the model with LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # print for debugging purposes


# Define the exact match (EM) metric
def exact_match(eval_pred):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    hits, total = 0.0, 0
    for pred, label in zip(decoded_preds, decoded_labels):
        total += 1
        if pred.strip().lower() == label:
            hits += 1.0

    return {"em": hits / total}


def train_model(data_dir="./data/tokenized_dataset", model_dir="./models/final_model"):
    """Train a Seq2Seq model using LoRA and Seq2SeqTrainer."""
    wandb.init(project="ru-word-games", name="train-transformer-with-lora")

    # Load tokenized dataset
    tokenized_dataset = load_from_disk(data_dir)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="lsml_model",
        overwrite_output_dir=True,
        metric_for_best_model="em",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        weight_decay=0.1,
        warmup_ratio=0.01,
        num_train_epochs=8,
        save_total_limit=1,
        fp16=True,
        predict_with_generate=True,
        report_to="wandb",
    )

    # Define Seq2SeqTrainer
    data_collator = DataCollatorForSeq2Seq(tokenizer)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=exact_match,
    )

    # Train the model
    trainer.train()

    # Save the trained model
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Model saved to {model_dir}")


if __name__ == "__main__":
    train_model()

from celery import Celery
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel, PeftConfig

# Initialize Celery
celery = Celery(
    "tasks",
    backend="redis://redis:6379/0",
    broker="redis://redis:6379/0",
)

# Load the model and tokenizer
model_dir = "/app/models/final_model"  # Use absolute path
peft_config = PeftConfig.from_pretrained(model_dir)
base_model = T5ForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path
)
model = PeftModel.from_pretrained(base_model, model_dir)
model = model.merge_and_unload()
tokenizer = T5Tokenizer.from_pretrained(peft_config.base_model_name_or_path)


@celery.task(name="tasks.generate_predictions_task")
def generate_predictions_task(input_text):
    """Asynchronous task for generating predictions."""
    # Tokenize input
    inputs = tokenizer(
        input_text, return_tensors="pt", max_length=64, truncation=True, padding=True
    )

    # Generate predictions
    outputs = model.generate(
        **inputs,
        max_length=8,
        num_beams=5,
        num_return_sequences=5,
        early_stopping=True,
    )

    # Decode predictions
    predictions = [
        tokenizer.decode(output, skip_special_tokens=True) for output in outputs
    ]
    return predictions

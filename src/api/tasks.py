from celery import Celery
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Initialize Celery
celery = Celery(
    "tasks",
    backend="redis://localhost:6379/0",
    broker="redis://localhost:6379/0",
)

# Load the model and tokenizer
model_dir = "../../models/final_model"
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)


@celery.task
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

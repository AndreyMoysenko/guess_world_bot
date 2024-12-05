import pymorphy3
from datasets import load_dataset
from transformers import T5Tokenizer

# Model name for the tokenizer
model_name = "ai-forever/ruT5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Prefix for task
prefix = "guess word:"


# Function to clean and format ellipses
def fix_ellipses(s):
    s = s.replace("....", "...")
    s = s.replace("(?)", "...")
    s = s.replace("…", "...")
    s = s.replace("...", " ... ")

    return " ".join(s.split())


# Function to preprocess data
def preprocess_data(examples):
    prompts = []

    for prompt in examples["prompt"]:
        prompt = fix_ellipses(prompt)
        prompt = prompt.replace("...", "")
        prompt = f"{prefix} {prompt}"
        prompts.append(prompt)

    labels = examples["answer"]

    model_inputs = tokenizer(prompts, max_length=64, padding=True, truncation=True)
    labels = tokenizer(labels, max_length=8, padding=True, truncation=True)

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


# Check if a word is a noun
def check_word_is_noun(word):
    morph = pymorphy3.MorphAnalyzer()
    p = morph.parse(word)[0]
    return p.tag.POS == "NOUN"


# Main dataset preparation function
def prepare_dataset(output_dir="./data/tokenized_dataset"):
    """Load, preprocess, and tokenize the dataset."""
    # Load dataset
    dataset = load_dataset("artemsnegirev/ru-word-games", split="train")
    subsets = ["350_zagadok", "ostrova", "ugadaj_slova", "umnyasha"]

    dataset = dataset.filter(lambda x: x["subset"] in subsets)
    dataset = dataset.filter(lambda x: check_word_is_noun(x["answer"]))

    # Class encode and split
    dataset = dataset.class_encode_column("subset")
    dataset = dataset.train_test_split(test_size=0.1, stratify_by_column="subset")

    # Apply preprocessing
    tokenized_dataset = dataset.map(preprocess_data, batched=True)
    tokenized_dataset.set_format(type="torch")

    # Save tokenized dataset
    tokenized_dataset.save_to_disk(output_dir)
    print(f"Tokenized dataset saved to {output_dir}")
    return tokenized_dataset


if __name__ == "__main__":
    prepare_dataset()

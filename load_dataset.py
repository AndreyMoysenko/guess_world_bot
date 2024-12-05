from datasets import load_dataset
import pymorphy3

morph = pymorphy3.MorphAnalyzer()

def check_word_is_noun(w):
    p = morph.parse(w)[0]
    return p.tag.POS == 'NOUN'

# Load the dataset
dataset = load_dataset("artemsnegirev/ru-word-games")
subsets = ["350_zagadok", "ostrova", "ugadaj_slova", "umnyasha"]

# Filter the dataset
dataset = dataset.filter(lambda x: x["subset"] in subsets)
dataset = dataset.filter(lambda x: check_word_is_noun(x["answer"]))

# Class encode and split
dataset = dataset.class_encode_column("subset")
dataset = dataset["train"].train_test_split(test_size=0.1, stratify_by_column="subset")

print(f"Train size: {len(dataset['train'])}")
print(f"Test size: {len(dataset['test'])}")
print(dataset)
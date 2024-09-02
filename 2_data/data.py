import re
from importlib.metadata import version
from simple_tokenizer_v1 import SimpleTokenizerV1

print("torch version: ", version("torch"))
print("tiktoken version: ", version("tiktoken"))

# Read the Data
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total Text: ", len(raw_text))
print("Initial few sentences: ", raw_text[:99])

# Tokenize
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
print(preprocessed[:38])
print("Length of tokens: ", len(preprocessed))

# Vocabulary
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print("Vocab Size: ", vocab_size)

vocab = {token: integer for integer, token in enumerate(all_words)}

tokenizer = SimpleTokenizerV1(vocab=vocab)
sample_text = "Poor Jack! It had always been his fate to have women say such things of him"
print("DEBUG: sample_text: ", sample_text)
ids = tokenizer.encode(text=sample_text)
print("DEBUG: Encode: ", ids)
original_text = tokenizer.decode(ids=ids)
print("DEBUG: Decode: ", original_text)

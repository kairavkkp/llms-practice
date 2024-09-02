import re
import tiktoken
from importlib.metadata import version
from simple_tokenizer_v1 import SimpleTokenizerV1
from supplementary import create_dataloader_v1

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

# BytePair Encoding
tokenizer = tiktoken.get_encoding("gpt2")
text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print("DEBUG: Integers from tiktoken: ", integers)

# Data Sampling with Sliding Window
dataloader = create_dataloader_v1(text=raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("Targets:\n", targets)

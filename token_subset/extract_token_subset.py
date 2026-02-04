"""
Extract a subset of tokenizer token IDs corresponding to numeric strings (0-999)
and a small set of delimiter/special characters, then save to JSON.
"""

from transformers import AutoTokenizer
import json

# Add punctuation and special characters
special_tokens = [",", ";", ".", "_", " ", "-"]

MODEL_NAME = 'meta-llama/Llama-3.2-1B'

# Load the llama tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Collect token ids for all numbers 0-999 (as strings)
number_token_ids = {}

for i in range(1000):
    num_str = str(i)
    token_ids = tokenizer.encode(num_str, add_special_tokens=False)
    # Only keep those that correspond to a single token
    if len(token_ids) == 1:
        number_token_ids[num_str] = token_ids[0]

print(f"Found {len(number_token_ids)} numbers that are single tokens")
print(list(number_token_ids.items())[:20])  # show sample

for s in special_tokens:
    token_ids = tokenizer.encode(s, add_special_tokens=False)
    if len(token_ids) == 1:
        number_token_ids[s] = token_ids[0]
    else:
        print(f"Warning: '{s}' is not a single token, tokenized as {token_ids}")

# Save to JSON
with open("llama3-2-1B_number_tokens.json", "w") as f:
    json.dump(number_token_ids, f, indent=2)

# sorted_ids = sorted(number_token_ids.values())
# print(sorted_ids[:10], sorted_ids[-10:])

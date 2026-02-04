"""
Generate Gaussian integer sequences, clip to token range, and save to JSONL.
Each line: {"sequence_id": "...", "sequence_content": "..."}
"""

import numpy as np
from pathlib import Path
import json
import argparse
import os

DELIMITER = ','
base_dir = os.path.join(os.getcwd(), "data", "sequences")

RANDOM_SEED = 18
LOWER_BOUND = 10
UPPER_BOUND = 990

def generate_gaussian_numerical_sequences(
    mean: int,
    std: int,
    num_sequences: int,
    len_sequences: int,
    seed: int=RANDOM_SEED
) -> np.ndarray:
    """generates normally distributed random numbers, rounds, and clips"""
    rng = np.random.default_rng(seed) 
    # generate random sequence
    samples = rng.normal(loc=mean, scale=std, size=(num_sequences, len_sequences))
    # round and pass as int (probably enough to do .astype(int))
    samples = np.round(samples).astype(int)
    # clips to maintain within token range
    samples = np.clip(samples, LOWER_BOUND, UPPER_BOUND)
    return samples


def numerical_sequences_to_textual_prompts(samples: np.ndarray, delimiter: str) -> str:
    """converts numerical sequence to delimited texts, eg: [512 324 18] to `512,324,18,`."""
    prompts = []
    for sample in samples:
        prompt = delimiter.join(map(str, sample)) + delimiter # adds a last delimiter at the end
        prompts.append(prompt)
    return prompts


def main():
    """writes to JSONL"""
    parser = argparse.ArgumentParser(description="Generate Gaussian integer sequences and save to JSONL.")
    parser.add_argument("--num-seq", type=int, default=10, help="Number of sequences to generate.")
    parser.add_argument("--len-seq", type=int, default=1000, help="Length of each sequence.")
    parser.add_argument("--mean", type=int, default=500, help="Mean of the Gaussian.")
    parser.add_argument("--std", type=int, default=100, help="Std dev of the Gaussian.")
    args = parser.parse_args()

    dataset_name = f"gaussian_m{args.mean}_s{args.std:03d}_l{args.len_seq}_n{args.num_seq}"
    jsonl_path = Path(base_dir) / f"{dataset_name}.jsonl"

    samples = generate_gaussian_numerical_sequences(args.mean, args.std, args.num_seq, args.len_seq)
    prompts = numerical_sequences_to_textual_prompts(samples, DELIMITER)

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for p, prompt in enumerate(prompts):
            print(prompt[:50])
            record = {
                "sequence_id": f"seq_{p:03d}",
                "sequence_content": prompt,
            }
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()









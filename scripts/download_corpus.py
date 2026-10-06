#!/usr/bin/env python3
"""
Download and sample a parallel corpus from Hugging Face Datasets.

- Downloads the specified dataset (default: cfilt/iitb-english-hindi), train split only.
- Samples N sentence pairs (default: 10,000; -1 means all).
- Saves to data/raw/corpus_sample_<dataset>.jsonl, each line: {"en": ..., "hi": ...}
"""

import argparse
import os
from pathlib import Path
from datasets import load_dataset
import json

def sanitize_dataset_name(name):
    # Replace characters unsuitable for filenames
    return name.replace("/", "-")

def main():
    parser = argparse.ArgumentParser(description="Download and sample a parallel corpus from Hugging Face Datasets.")
    parser.add_argument("--dataset", type=str, default="cfilt/iitb-english-hindi",
                        help="Hugging Face dataset name (default: cfilt/iitb-english-hindi)")
    parser.add_argument("--sample-size", type=int, default=10000,
                        help="Number of sentence pairs to sample (default: 10000, -1 for all)")
    args = parser.parse_args()

    dataset_name = args.dataset
    sample_size = args.sample_size
    dataset_name_safe = sanitize_dataset_name(dataset_name)

    # Prepare output path
    script_dir = Path(__file__).resolve().parent
    out_dir = (script_dir.parent / "data" / "raw").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"corpus_sample_{dataset_name_safe}_{str(sample_size)}.jsonl"

    print(f"📥 Loading dataset '{dataset_name}' (train split)...")
    try:
        ds = load_dataset(dataset_name, split="train")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    n_rows = len(ds)
    print(f"✅ Loaded {n_rows} rows from train split.")

    # Determine sample indices
    if sample_size == -1 or sample_size >= n_rows:
        sample_indices = list(range(n_rows))
        print(f"ℹ️ Using all {n_rows} rows.")
    else:
        # Shuffle and select sample_size indices
        import random
        sample_indices = random.sample(range(n_rows), sample_size)
        print(f"ℹ️ Sampling {sample_size} rows.")

    # Write to JSONL
    with out_path.open("w", encoding="utf-8") as f:
        for idx in sample_indices:
            item = ds[idx]
            # Expecting item["translation"] = {"en": ..., "hi": ...}
            translation = item.get("translation", {})
            en = translation.get("en", "").strip()
            hi = translation.get("hi", "").strip()
            if en and hi:
                json.dump({"en": en, "hi": hi}, f, ensure_ascii=False)
                f.write("\n")
    print(f"✅ Sample written to {out_path} ({len(sample_indices)} lines)")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Explore and visualize a sampled parallel corpus.

- Reads a JSONL file of {"en": ..., "hi": ...} lines.
- Computes word count statistics, length buckets, and correlation.
- Saves visualizations and analysis files in data/raw/.
"""

import argparse
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import json

def sanitize_dataset_name(name):
    return name.replace("/", "-")

def main():
    parser = argparse.ArgumentParser(description="Explore and visualize a sampled parallel corpus.")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to sample JSONL (required)")
    args = parser.parse_args()

    input_path = args.input
    input_filename = Path(input_path).stem  # e.g., "corpus_sample_cfilt-iitb-english-hindi"
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read JSONL into Polars DataFrame
    print(f"📖 Reading sample from {input_path}")
    df = pl.read_ndjson(input_path)

    # Compute word counts
    df = df.with_columns([
        pl.col("en").str.split(" ").list.len().alias("en_word_count"),
        pl.col("hi").str.split(" ").list.len().alias("hi_word_count")
    ])

    print(f"📊 Sample shape: {df.shape}")
    print(f"📊 Columns: {list(df.columns)}")

    # Word count stats
    en_stats = df["en_word_count"].describe()
    hi_stats = df["hi_word_count"].describe()
    print("\n📈 English word count stats:\n", en_stats)
    print("\n📈 Hindi word count stats:\n", hi_stats)

    # Correlation
    corr = df.select(pl.corr("en_word_count", "hi_word_count")).item()
    print(f"\n🔗 Correlation (en vs hi word count): {corr:.3f}")

    # Length buckets
    bins = [0, 10, 15, 20, 25, 30, 100]
    labels = ["≤0", "1-10", "11-15", "16-20", "21-25", "26-30", "31-100", ">100"]
    df = df.with_columns(
        pl.col("en_word_count").cut(bins, labels=labels).alias("length_bucket")
    )
    length_counts = df.group_by("length_bucket").len().sort("length_bucket")
    length_counts_dict = {row[0]: row[1] for row in length_counts.rows()}

    # Save stats
    stats = {
        "english_stats": {key: value.to_list() for key, value in en_stats.to_dict().items()},
        "hindi_stats": {key: value.to_list() for key, value in hi_stats.to_dict().items()},
        "correlation": corr,
        "length_bucket_counts": length_counts_dict
    }
    stats_path = out_dir / f"length_analysis_{input_filename}.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"✅ Length analysis saved to {stats_path}")

    # Bucket analysis (5-30 words)
    buckets = [(5, 10), (11, 15), (16, 20), (21, 25), (26, 30)]
    bucket_analysis = {}
    for min_len, max_len in buckets:
        mask = (pl.col("en_word_count") >= min_len) & (pl.col("en_word_count") <= max_len)
        bucket_df = df.filter(mask)
        count = bucket_df.height
        hi_lengths = bucket_df["hi_word_count"]
        bucket_info = {
            "english_range": f"{min_len}-{max_len}",
            "count": int(count),
            "hindi_length_mean": float(hi_lengths.mean()) if count > 0 else 0,
            "hindi_length_std": float(hi_lengths.std()) if count > 0 else 0,
            "length_ratio_mean": float((hi_lengths / bucket_df["en_word_count"]).mean()) if count > 0 else 0
        }
        bucket_analysis[f"bucket_{min_len}_{max_len}"] = bucket_info

    bucket_path = out_dir / f"bucket_analysis_{input_filename}.json"
    with open(bucket_path, "w", encoding="utf-8") as f:
        json.dump(bucket_analysis, f, indent=2, ensure_ascii=False)
    print(f"✅ Bucket analysis saved to {bucket_path}")

    # Visualization
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 2, 1)
    sns.histplot(df["en_word_count"], bins=50, color="blue", kde=False)
    plt.title("English Sentence Length Distribution")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")

    plt.subplot(2, 2, 2)
    sns.histplot(df["hi_word_count"], bins=50, color="orange", kde=False)
    plt.title("Hindi Sentence Length Distribution")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")

    plt.subplot(2, 2, 3)
    plt.scatter(df["en_word_count"], df["hi_word_count"], alpha=0.3)
    plt.title("English vs Hindi Length Correlation")
    plt.xlabel("English Word Count")
    plt.ylabel("Hindi Word Count")

    plt.subplot(2, 2, 4)
    plt.bar(list(length_counts_dict.keys()), list(length_counts_dict.values()))
    plt.title("Sentences by Length Buckets (English)")
    plt.xlabel("Length Bucket")
    plt.ylabel("Count")
    plt.xticks(rotation=45)

    plt.tight_layout()
    viz_path = out_dir / f"dataset_exploration_{input_filename}.png"
    plt.savefig(viz_path, dpi=300, bbox_inches="tight")
    print(f"✅ Visualization saved to {viz_path}")

if __name__ == "__main__":
    main()

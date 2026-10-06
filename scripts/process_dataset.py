#!/usr/bin/env python3
"""
Dataset processing script to augment JSONL file with TTS/STT pipeline keys.
"""

import json
import yaml
import polars as pl
from pathlib import Path
from typing import Dict, List, Any
import uuid
from collections import defaultdict
import random

def load_config(config_path: str = "config/models.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_word_count_bin(text: str, bin_ranges: List[List[int]]) -> str:
    """
    Determine the bin for a text based on word count.
    Uses English text for binning as specified.
    """
    word_count = len(text.split())
    
    for i, (start, end) in enumerate(bin_ranges):
        if start <= word_count < end:
            return f"{start}-{end-1}"
    
    # Handle edge case for texts longer than max bin
    max_range = bin_ranges[-1]
    return f"{max_range[0]}+"


def assign_models_to_records(
    paired_data: List[Dict], 
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Create cross-product of TTS/STT models for each language and sentence.
    """
    results = []
    language_models = config['experiment_config']['language_models']
    
    for pair in paired_data:
        en_data = pair['en']
        hi_data = pair['hi']
        
        # Process both languages
        for lang_data in [en_data, hi_data]:
            lang = lang_data['lang']
            
            # Get TTS and STT models for this language
            tts_models = language_models[lang]['tts']
            stt_models = language_models[lang]['stt']
            
            # Create cross-product of TTS and STT models
            for tts_model in tts_models:
                for stt_model in stt_models:
                    # Create augmented record with assigned models
                    augmented_record = {
                        "id": lang_data['id'],
                        "lang": lang_data['lang'],
                        "text": lang_data['text'],
                        "bin": lang_data['bin'],
                        "len_en": lang_data['len_en'],
                        "len_hi": lang_data['len_hi'],
                        "tts_model_name": tts_model,
                        "stt_model_name": stt_model
                    }
                    
                    results.append(augmented_record)
    
    return results


def pair_translations(input_file: str) -> List[Dict]:
    """
    Group English-Hindi translation pairs by ID.
    """
    pairs = defaultdict(dict)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            pairs[record['id']][record['lang']] = record
    
    # Convert to list of paired data
    paired_data = []
    for pair_id, languages in pairs.items():
        if 'en' in languages and 'hi' in languages:
            paired_data.append({
                'id': pair_id,
                'en': languages['en'],
                'hi': languages['hi']
            })
    
    return paired_data


def sample_data_by_bin(
    paired_data: List[Dict], 
    samples_per_bucket: int = 100
) -> List[Dict]:
    """
    Sample data evenly across bins to get specified number per bucket.
    """
    bins = defaultdict(list)
    
    # Group by bin
    for pair in paired_data:
        bin_key = pair['en']['bin']  # Use English bin
        bins[bin_key].append(pair)
    
    # Sample from each bin
    sampled_data = []
    for bin_key, bin_data in bins.items():
        if len(bin_data) >= samples_per_bucket:
            sampled = bin_data[:samples_per_bucket]
        else:
            sampled = bin_data
            print(f"Warning: Bin {bin_key} has only {len(bin_data)} samples, less than requested {samples_per_bucket}")
        
        sampled_data.extend(sampled)
    
    return sampled_data


def process_dataset(input_file: str, output_file: str, config_path: str = "config/models.yaml"):
    """
    Main processing function to augment dataset with TTS/STT pipeline keys.
    """
    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    print(f"Reading input file: {input_file}")
    paired_data = pair_translations(input_file)
    print(f"Found {len(paired_data)} translation pairs")
    
    # Sample data by bins
    samples_per_bucket = config['experiment_config']['samples_per_bucket']
    print(f"Sampling {samples_per_bucket} samples per bucket")
    sampled_data = sample_data_by_bin(paired_data, samples_per_bucket)
    print(f"Sampled {len(sampled_data)} pairs")
    
    # Assign models to records
    print("Assigning TTS/STT models to records...")
    augmented_records = assign_models_to_records(sampled_data, config)
    print(f"Generated {len(augmented_records)} records with assigned models")
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create audio output directory
    audio_dir = Path(config['paths']['audio_output_dir'])
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Write augmented dataset
    print(f"Writing augmented dataset to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in augmented_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print("Dataset processing completed!")
    
    # Print statistics
    stats = defaultdict(int)
    for record in augmented_records:
        stats[f"{record['lang']}_{record['tts_model_name']}_{record['stt_model_name']}"] += 1
    
    print("\nExperiment Statistics:")
    for combo, count in stats.items():
        print(f"  {combo}: {count} samples")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process dataset for TTS-STT pipeline")
    parser.add_argument("input_file", help="Input JSONL file")
    parser.add_argument("output_file", help="Output augmented JSONL file")
    parser.add_argument("--config", default="config/models.yaml", help="Configuration file path")
    
    args = parser.parse_args()
    
    process_dataset(args.input_file, args.output_file, args.config)
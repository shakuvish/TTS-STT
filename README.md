# TTS-STT Pipeline for Quality Evaluation

This project implements an end-to-end Text-to-Speech (TTS) to Speech-to-Text (STT) pipeline for evaluating TTS quality degradation with increasing sentence length using the IITB English-Hindi parallel corpus.

## Project Structure

```
.
├── config/
│   └── models.yaml           # Model configurations and experiment settings
├── data/
│   ├── raw/                  # Raw downloaded data and initial analysis
│   ├── processed/            # Processed datasets with model assignments
│   └── audio/                # Generated audio files from TTS
├── deploy/
│   ├── common.py             # Shared Modal infrastructure
│   ├── tts_service.py        # Cloud TTS services (Parler, IndicParle)
│   ├── stt_service.py        # Cloud STT services (Conformer, IndicConformer)
│   ├── local_tts.py          # Local TTS services (Gemini)
│   └── local_stt.py          # Local STT services (Gemini)
└── scripts/
    ├── setup_nlp_models.py   # Download required spaCy and Stanza models
    ├── download_corpus.py    # Download and sample the parallel corpus
    ├── explore_corpus.py     # Analyze and visualize the sampled corpus
    ├── process_dataset.py    # Create cross-product model assignments
    └── run_pipeline.py       # End-to-end pipeline orchestrator
```

## Getting Started

### 1. Install Dependencies

From the project root directory, install the required dependencies:

```bash
uv sync
```

This will install all dependencies listed in `pyproject.toml`, including:
- `datasets` for downloading Hugging Face datasets
- `polars` and `numpy` for data manipulation
- `matplotlib` and `seaborn` for visualization
- `spacy` and `stanza` for linguistic analysis
- `polars` for fast DataFrame operations (used in corpus exploration)
- Audio processing libraries for future TTS work

### 2. Setup NLP Models

Before running any data scripts, download the required spaCy and Stanza models:

```bash
python scripts/setup_nlp_models.py
```
Use the below command if faced with any issue with pip
```bash
uv run -m ensurepip --upgrade
```
This will ensure the English spaCy model and Hindi Stanza model are available for complexity analysis.

### 3. Download and Sample the Corpus

Run the download script:

```bash
python scripts/download_corpus.py
```

This script will:
- 📥 Download the CFILT IITB English-Hindi parallel corpus from Hugging Face
- 🔍 Perform initial data exploration and sampling
- 💾 Save a sample of the corpus and basic dataset info

### 4. Explore and Analyze the Corpus

Run the exploration script to generate statistics and visualizations:

```bash
python scripts/explore_corpus.py --input data/raw/corpus_sample_cfilt-iitb-english-hindi.jsonl
```

This script will:
- 📊 Compute word count statistics and length buckets
- 🔗 Analyze correlation between English and Hindi sentence lengths
- 📈 Generate and save visualizations
- 📝 Save detailed analysis files for further study

### 5. Process Dataset for Pipeline

Create cross-product model assignments for comprehensive evaluation:

```bash
python scripts/process_dataset.py data/raw/input_dataset.jsonl data/processed/sample_processed.jsonl
```

This creates all TTS/STT model combinations per language:
- **English**: parler + gemini_flash (TTS) × conformer + gemini_flash (STT) = 4 combinations
- **Hindi**: indic_parle + gemini_flash (TTS) × indic_conformer + gemini_flash (STT) = 4 combinations

### 6. Deploy Cloud Services

Deploy TTS and STT services to Modal cloud:

```bash
# Deploy TTS services (Parler, IndicParle)
modal deploy deploy/tts_service.py

# Deploy STT services (Conformer, IndicConformer)
modal deploy deploy/stt_service.py
```

### 7. Run Complete Pipeline

Execute the end-to-end TTS-STT pipeline:

```bash
python scripts/run_pipeline.py data/processed/sample_processed.jsonl data/results/pipeline_results.jsonl --output-dir data/audio
```

Optional flags:
- `--resume`: Skip already processed records
- `--output-dir`: Directory for audio files (default: data/audio)

### 8. Output Files

After running the complete pipeline, you'll find:

**Raw Data (`data/raw/`)**:
- `dataset_info.json` - Basic information about the downloaded dataset
- `corpus_sample.csv` - Sample of the dataset in CSV format for inspection
- `corpus_sample_cfilt-iitb-english-hindi.jsonl` - Sampled parallel corpus (JSONL)
- `dataset_exploration_cfilt-iitb-english-hindi.png` - Visualizations of length distributions and correlations
- `length_analysis_cfilt-iitb-english-hindi.json` - Statistical analysis of sentence lengths
- `bucket_analysis_cfilt-iitb-english-hindi.json` - Analysis of target buckets for the TTS study

**Processed Data (`data/processed/`)**:
- `sample_processed.jsonl` - Records with TTS/STT model assignments

**Audio Files (`data/audio/`)**:
- `{id}_{lang}_{tts_model}_{uuid}.mp3` - Generated audio files

**Results (`data/results/`)**:
- `pipeline_results.jsonl` - Complete pipeline results with:
  - Original text and metadata
  - TTS/STT model assignments
  - Generated audio file paths
  - STT transcriptions
  - Processing times and status

## TTS/STT Models

### Text-to-Speech (TTS) Models:
- **Parler TTS** (English): Hugging Face parler-tts-mini-v1.1 model
- **IndicParle** (Hindi): AI4Bharat's IndicParle model for Hindi synthesis
- **Gemini Flash TTS** (Both): Google's Gemini API for multilingual TTS

### Speech-to-Text (STT) Models:
- **SpeechBrain Conformer** (English): Conformer-based ASR for English
- **IndicConformer** (Hindi): AI4Bharat's Conformer model for Hindi ASR
- **Gemini Flash STT** (Both): Google's Gemini API for multilingual STT

## Pipeline Architecture

### Cloud Services (Modal):
- **GPU-accelerated** inference for Parler, IndicParle, Conformer, IndicConformer
- **Auto-scaling** and **fault-tolerant** deployment
- **Shared volumes** for model caching

### Local Services:
- **Gemini API** integration for TTS and STT
- **Audio format conversion** (PCM ↔ MP3)
- **Language-specific prompting**

### Pipeline Orchestrator:
- **Cross-product evaluation** of all TTS/STT combinations
- **Immediate result persistence** for fault tolerance
- **Resume functionality** for interrupted processing
- **Comprehensive error handling** and progress tracking

## Next Steps

After generating pipeline data:

1. **Evaluation Framework** - Implement WER, MCD, and automatic MOS evaluation
2. **Quality Analysis** - Analyze TTS quality degradation with sentence length
3. **Model Comparison** - Compare performance across different TTS/STT combinations
4. **Statistical Analysis** - Generate insights on length vs quality correlation

## Key Research Questions

This dataset will help us answer:
- How does TTS quality degrade as sentence length increases?
- Which TTS system is most robust to longer sentences?
- How do different complexity levels (simple/compound/complex) affect synthesis quality?
- What's the correlation between different evaluation metrics (WER, MCD, MOS)?

## Dataset Information

**Source**: [CFILT IITB English-Hindi Parallel Corpus](https://huggingface.co/datasets/cfilt/iitb-english-hindi)

**Paper**: "The IIT Bombay English-Hindi Parallel Corpus" by Kunchukuttan et al.

**Languages**: English ↔ Hindi

**Domain**: General parallel text from various sources

**Size**: ~1.5M sentence pairs

This corpus is ideal for our study because:
- High-quality parallel translations ensure fair comparison between languages
- Large size allows for robust sampling across length buckets
- Diverse content provides realistic linguistic complexity

## Script Overview

### Data Preparation:
- **scripts/setup_nlp_models.py**: Downloads and verifies required spaCy and Stanza models for English and Hindi
- **scripts/download_corpus.py**: Downloads the parallel corpus and creates a sample for analysis
- **scripts/explore_corpus.py**: Analyzes the sampled corpus, computes statistics, and generates visualizations

### Pipeline Processing:
- **scripts/process_dataset.py**: Creates cross-product TTS/STT model assignments for comprehensive evaluation
- **scripts/run_pipeline.py**: End-to-end orchestrator that processes records through TTS→STT pipeline

### Service Deployment:
- **deploy/tts_service.py**: Cloud TTS services (Parler, IndicParle) on Modal
- **deploy/stt_service.py**: Cloud STT services (Conformer, IndicConformer) on Modal  
- **deploy/local_tts.py**: Local Gemini TTS service
- **deploy/local_stt.py**: Local Gemini STT service
- **deploy/common.py**: Shared Modal infrastructure and dependencies

### Configuration:
- **config/models.yaml**: Central configuration for all TTS/STT models and experiment settings

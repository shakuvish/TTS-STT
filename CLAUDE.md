# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Natural Language Processing research project studying TTS (Text-to-Speech) quality degradation with increasing sentence length, using the IITB English-Hindi parallel corpus.

## Development Commands

```bash
# Install dependencies (using uv package manager)
uv sync

# Download required NLP models (spaCy and Stanza)
python setup_nlp_models.py

# Download the IITB corpus data
cd data && python download_corpus.py
```

## Code Architecture

### Technology Stack
- **Python 3.12+** with `uv` package manager
- **Data Processing**: Polars (preferred over pandas for performance), NumPy
- **NLP**: spaCy, Stanza, indic-nlp-library
- **Audio**: Librosa, Soundfile
- **Evaluation**: jiwer (Word Error Rate)
- **ML**: Hugging Face Datasets, DSPy

### Project Structure
- `data/raw/`: Downloaded corpus data
- `data/processed/`: Processed datasets
- `download_corpus.py`: Downloads IITB English-Hindi corpus
- `setup_nlp_models.py`: Downloads spaCy and Stanza models

### Research Workflow
1. **Data Processing**: Analyze IITB corpus for sentence length distribution
2. **Complexity Classification**: Categorize sentences (simple/compound/complex)
3. **Sampling**: Create balanced datasets for each length bucket (5-10, 11-15, 16-20, 21-25, 26-30 words)
4. **TTS Synthesis**: Generate audio using IndicTTS, IndicF5, Google TTS
5. **Evaluation**: Calculate WER, MCD, and automatic MOS scores

## Key Considerations

- Use Polars instead of pandas for data processing (already in dependencies)
- The project focuses on Hindi TTS quality evaluation
- Length buckets are predefined: 5-10, 11-15, 16-20, 21-25, 26-30 words
- Multiple TTS systems will be compared
- Evaluation metrics include WER (Word Error Rate), MCD (Mel Cepstral Distortion), and automatic MOS

## DEBUGGING  problems and solutions

1. 
problem: 
/home/naveen/projects/iasnlp25/.venv/bin/python3: No module named pip
Traceback (most recent call last):
  File "/home/naveen/projects/iasnlp25/setup_nlp_models.py", line 19, in install_spacy_model
    nlp = spacy.load("en_core_web_sm")
solution:
python3 -m ensurepip --upgrade

## Current Implementation Status

✅ **Complete**:
- Data download and exploration pipeline
- Cross-product model assignment generation
- Cloud service deployment (Modal) for GPU models
- Local service integration (Gemini TTS/STT)
- End-to-end pipeline orchestrator with fault tolerance
- Immediate result persistence and resume functionality

🔄 **In Progress**:
- Audio file generation and STT transcription
- Comprehensive error handling and recovery

⏳ **Planned**:
- WER calculation between original and transcribed text
- MCD (Mel Cepstral Distortion) evaluation
- Automatic MOS (Mean Opinion Score) calculation
- Statistical analysis of quality degradation vs sentence length
- Model comparison and performance benchmarking
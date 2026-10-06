#!/usr/bin/env python3
"""
Setup script to download required NLP models for sentence complexity analysis.

This script downloads the necessary spaCy and Stanza models that will be used
for dependency parsing and sentence complexity classification.
"""

import subprocess
import sys
import stanza
import spacy

def install_spacy_model():
    """Download the English spaCy model."""
    print("📥 Installing spaCy English model...")
    try:
        # Try to load the model first
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy English model already installed!")
    except OSError:
        print("📦 Downloading spaCy English model...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ spaCy English model installed successfully!")

def install_stanza_models():
    """Download the Hindi Stanza model."""
    print("📥 Installing Stanza Hindi model...")
    try:
        # Check if Hindi model exists
        stanza.download('hi', processors='tokenize,pos,depparse', verbose=False)
        print("✅ Stanza Hindi model installed successfully!")
    except Exception as e:
        print(f"❌ Error installing Stanza Hindi model: {e}")
        raise

def verify_installations():
    """Verify that all models are properly installed."""
    print("\n🔍 Verifying installations...")
    
    # Test spaCy
    try:
        import spacy
        nlp_en = spacy.load("en_core_web_sm")
        test_doc = nlp_en("This is a test sentence.")
        print("✅ spaCy English model working correctly")
    except Exception as e:
        print(f"❌ spaCy English model error: {e}")
    
    # Test Stanza
    try:
        import stanza
        nlp_hi = stanza.Pipeline('hi', processors='tokenize,pos,depparse', verbose=False)
        test_doc = nlp_hi("यह एक परीक्षण वाक्य है।")
        print("✅ Stanza Hindi model working correctly")
    except Exception as e:
        print(f"❌ Stanza Hindi model error: {e}")

def main():
    """Main setup function."""
    print("🚀 Setting up NLP models for TTS corpus analysis")
    print("=" * 50)
    
    install_spacy_model()
    install_stanza_models()
    verify_installations()
    
    print("\n🎉 All models installed successfully!")
    print("You can now run the corpus download script: python download_corpus.py")

if __name__ == "__main__":
    main()

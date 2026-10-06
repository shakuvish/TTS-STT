#!/usr/bin/env python3
"""
Local STT services for models that run locally (not on Modal cloud).
"""

import os
from typing import Optional
from pathlib import Path

from google import genai
from google.genai import types


class GeminiSTTService:
    """
    Local Gemini STT service using Google GenAI API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini STT service.
        
        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.
        """
        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set or api_key not provided")
        
        self.client = genai.Client(api_key=api_key)
    
    def transcribe_audio(
        self, 
        audio_data: bytes, 
        language: str = "en",
        audio_format: str = "mp3"
    ) -> str:
        """
        Transcribe audio using Gemini STT.
        
        Args:
            audio_data: Audio data as bytes
            language: Language code ('en' or 'hi')
            audio_format: Audio format ('mp3', 'wav', etc.)
            
        Returns:
            Transcribed text
        """
        # Set up language-specific transcription prompt
        if language == "hi":
            prompt = "Transcribe this Hindi audio clip."
        else:
            prompt = "Transcribe this English audio clip."
        
        # Determine MIME type
        mime_type = f"audio/{audio_format}"
        
        try:
            # Request transcription from Gemini
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    prompt,
                    types.Part.from_bytes(data=audio_data, mime_type=mime_type)
                ]
            )
            print("Prompt tokens:", response.usage_metadata.prompt_token_count)
            print("Output tokens:", response.usage_metadata.candidates_token_count)
            print("Total tokens:", response.usage_metadata.total_token_count)
            return response.text.strip()
            
        except Exception as e:
            print(f"Error transcribing audio with Gemini STT: {e}")
            return ""
    
    def transcribe_file(
        self, 
        audio_path: str, 
        language: str = "en"
    ) -> str:
        """
        Transcribe audio from file path.
        
        Args:
            audio_path: Path to audio file
            language: Language code ('en' or 'hi')
            
        Returns:
            Transcribed text
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Read audio file
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        
        # Get file extension for format detection
        audio_format = audio_path.suffix.lstrip('.').lower()
        if audio_format == '':
            audio_format = 'mp3'  # Default to MP3
        
        return self.transcribe_audio(audio_data, language, audio_format)


def create_stt_service(model_name: str) -> object:
    """
    Factory function to create STT service based on model name.
    
    Args:
        model_name: Name of the STT model ('gemini_flash', etc.)
        
    Returns:
        STT service instance
    """
    if model_name == "gemini_flash":
        return GeminiSTTService()
    else:
        raise ValueError(f"Unknown local STT model: {model_name}")


if __name__ == "__main__":
    # Test the Gemini STT service
    stt = GeminiSTTService()
    
    # Test with an existing audio file (if available)
    test_files = ["test_en.mp3", "test_hi.mp3", "output_hindi.mp3"]
    
    for test_file in test_files:
        if Path(test_file).exists():
            print(f"\nTesting {test_file}:")
            # Detect language from filename
            language = "hi" if "hi" in test_file.lower() or "hindi" in test_file.lower() else "en"
            
            try:
                transcription = stt.transcribe_file(test_file, language=language)
                print(f"Transcription: {transcription}")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print(f"Test file {test_file} not found, skipping...")
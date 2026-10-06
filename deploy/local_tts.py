#!/usr/bin/env python3
"""
Local TTS and STT services for models that run locally (not on Modal cloud).
"""

import os
import io
from typing import Optional
from pathlib import Path

from gtts import gTTS


class GeminiTTSService:
    """
    Local TTS service using Google Text-to-Speech (gTTS).
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize gTTS service.
        
        Args:
            api_key: Not used for gTTS (kept for compatibility)
        """
        # gTTS doesn't require API key - it uses Google Translate's TTS API
        pass
    
    def generate_speech(
        self, 
        text: str, 
        language: str = "en", 
        voice_name: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> bytes:
        """
        Generate speech from text using gTTS.
        
        Args:
            text: Text to convert to speech
            language: Language code ('en' or 'hi')
            voice_name: Not used for gTTS (kept for compatibility)
            output_path: Optional path to save MP3 file
            
        Returns:
            MP3 audio data as bytes
        """
        try:
            # Create gTTS instance
            tts = gTTS(text=text, lang=language, slow=False)
            
            # Generate audio in memory
            mp3_io = io.BytesIO()
            tts.write_to_fp(mp3_io)
            mp3_bytes = mp3_io.getvalue()
            
            # Save to file if path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(mp3_bytes)
            
            return mp3_bytes
            
        except Exception as e:
            print(f"Error generating speech with gTTS: {e}")
            return None


def create_tts_service(model_name: str) -> object:
    """
    Factory function to create TTS service based on model name.
    
    Args:
        model_name: Name of the TTS model ('gemini_flash', etc.)
        
    Returns:
        TTS service instance
    """
    if model_name == "gemini_flash":
        return GeminiTTSService()
    else:
        raise ValueError(f"Unknown local TTS model: {model_name}")


if __name__ == "__main__":
    # Test the gTTS service
    tts = GeminiTTSService()
    
    # Test English
    english_text = "Hello, this is a test of the gTTS text-to-speech service."
    audio_bytes = tts.generate_speech(english_text, language="en", output_path="test_en.mp3")
    if audio_bytes:
        print(f"Generated {len(audio_bytes)} bytes of English audio")
    
    # Test Hindi
    hindi_text = "यह जीटीटीएस टेक्स्ट-टू-स्पीच सेवा का परीक्षण है।"
    audio_bytes = tts.generate_speech(hindi_text, language="hi", output_path="test_hi.mp3")
    if audio_bytes:
        print(f"Generated {len(audio_bytes)} bytes of Hindi audio")
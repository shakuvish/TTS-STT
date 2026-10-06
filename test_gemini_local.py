#!/usr/bin/env python3
"""
Test script for local Gemini TTS and STT services.
Tests with one English and one Hindi sentence.
"""

import os
from pathlib import Path
from deploy.local_tts import GeminiTTSService
from deploy.local_stt import GeminiSTTService
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_gemini_services():
    """Test Gemini TTS and STT with sample sentences."""
    
    # Test sentences (exactly 10 words each)
    english_text = "The quick brown fox jumps over the lazy sleeping dog"  # 10 words
    hindi_text = "यह एक परीक्षण वाक्य है जो दस शब्दों में लिखा गया है"  # 10 words
    
    print("🧪 Testing Gemini TTS and STT Services")
    print("=" * 50)
    
    try:
        # Initialize services
        print("🔧 Initializing Gemini services...")
        tts_service = GeminiTTSService()
        stt_service = GeminiSTTService()
        print("✅ Services initialized successfully")
        
        # Test English TTS → STT
        print(f"\n📝 English Test: '{english_text}'")
        print("🔊 Generating English speech...")
        
        english_audio = tts_service.generate_speech(
            text=english_text, 
            language="en", 
            output_path="test_english.mp3"
        )
        
        if english_audio and len(english_audio) > 0:
            print(f"✅ Generated {len(english_audio)} bytes of English audio")
            
            print("🎧 Transcribing English audio...")
            english_transcription = stt_service.transcribe_audio(
                audio_data=english_audio,
                language="en"
            )
            
            if english_transcription:
                print(f"✅ English transcription: '{english_transcription}'")
                print(f"📊 Original vs Transcribed:")
                print(f"   Original:     '{english_text}'")
                print(f"   Transcribed:  '{english_transcription}'")
            else:
                print("❌ English transcription failed")
        else:
            print("❌ English TTS failed")
        
        # Test Hindi TTS → STT
        print(f"\n📝 Hindi Test: '{hindi_text}'")
        print("🔊 Generating Hindi speech...")
        
        hindi_audio = tts_service.generate_speech(
            text=hindi_text,
            language="hi",
            output_path="test_hindi.mp3"
        )
        
        if hindi_audio and len(hindi_audio) > 0:
            print(f"✅ Generated {len(hindi_audio)} bytes of Hindi audio")
            
            print("🎧 Transcribing Hindi audio...")
            hindi_transcription = stt_service.transcribe_audio(
                audio_data=hindi_audio,
                language="hi"
            )
            
            if hindi_transcription:
                print(f"✅ Hindi transcription: '{hindi_transcription}'")
                print(f"📊 Original vs Transcribed:")
                print(f"   Original:     '{hindi_text}'")
                print(f"   Transcribed:  '{hindi_transcription}'")
            else:
                print("❌ Hindi transcription failed")
        else:
            print("❌ Hindi TTS failed")
            
        print("\n🎉 Test completed!")
        print("🎵 Check test_english.mp3 and test_hindi.mp3 files")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("💡 Make sure GEMINI_API_KEY is set in your .env file")
        print("💡 Install system dependencies:")
        print("   sudo apt install ffmpeg libsndfile1 sox")


if __name__ == "__main__":
    test_gemini_services()
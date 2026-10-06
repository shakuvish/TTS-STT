# deploy/test.py
import sys
from pathlib import Path
import os
from datetime import datetime
import time

# sys.path.append(str(Path(__file__).resolve().parents[1]))

from deploy.common import app
from deploy import tts_service
from deploy import stt_service

# --- Test Data ---
HINDI_TEXT = "नमस्ते, यह एक परीक्षण है।"
DESCRIPTION = "A female voice speaks in a clear and calm tone."
# ---

@app.local_entrypoint()
def main():
    print("🚀 Starting TTS -> STT pipeline test...")
    total_start_time = time.monotonic()
    
    # Instantiate the remote classes
    tts_model = tts_service.IndicParlerModel()
    stt_model = stt_service.IndicConformerModel()

    # 1. --- Call the Text-to-Speech service ---
    print("\n[STEP 1/3] Synthesizing audio from text (TTS)...")
    tts_start_time = time.monotonic()
    try:
        # Call the 'generate' method on the class instance
        audio_bytes = tts_model.generate.remote(
            prompt=HINDI_TEXT,
            description=DESCRIPTION
        )
        tts_duration = time.monotonic() - tts_start_time
        print(f"   ✅ TTS synthesis successful in {tts_duration:.2f} seconds.")
    except Exception as e:
        print(f"   ❌ TTS synthesis failed: {e}")
        return

    # ... (Step 2: Save audio remains the same) ...

    # 3. --- Call the Speech-to-Text service ---
    print("\n[STEP 3/3] Transcribing audio back to text (STT)...")
    stt_start_time = time.monotonic()
    try:
        # Call the 'transcribe' method on the class instance
        result = stt_model.transcribe.remote(
            audio_bytes=audio_bytes,
            language='hi'
        )
        stt_duration = time.monotonic() - stt_start_time
        print(f"   ✅ STT transcription successful in {stt_duration:.2f} seconds.")
    except Exception as e:
        print(f"   ❌ STT transcription failed: {e}")
        return

    # ... (Display final results remains the same) ...
# deploy/test_batch.py
import sys
from pathlib import Path
import os
import time
import numpy as np

# This line is critical for making imports work correctly.
# sys.path.append(str(Path(__file__).resolve().parents[1]))

from deploy.common import app
from deploy import tts_service
from deploy import stt_service

# --- Test Configuration ---
HINDI_TEXT = "नमस्ते, यह एक परीक्षण है।"
DESCRIPTION = "A female voice speaks in a clear and calm tone."
NUM_ITERATIONS = 10
# ---

@app.local_entrypoint()
def main():
    print(f"🚀 Starting Batch Performance Test ({NUM_ITERATIONS} iterations)...")
    
    # Instantiate the remote classes once to warm them up
    print("Pre-warming models (first run will be slower)...")
    tts_model = tts_service.IndicParlerModel()
    stt_model = stt_service.IndicConformerModel()

    tts_durations = []
    stt_durations = []

    for i in range(NUM_ITERATIONS):
        print(f"\n--- Iteration {i + 1}/{NUM_ITERATIONS} ---")

        # --- Run TTS ---
        tts_start_time = time.monotonic()
        try:
            audio_bytes = tts_model.generate.remote(prompt=HINDI_TEXT, description=DESCRIPTION)
            tts_duration = time.monotonic() - tts_start_time
            tts_durations.append(tts_duration)
            print(f"   TTS successful in {tts_duration:.2f} seconds.")
        except Exception as e:
            print(f"   TTS failed on iteration {i+1}: {e}")
            continue # Skip to next iteration on failure

        # --- Run STT ---
        stt_start_time = time.monotonic()
        try:
            stt_model.transcribe.remote(audio_bytes=audio_bytes, language='hi')
            stt_duration = time.monotonic() - stt_start_time
            stt_durations.append(stt_duration)
            print(f"   STT successful in {stt_duration:.2f} seconds.")
        except Exception as e:
            print(f"   STT failed on iteration {i+1}: {e}")

    # --- Calculate and Display Statistics ---
    if not tts_durations or not stt_durations:
        print("\nNo successful runs to analyze. Exiting.")
        return

    # Convert to numpy arrays for easy calculation
    tts_times = np.array(tts_durations)
    stt_times = np.array(stt_durations)

    print("\n" + "="*60)
    print("📊 BATCH TEST RESULTS 📊")
    print("="*60)
    
    # Cold start (first run) vs Warm runs (all subsequent runs)
    print(f"Cold Start (1st run):")
    print(f"  - TTS: {tts_times[0]:.2f}s")
    print(f"  - STT: {stt_times[0]:.2f}s")
    
    if NUM_ITERATIONS > 1:
        warm_tts = tts_times[1:]
        warm_stt = stt_times[1:]
        print(f"\nWarm Runs (average of last {len(warm_tts)} runs):")
        print(f"  - TTS: Mean: {np.mean(warm_tts):.2f}s, Std: {np.std(warm_tts):.2f}s, Min: {np.min(warm_tts):.2f}s, Max: {np.max(warm_tts):.2f}s")
        print(f"  - STT: Mean: {np.mean(warm_stt):.2f}s, Std: {np.std(warm_stt):.2f}s, Min: {np.min(warm_stt):.2f}s, Max: {np.max(warm_stt):.2f}s")
    
    print("="*60)
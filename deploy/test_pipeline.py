# deploy/test_pipeline.py
"""
Extended TTS-STT pipeline using Modal for batch processing.
Processes JSONL input files through TTS → STT pipeline with configurable models.
"""

import sys
import json
import time
import uuid
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

from deploy.common import app
from deploy import tts_service
from deploy import stt_service
from deploy.local_tts import GeminiTTSService
from deploy.local_stt import GeminiSTTService
from dotenv import load_dotenv
load_dotenv()

class PipelineProcessor:
    """Handles TTS-STT pipeline processing with multiple model support."""
    
    def __init__(self, output_dir: str = "data/audio"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cloud models
        self.indic_parler_tts = tts_service.IndicParlerModel()
        self.parler_tts = tts_service.ParlerTTSModel()
        self.indic_conformer_stt = stt_service.IndicConformerModel()
        self.conformer_stt = stt_service.SpeechBrainConformerModel()
        
        # Initialize local models (disable in cloud environment)
        try:
            self.gemini_tts = GeminiTTSService()
            self.gemini_stt = GeminiSTTService()
            print("✅ Local Gemini services initialized")
        except Exception as e:
            print(f"⚠️ Local Gemini services disabled: {e}")
            self.gemini_tts = None
            self.gemini_stt = None
        
        # Performance tracking
        self.stats = {
            'total_processed': 0,
            'total_skipped': 0,
            'total_failed': 0,
            'tts_times': [],
            'stt_times': [],
            'total_times': []
        }
    
    def generate_tts_audio(self, text: str, tts_model: str, language: str) -> Optional[bytes]:
        """Generate audio using specified TTS model."""
        try:
            if tts_model == "gemini_flash":
                if self.gemini_tts is None:
                    raise Exception("Gemini TTS service not available")
                return self.gemini_tts.generate_speech(text, language=language)
            
            elif tts_model == "indic_parle":
                desc = "A clear, natural voice speaking in Hindi" if language == "hi" else "A clear, natural voice"
                return self.indic_parler_tts.generate.remote(prompt=text, description=desc)
            
            elif tts_model == "parler":
                desc = "A clear, neutral voice speaks at a moderate pace."
                return self.parler_tts.generate.remote(prompt=text, description=desc)
            
            else:
                raise ValueError(f"Unknown TTS model: {tts_model}")
                
        except Exception as e:
            print(f"❌ TTS Error ({tts_model}): {e}")
            return None
    
    def transcribe_audio(self, audio_bytes: bytes, stt_model: str, language: str) -> Optional[str]:
        """Transcribe audio using specified STT model."""
        try:
            if stt_model == "gemini_flash":
                if self.gemini_stt is None:
                    raise Exception("Gemini STT service not available")
                return self.gemini_stt.transcribe_audio(audio_data=audio_bytes, language=language)
            
            elif stt_model == "indic_conformer":
                result = self.indic_conformer_stt.transcribe.remote(audio_bytes=audio_bytes, language=language)
                return result.get("ctc_transcription", "")
            
            elif stt_model == "conformer":
                result = self.conformer_stt.transcribe.remote(audio_bytes=audio_bytes, language=language)
                return result.get("transcription", "")
            
            else:
                raise ValueError(f"Unknown STT model: {stt_model}")
                
        except Exception as e:
            print(f"❌ STT Error ({stt_model}): {e}")
            return None
    
    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single record through the TTS-STT pipeline."""
        start_time = time.monotonic()
        
        # Extract record data
        record_id = record["id"]
        language = record["lang"]
        text = record["text"]
        tts_model = record["tts_model_name"]
        stt_model = record["stt_model_name"]
        
        print(f"🔄 Processing {record_id} ({language}): {tts_model} → {stt_model}")
        
        # Generate unique audio filename
        audio_filename = f"{record_id}_{language}_{tts_model}_{uuid.uuid4().hex[:8]}.mp3"
        audio_filepath = self.output_dir / audio_filename
        
        # Step 1: TTS - Generate audio
        tts_start = time.monotonic()
        audio_bytes = self.generate_tts_audio(text, tts_model, language)
        tts_time = time.monotonic() - tts_start
        
        if audio_bytes is None or len(audio_bytes) == 0:
            self.stats['total_failed'] += 1
            return {
                **record,
                "tts_audio_filepath": "",
                "stt_text": "",
                "processing_time_tts": tts_time,
                "processing_time_stt": 0.0,
                "audio_duration": 0.0,
                "pipeline_status": "tts_failed"
            }
        
        # Save audio file
        try:
            with open(audio_filepath, 'wb') as f:
                f.write(audio_bytes)
            print(f"💾 Saved audio: {audio_filepath}")
        except Exception as e:
            print(f"❌ Failed to save audio: {e}")
            self.stats['total_failed'] += 1
            return {
                **record,
                "tts_audio_filepath": "",
                "stt_text": "",
                "processing_time_tts": tts_time,
                "processing_time_stt": 0.0,
                "audio_duration": 0.0,
                "pipeline_status": "save_failed"
            }
        
        # Step 2: STT - Transcribe audio
        stt_start = time.monotonic()
        transcribed_text = self.transcribe_audio(audio_bytes, stt_model, language)
        stt_time = time.monotonic() - stt_start
        
        if transcribed_text is None:
            transcribed_text = ""
            status = "stt_failed"
            self.stats['total_failed'] += 1
        else:
            status = "success"
            self.stats['total_processed'] += 1
        
        # Calculate audio duration (approximate)
        try:
            import librosa
            duration = librosa.get_duration(filename=audio_filepath)
        except:
            duration = len(audio_bytes) / 16000  # Rough estimate
        
        total_time = time.monotonic() - start_time
        
        # Track performance stats
        self.stats['tts_times'].append(tts_time)
        self.stats['stt_times'].append(stt_time)
        self.stats['total_times'].append(total_time)
        
        # Create enhanced record
        result = {
            **record,
            "tts_audio_filepath": str(audio_filepath),
            "stt_text": transcribed_text,
            "processing_time_tts": round(tts_time, 2),
            "processing_time_stt": round(stt_time, 2),
            "processing_time_total": round(total_time, 2),
            "audio_duration": round(duration, 2),
            "pipeline_status": status
        }
        
        print(f"✅ Completed {record_id}: {status} (TTS: {tts_time:.1f}s, STT: {stt_time:.1f}s)")
        return result
    
    def print_stats(self):
        """Print performance statistics."""
        if not self.stats['tts_times']:
            print("No statistics to display.")
            return
        
        tts_times = np.array(self.stats['tts_times'])
        stt_times = np.array(self.stats['stt_times'])
        total_times = np.array(self.stats['total_times'])
        
        print("\n" + "="*60)
        print("📊 PIPELINE PERFORMANCE STATISTICS")
        print("="*60)
        print(f"Records processed: {self.stats['total_processed']}")
        print(f"Records failed: {self.stats['total_failed']}")
        print(f"Records skipped: {self.stats['total_skipped']}")
        print()
        print("TTS Performance:")
        print(f"  Mean: {np.mean(tts_times):.2f}s, Std: {np.std(tts_times):.2f}s")
        print(f"  Min: {np.min(tts_times):.2f}s, Max: {np.max(tts_times):.2f}s")
        print()
        print("STT Performance:")
        print(f"  Mean: {np.mean(stt_times):.2f}s, Std: {np.std(stt_times):.2f}s")
        print(f"  Min: {np.min(stt_times):.2f}s, Max: {np.max(stt_times):.2f}s")
        print()
        print("Total Processing Time:")
        print(f"  Mean: {np.mean(total_times):.2f}s, Std: {np.std(total_times):.2f}s")
        print(f"  Min: {np.min(total_times):.2f}s, Max: {np.max(total_times):.2f}s")
        print("="*60)


def load_processed_ids(output_path: Path) -> set:
    """Load already processed record IDs for resume functionality."""
    processed_ids = set()
    if output_path.exists():
        print(f"📄 Loading existing results from {output_path}")
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    record_key = f"{record['id']}_{record['lang']}_{record['tts_model_name']}_{record['stt_model_name']}"
                    processed_ids.add(record_key)
        print(f"🔄 Found {len(processed_ids)} already processed records")
    return processed_ids


@app.local_entrypoint()
def main(
    input_file: str = "sample_processed.jsonl",
    output_file: str = "output.jsonl", 
    output_dir: str = "data/audio",
    resume: bool = True,
    limit: Optional[int] = None,
    reverse: bool = True
):
    
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    print("🚀 Initializing TTS-STT pipeline processor...")
    processor = PipelineProcessor(output_dir=output_dir)
    
    # Load existing results for resume functionality
    processed_ids = set()
    if resume:
        processed_ids = load_processed_ids(output_path)
    
    # Read input records
    print(f"📖 Reading input from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        input_records = [json.loads(line) for line in f if line.strip()]
    
    # Apply limit if specified
    if limit:
        input_records = input_records[:limit]
        print(f"🔍 Limited to first {limit} records")
    
    print(f"📊 Total records to process: {len(input_records)}")
    if reverse:
        input_records = input_records[::-1]  # Reverse for starting with the longest records first
    # Process records and append results immediately
    with open(output_path, 'a', encoding='utf-8') as output_f:
        for i, record in enumerate(input_records, 1):
            # Create unique record ID for resume check
            record_key = f"{record['id']}_{record['lang']}_{record['tts_model_name']}_{record['stt_model_name']}"
            
            if resume and record_key in processed_ids:
                processor.stats['total_skipped'] += 1
                print(f"⏭️  Skipping {i}/{len(input_records)}: {record_key} (already processed)")
                continue
            
            print(f"\n🚀 Processing {i}/{len(input_records)}")
            
            try:
                # Process the record
                result = processor.process_record(record)
                
                # Immediately append to output file
                output_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                output_f.flush()  # Ensure immediate write to disk
                
            except Exception as e:
                print(f"❌ Failed to process record {record.get('id', 'unknown')}: {e}")
                processor.stats['total_failed'] += 1
                # Write failed record with error info
                error_result = {
                    **record,
                    "tts_audio_filepath": "",
                    "stt_text": "",
                    "processing_time_tts": 0.0,
                    "processing_time_stt": 0.0,
                    "processing_time_total": 0.0,
                    "audio_duration": 0.0,
                    "pipeline_status": "error",
                    "error_message": str(e)
                }
                output_f.write(json.dumps(error_result, ensure_ascii=False) + '\n')
                output_f.flush()
    
    print(f"\n🎉 Pipeline completed!")
    print(f"   Output: {output_path}")
    
    # Print performance statistics
    processor.print_stats()


if __name__ == "__main__":
    main()
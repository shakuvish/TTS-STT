#!/usr/bin/env python3
"""
End-to-end TTS-STT pipeline orchestrator.
Processes sample_processed.jsonl through TTS → STT pipeline and appends results immediately.
"""

import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
import argparse

# Modal imports for cloud services
import modal

# Local service imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from deploy.local_tts import GeminiTTSService
from deploy.local_stt import GeminiSTTService


class PipelineOrchestrator:
    def __init__(self, output_dir: str = "data/audio"):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            output_dir: Directory to save audio files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize local services
        print("🔧 Initializing local services...")
        self.gemini_tts = GeminiTTSService()
        self.gemini_stt = GeminiSTTService()
        
        # Initialize Modal cloud services
        print("☁️ Connecting to Modal cloud services...")
        self._init_modal_services()
        
        print("✅ Pipeline orchestrator initialized")
    
    def _init_modal_services(self):
        """Initialize connections to Modal cloud services."""
        try:
            # Import Modal app and services
            from deploy.tts_service import IndicParlerModel, ParlerTTSModel
            from deploy.stt_service import IndicConformerModel, SpeechBrainConformerModel
            
            # Get Modal app
            app = modal.App.lookup("indic-speech-pipeline")
            
            # Initialize cloud TTS services
            self.indic_parler_tts = IndicParlerModel()
            self.parler_tts = ParlerTTSModel()
            
            # Initialize cloud STT services  
            self.indic_conformer_stt = IndicConformerModel()
            self.conformer_stt = SpeechBrainConformerModel()
            
        except Exception as e:
            print(f"Warning: Could not connect to Modal services: {e}")
            print("Cloud services will not be available")
            self.indic_parler_tts = None
            self.parler_tts = None
            self.indic_conformer_stt = None
            self.conformer_stt = None
    
    def generate_tts_audio(self, text: str, tts_model: str, language: str) -> Optional[bytes]:
        """
        Generate audio using specified TTS model.
        
        Returns:
            Audio bytes or None if failed
        """
        try:
            if tts_model == "gemini_flash":
                return self.gemini_tts.generate_speech(text, language=language)
            
            elif tts_model == "indic_parle":
                if self.indic_parler_tts is None:
                    raise Exception("IndicParler service not available")
                # Default description for IndicParler
                desc = "A clear, natural voice speaking in Hindi" if language == "hi" else "A clear, natural voice"
                return self.indic_parler_tts.generate.remote(text, desc)
            
            elif tts_model == "parler":
                if self.parler_tts is None:
                    raise Exception("Parler service not available")
                # Default description for Parler
                desc = "A clear, neutral voice speaks at a moderate pace."
                return self.parler_tts.generate.remote(text, desc)
            
            else:
                raise Exception(f"Unknown TTS model: {tts_model}")
                
        except Exception as e:
            print(f"❌ TTS Error ({tts_model}): {e}")
            return None
    
    def transcribe_audio(self, audio_bytes: bytes, stt_model: str, language: str) -> Optional[str]:
        """
        Transcribe audio using specified STT model.
        
        Returns:
            Transcribed text or None if failed
        """
        try:
            if stt_model == "gemini_flash":
                return self.gemini_stt.transcribe_audio(audio_bytes, language=language)
            
            elif stt_model == "indic_conformer":
                if self.indic_conformer_stt is None:
                    raise Exception("IndicConformer service not available")
                result = self.indic_conformer_stt.transcribe.remote(audio_bytes, language=language)
                # Return CTC transcription as default
                return result.get("ctc_transcription", "")
            
            elif stt_model == "conformer":
                if self.conformer_stt is None:
                    raise Exception("SpeechBrain Conformer service not available")
                result = self.conformer_stt.transcribe.remote(audio_bytes, language=language)
                return result.get("transcription", "")
            
            else:
                raise Exception(f"Unknown STT model: {stt_model}")
                
        except Exception as e:
            print(f"❌ STT Error ({stt_model}): {e}")
            return None
    
    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single record through the TTS-STT pipeline.
        
        Returns:
            Enhanced record with pipeline results
        """
        start_time = time.time()
        
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
        tts_start = time.time()
        audio_bytes = self.generate_tts_audio(text, tts_model, language)
        tts_time = time.time() - tts_start
        
        if audio_bytes is None:
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
        stt_start = time.time()
        transcribed_text = self.transcribe_audio(audio_bytes, stt_model, language)
        stt_time = time.time() - stt_start
        
        if transcribed_text is None:
            transcribed_text = ""
            status = "stt_failed"
        else:
            status = "success"
        
        # Calculate audio duration (approximate)
        try:
            import librosa
            duration = librosa.get_duration(filename=audio_filepath)
        except:
            duration = len(audio_bytes) / 16000  # Rough estimate
        
        total_time = time.time() - start_time
        
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
    
    def run_pipeline(self, input_file: str, output_file: str, resume: bool = False):
        """
        Run the complete pipeline on input JSONL file.
        
        Args:
            input_file: Path to input JSONL file
            output_file: Path to output JSONL file  
            resume: Skip records already in output file
        """
        input_path = Path(input_file)
        output_path = Path(output_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing results for resume functionality
        processed_ids = set()
        if resume and output_path.exists():
            print(f"📄 Loading existing results from {output_path}")
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        processed_ids.add(f"{record['id']}_{record['lang']}_{record['tts_model_name']}_{record['stt_model_name']}")
            print(f"🔄 Found {len(processed_ids)} already processed records")
        
        # Read input records
        print(f"📖 Reading input from {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            input_records = [json.loads(line) for line in f if line.strip()]
        
        print(f"📊 Total records to process: {len(input_records)}")
        
        # Process records and append results immediately
        processed_count = 0
        skipped_count = 0
        
        with open(output_path, 'a', encoding='utf-8') as output_f:
            for i, record in enumerate(input_records, 1):
                # Create unique record ID for resume check
                record_key = f"{record['id']}_{record['lang']}_{record['tts_model_name']}_{record['stt_model_name']}"
                
                if resume and record_key in processed_ids:
                    skipped_count += 1
                    print(f"⏭️  Skipping {i}/{len(input_records)}: {record_key} (already processed)")
                    continue
                
                print(f"\n🚀 Processing {i}/{len(input_records)}")
                
                try:
                    # Process the record
                    result = self.process_record(record)
                    
                    # Immediately append to output file
                    output_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    output_f.flush()  # Ensure immediate write to disk
                    
                    processed_count += 1
                    
                except Exception as e:
                    print(f"❌ Failed to process record {record.get('id', 'unknown')}: {e}")
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
        print(f"   Processed: {processed_count}")
        print(f"   Skipped: {skipped_count}")
        print(f"   Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run TTS-STT pipeline on processed dataset")
    parser.add_argument("input_file", help="Input JSONL file (sample_processed.jsonl)")
    parser.add_argument("output_file", help="Output JSONL file for results")
    parser.add_argument("--output-dir", default="data/audio", help="Directory to save audio files")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(output_dir=args.output_dir)
    
    # Run pipeline
    orchestrator.run_pipeline(
        input_file=args.input_file,
        output_file=args.output_file,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
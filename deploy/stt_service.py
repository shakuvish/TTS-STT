# deploy/stt_service.py
from deploy.common import app, image, volume, CACHE_PATH, _install_flash_attn
from modal import enter, method
import modal

# This helper function can live outside the class
def _preprocess_audio(audio_file_obj):
    import torch
    import torchaudio
    wav, original_sample_rate = torchaudio.load(audio_file_obj)
    if wav.shape[0] > 1: wav = torch.mean(wav, dim=0, keepdim=True)
    if original_sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=16000)
        wav = resampler(wav)
    max_val = torch.max(torch.abs(wav))
    if max_val > 0: wav = wav / max_val
    return wav

@app.cls(
    image=image,
    gpu="A10G",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_PATH: volume},
    timeout=7200,
)
class IndicConformerModel:
    @modal.enter()
    def setup(self):
        """
        This method runs once when the container starts.
        """
        import os
        from transformers import AutoModel
        
        # Attempt to install flash-attn for consistency
        # _install_flash_attn()

        print("🗣️ Loading Indic-Conformer ASR model...")
        model_name = "ai4bharat/indic-conformer-600m-multilingual"
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=CACHE_PATH,
            token=os.getenv("HF_TOKEN")
        )
        print("✅ ASR Model setup complete.")

    @modal.method()
    def transcribe(self, audio_bytes: bytes, language: str = "hi") -> dict:
        import io

        audio_file_obj = io.BytesIO(audio_bytes)
        wav = _preprocess_audio(audio_file_obj)

        transcription_ctc = self.model(wav, language, "ctc")
        transcription_rnnt = self.model(wav, language, "rnnt")

        return {
            "ctc_transcription": transcription_ctc,
            "rnnt_transcription": transcription_rnnt,
            "language": language,
            "audio_duration_seconds": wav.shape[1] / 16000
        }


@app.cls(
    image=image,
    gpu="A10G",
    volumes={CACHE_PATH: volume},
    timeout=7200,
)
class SpeechBrainConformerModel:
    @modal.enter()
    def setup(self):
        """
        This method runs once when the container starts.
        Sets up SpeechBrain Conformer ASR model for English.
        """
        import torch
        from speechbrain.inference.ASR import EncoderDecoderASR

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("🗣️ Loading SpeechBrain Conformer ASR model...")
        
        # Load the pre-trained ASR model
        self.asr_model = EncoderDecoderASR.from_hparams(
            source="speechbrain/asr-conformer-transformerlm-librispeech",
            savedir=f"{CACHE_PATH}/pretrained_models/asr-conformer-en-librispeech",
            run_opts={"device": self.device}
        )
        
        print("✅ SpeechBrain Conformer ASR Model setup complete.")

    @modal.method()
    def transcribe(self, audio_bytes: bytes, language: str = "en") -> dict:
        """
        Transcribe audio using SpeechBrain Conformer model.
        
        Args:
            audio_bytes: Audio data as bytes
            language: Language code (should be 'en' for this model)
            
        Returns:
            Dictionary with transcription results
        """
        import io
        import tempfile
        import os
        from pathlib import Path
        
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_audio_path = temp_file.name
        
        try:
            # Transcribe the audio file
            transcribed_text = self.asr_model.transcribe_file(temp_audio_path)
            
            # Calculate audio duration (approximate)
            import torchaudio
            wav, sample_rate = torchaudio.load(temp_audio_path)
            duration_seconds = wav.shape[1] / sample_rate
            
            return {
                "transcription": transcribed_text,
                "language": language,
                "audio_duration_seconds": float(duration_seconds),
                "model": "speechbrain-conformer"
            }
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            return {
                "transcription": "",
                "language": language,
                "audio_duration_seconds": 0.0,
                "model": "speechbrain-conformer",
                "error": str(e)
            }
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
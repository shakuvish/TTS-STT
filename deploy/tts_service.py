# deploy/tts_service.py
from deploy.common import app, image, volume, CACHE_PATH, _install_flash_attn
from modal import enter, method
import torch
import modal
@app.cls(
    image=image,
    gpu="A10G",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_PATH: volume},
    timeout=7200,
)
class IndicParlerModel:
    @modal.enter()
    def setup(self):
        """
        This method runs once when the container starts.
        It handles all one-time setup tasks.
        """
        import os
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer, AutoFeatureExtractor
        import nltk

        # 1. Attempt to install flash-attn and check if it succeeded.
        # flash_attn_available = _install_flash_attn()
        flash_attn_available = False
        # 2. Set up device and data type
        self.device = "cuda"
        self.torch_dtype = torch.bfloat16

        # 3. Conditionally set the attention implementation.
        # Fallback to 'eager' if flash_attn is not available.
        attn_impl = "flash_attention_2" if flash_attn_available else "eager"
        print(f"✅ Using attention implementation: {attn_impl}")

        # 4. Load model and tokenizers
        print("🎙️ Loading Indic-Parler TTS model...")
        model_id = "ai4bharat/indic-parler-tts"
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_id,
            cache_dir=CACHE_PATH,
            token=os.getenv("HF_TOKEN"),
            attn_implementation=attn_impl, # Use the determined implementation
            torch_dtype=self.torch_dtype
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_PATH)
        self.desc_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large", cache_dir=CACHE_PATH)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, cache_dir=CACHE_PATH)
        
        # 5. Download NLTK data needed for sentence tokenization
        nltk.download("punkt_tab", quiet=True)

        print("✅ TTS Model setup complete.")

    @modal.method()
    def generate(self, prompt: str, description: str) -> bytes:
        """
        This is the method that performs the TTS generation.
        It uses the pre-loaded models from the setup method.
        """
        import numpy as np
        import io
        from nltk.tokenize import sent_tokenize
        from pydub import AudioSegment

        sampling_rate = self.feature_extractor.sampling_rate
        desc_inputs = self.desc_tokenizer(description, return_tensors="pt").to(self.device)
        
        # Process text into chunks
        sentences = sent_tokenize(prompt)
        chunk_size = 25
        chunks = []
        current_sentence = ""
        for sentence in sentences:
            candidate = " ".join([current_sentence, sentence]).strip()
            if len(candidate.split()) >= chunk_size:
                if current_sentence:
                    chunks.append(current_sentence.strip())
                current_sentence = sentence
            else:
                current_sentence = candidate
        if current_sentence:
            chunks.append(current_sentence.strip())
        
        print(f"Processing {len(chunks)} chunks: {chunks}")
        
        # Generate audio for each chunk
        audio_list = []
        for chunk in chunks:
            prompt_inputs = self.tokenizer(chunk, return_tensors="pt").to(self.device)
            generation = self.model.generate(
                input_ids=desc_inputs.input_ids,
                attention_mask=desc_inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
                do_sample=True, return_dict_in_generate=True
            )
            
            if hasattr(generation, 'sequences') and hasattr(generation, 'audios_length'):
                audio = generation.sequences[0, :generation.audios_length[0]]
                audio_np = audio.to(torch.float32).cpu().numpy().squeeze()
                if audio_np.ndim > 1:
                    audio_np = audio_np.flatten()
                audio_list.append(audio_np)
        
        if not audio_list:
            return b''

        combined_audio = np.concatenate(audio_list)
        
        # Convert to MP3
        if np.issubdtype(combined_audio.dtype, np.floating):
            max_val = np.max(np.abs(combined_audio))
            if max_val > 0:
                combined_audio = (combined_audio / max_val) * 32767
            combined_audio = combined_audio.astype(np.int16)

        audio_segment = AudioSegment(
            combined_audio.tobytes(),
            frame_rate=sampling_rate,
            sample_width=combined_audio.dtype.itemsize,
            channels=1
        )
        mp3_io = io.BytesIO()
        audio_segment.export(mp3_io, format="mp3", bitrate="320k")
        return mp3_io.getvalue()


@app.cls(
    image=image,
    gpu="A10G",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_PATH: volume},
    timeout=7200,
)
class ParlerTTSModel:
    @modal.enter()
    def setup(self):
        """
        This method runs once when the container starts.
        It handles all one-time setup tasks for Parler-TTS.
        """
        import os
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer, AutoFeatureExtractor
        import nltk

        # 1. Attempt to install flash-attn and check if it succeeded.
        # flash_attn_available = _install_flash_attn()
        flash_attn_available = False
        # 2. Set up device and data type
        self.device = "cuda"
        self.torch_dtype = torch.bfloat16

        # 3. Conditionally set the attention implementation.
        # Fallback to 'eager' if flash_attn is not available.
        attn_impl = "flash_attention_2" if flash_attn_available else "eager"
        print(f"✅ Using attention implementation: {attn_impl}")

        # 4. Load Parler-TTS model and tokenizers
        print("🎙️ Loading Parler-TTS model...")
        model_id = "parler-tts/parler-tts-mini-v1.1"
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_id,
            cache_dir=CACHE_PATH,
            token=os.getenv("HF_TOKEN"),
            attn_implementation=attn_impl,
            torch_dtype=self.torch_dtype
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_PATH)
        self.desc_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large", cache_dir=CACHE_PATH)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, cache_dir=CACHE_PATH)
        
        # 5. Download NLTK data needed for sentence tokenization
        nltk.download("punkt_tab", quiet=True)

        print("✅ Parler-TTS Model setup complete.")

    @modal.method()
    def generate(self, prompt: str, description: str = "A clear, neutral voice speaks at a moderate pace.") -> bytes:
        """
        This method performs TTS generation using Parler-TTS.
        Uses the pre-loaded models from the setup method.
        """
        import numpy as np
        import io
        from nltk.tokenize import sent_tokenize
        from pydub import AudioSegment

        sampling_rate = self.feature_extractor.sampling_rate
        desc_inputs = self.desc_tokenizer(description, return_tensors="pt").to(self.device)
        
        # Process text into chunks
        sentences = sent_tokenize(prompt)
        chunk_size = 25
        chunks = []
        current_sentence = ""
        for sentence in sentences:
            candidate = " ".join([current_sentence, sentence]).strip()
            if len(candidate.split()) >= chunk_size:
                if current_sentence:
                    chunks.append(current_sentence.strip())
                current_sentence = sentence
            else:
                current_sentence = candidate
        if current_sentence:
            chunks.append(current_sentence.strip())
        
        print(f"Processing {len(chunks)} chunks: {chunks}")
        
        # Generate audio for each chunk
        audio_list = []
        for chunk in chunks:
            prompt_inputs = self.tokenizer(chunk, return_tensors="pt").to(self.device)
            generation = self.model.generate(
                input_ids=desc_inputs.input_ids,
                attention_mask=desc_inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
                do_sample=True, 
                return_dict_in_generate=True
            )
            
            if hasattr(generation, 'sequences') and hasattr(generation, 'audios_length'):
                audio = generation.sequences[0, :generation.audios_length[0]]
                audio_np = audio.to(torch.float32).cpu().numpy().squeeze()
                if audio_np.ndim > 1:
                    audio_np = audio_np.flatten()
                audio_list.append(audio_np)
        
        if not audio_list:
            return b''

        combined_audio = np.concatenate(audio_list)
        
        # Convert to MP3
        if np.issubdtype(combined_audio.dtype, np.floating):
            max_val = np.max(np.abs(combined_audio))
            if max_val > 0:
                combined_audio = (combined_audio / max_val) * 32767
            combined_audio = combined_audio.astype(np.int16)

        audio_segment = AudioSegment(
            combined_audio.tobytes(),
            frame_rate=sampling_rate,
            sample_width=combined_audio.dtype.itemsize,
            channels=1
        )
        mp3_io = io.BytesIO()
        audio_segment.export(mp3_io, format="mp3", bitrate="320k")
        return mp3_io.getvalue()
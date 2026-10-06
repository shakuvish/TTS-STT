# deploy/common.py
from modal import App, Image, Volume
import subprocess

# 1. A single, more general App for the entire pipeline.
app = App("indic-speech-pipeline")

def _install_flash_attn() -> bool:
    """
    Tries to install flash-attn at runtime. Returns True on success, False on failure.
    """
    try:
        print("⚡️ Attempting to install flash-attn...")
        # Running pip install and capturing output
        subprocess.run(
            ["pip", "install", "flash-attn==2.5.8", "--no-build-isolation"],
            check=True,
            capture_output=True, text=True,
        )
        print("✅ flash-attn installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print("⚠️ Could not install flash-attn. Will use default attention mechanism.")
        print("   This is expected on some environments and is not a fatal error.")
        # print("Stderr:", e.stderr) # Uncomment for deep debugging
        return False
# 2. A single Image combining all dependencies from both services.
image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", "ffmpeg", "libsndfile1", "sox", "libavcodec-extra",
        "libavformat-dev", "libavdevice-dev", "libavutil-dev", "build-essential",
    )
    .pip_install(
        "torch", "transformers", "soundfile", "torchaudio",
        "librosa", "onnxruntime-gpu", "pydub", "nltk",
        "git+https://github.com/huggingface/parler-tts.git",
        "speechbrain",
        # flash-attn is now removed from the build
    )
)

# 3. A single, shared Volume for caching all models.
volume = Volume.from_name("indic-speech-cache", create_if_missing=True)
CACHE_PATH = "/cache"
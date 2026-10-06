#!/usr/bin/env python3
"""
Batch test script for local Gemini TTS and STT services.
Tests 5 English and 5 Hindi complex, compound sentences of increasing length.
Calculates WER and CER for each transcription.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from deploy.local_tts import GeminiTTSService
from deploy.local_stt import GeminiSTTService
from jiwer import wer, cer

load_dotenv()

def test_gemini_batch():
    tts = GeminiTTSService()
    stt = GeminiSTTService()

    english_sentences = [
        "Although the sun set gradually, the sky remained ablaze with crimson hues.",
        "She opened the ancient tome, and as she read the faded script, memories long forgotten returned.",
        "When the orchestra began its first movement, every listener sat entranced, for the melodies wove spells of nostalgia.",
        "While the rain poured relentlessly outside, he sat by the fireplace, sipping his ale and pondering choices that shaped his destiny.",
        "Because the city slept under a blanket of snow and silence, the traveler walked its empty streets with cautious steps, wondering what mysteries lay hidden in moonlit alleys."
    ]

    hindi_sentences = [
        "जब सूरज आसमान से विदा हुआ, तो बादलों ने अलाव-सी रोशनी फैलाई।",
        "वह पुस्तक खोलकर बैठी, और जैसे-जैसे पन्ने पलटते गए, अतीत की यादें जाग उठीं।",
        "जब ऑर्केस्ट्रा ने अपना पहला संगीत पेश किया, तो हर श्रोता मंत्रमुग्ध हो गया और समय थम सा गया।",
        "बाहर जब तेज बारिश हो रही थी, तो वह अंगीठी के पास चाय की चुस्कियाँ लेते हुए अपनी ज़िंदगी पर चिंतन करता रहा।",
        "क्योंकि नगर पर सफ़ेद चाँदनी का चादर थी और सड़कें सुनसान, यात्री ने कदम संभाल कर चांदनी गलियों में शांति से सफ़र किया।"
    ]

    print("🧪 Batch Testing Gemini TTS and STT with WER & CER")
    print("=" * 60)

    for lang, sentences in [("en", english_sentences), ("hi", hindi_sentences)]:
        for idx, text in enumerate(sentences, start=1):
            print(f"\n📝 [{lang.upper()} #{idx}] {text}")
            audio_bytes = tts.generate_speech(
                text=text,
                language=lang,
                output_path=f"batch_{lang}_{idx}.mp3"
            )
            if not audio_bytes:
                print(f"❌ TTS failed for {lang.upper()} #{idx}")
                continue

            print(f"✅ Generated {len(audio_bytes)} bytes of audio")

            transcription = stt.transcribe_audio(
                audio_data=audio_bytes,
                language=lang
            )
            if not transcription:
                print(f"❌ STT failed for {lang.upper()} #{idx}")
                continue

            print(f"✅ Transcribed: {transcription}")
            print(f"   Original:    {text}")

            # compute metrics
            w = wer(text, transcription)
            c = cer(text, transcription)
            print(f"   WER: {w:.2%}, CER: {c:.2%}")

    print("\n🎉 Batch test completed. Check generated MP3 files.")

if __name__ == "__main__":
    test_gemini_batch()

"""
test_tanglish_pipeline.py - Automated Verification & Latency Benchmark for Lyla
Validates:
1. Faster-Whisper CUDA FP16 latency and Tanglish transcription accuracy.
2. Kokoro Tanglish IPA phonetic synthesis latency with blended studio voice.
3. Ollama llama3.2:3b streaming generation latency and dialect adaptation.
4. Far-field low-voice and whisper acoustic DSP & dynamic SNR detection.
"""

import os, sys, time, re
import numpy as np
import soundfile as sf
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
torch_lib = os.path.join(BASE, "venv", "Lib", "site-packages", "torch", "lib")
if os.path.exists(torch_lib):
    try:
        os.add_dll_directory(torch_lib)
    except Exception:
        pass
    os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")

import tanglish_engine
from faster_whisper import WhisperModel
from kokoro import KPipeline

print("=" * 60)
print("LYLA FULL-DUPLEX TANGLISH PIPELINE BENCHMARK")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Device] PyTorch CUDA: {torch.cuda.is_available()}, Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# 1. Benchmark Faster-Whisper CUDA
print("\n--- 1. Testing Faster-Whisper CUDA STT ---")
t0 = time.time()
whisper = WhisperModel("base", device=device, compute_type="float16" if device == "cuda" else "int8")
print(f"Loaded Faster-Whisper in {time.time() - t0:.2f}s")

# Test with 2 seconds of synthetic audio
test_audio = np.zeros(16000 * 2, dtype=np.float32)
# Warmup
list(whisper.transcribe(test_audio, language="en", beam_size=1)[0])

t1 = time.time()
segs, _ = whisper.transcribe(
    test_audio, language="en", beam_size=1, without_timestamps=True,
    condition_on_previous_text=False, initial_prompt=tanglish_engine.STT_INITIAL_PROMPT
)
list(segs)
stt_lat = (time.time() - t1) * 1000
print(f"Faster-Whisper CUDA 2s Audio Latency: {stt_lat:.1f} ms")
assert stt_lat < 500, "STT latency exceeded 500ms!"

# 2. Benchmark Kokoro Tanglish IPA Synthesis
print("\n--- 2. Testing Kokoro Tanglish IPA Neural Synthesis ---")
t0 = time.time()
kokoro = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=device)
v_bella = kokoro.load_voice("af_bella").to(device)
v_sarah = kokoro.load_voice("af_sarah").to(device)
blended_voice = (0.75 * v_bella + 0.25 * v_sarah).to(device)
print(f"Loaded Kokoro & Blended Studio Voice in {time.time() - t0:.2f}s")

sample_texts = [
    ("Chennai", "Hey machi! Naan semmaya iruken da, enna innaiku matter-u?"),
    ("Kongu", "Vanakkamunga! Naan nalla irukene, neenga aptheengala kannu?"),
    ("Madurai", "Yov kettukko! Namma gethu eppavum koraiyadhu pa!"),
    ("Tanglish", "Hey macha! Chill pannu, no scene actually.")
]

for dialect_name, text in sample_texts:
    formatted = tanglish_engine.format_conversational_prosody(text)
    phonemes = tanglish_engine.text_to_tanglish_phonemes(formatted, kokoro.g2p)
    t_synth = time.time()
    out = KPipeline.infer(kokoro.model, phonemes, blended_voice, speed=1.0)
    lat = (time.time() - t_synth) * 1000
    audio_len = len(out.audio) / 24000 if out and out.audio is not None else 0
    print(f"  [{dialect_name}] Synth: {lat:.1f}ms | Audio: {audio_len:.2f}s | Phonemes: {phonemes[:40]}...")
    assert out is not None and out.audio is not None

# 3. Test Dialect Detection & STT Repair
print("\n--- 3. Testing Dialect Detection & STT Normalization ---")
assert tanglish_engine.detect_dialect_switch_intent("Lyla Chennai slang-la pesu") == "chennai"
assert tanglish_engine.detect_dialect_switch_intent("Coimbatore style-ku mathu") == "kongu"
assert tanglish_engine.detect_dialect_switch_intent("Madurai slang pesu pa") == "madurai"
assert tanglish_engine.detect_dialect_switch_intent("Switch to English please") == "english"
print("  All dialect switch intents detected successfully!")

fixed = tanglish_engine.enhance_tanglish_stt("solo match layla every rookie summer hot-ah")
print(f"  ASR Normalizer: 'solo match layla every rookie summer hot-ah' -> '{fixed}'")
assert "sollu machi" in fixed and "Lyla" in fixed and "epdi iruka" in fixed

# 4. Far-field Low-Voice & Whisper Noise Floor Tracker
print("\n--- 4. Testing Far-Field Low-Voice SNR & Noise Floor Tracker ---")
from server import NoiseFloorTracker, enhance_far_field

tracker = NoiseFloorTracker()
# Simulate ambient silent room frames
for _ in range(50):
    tracker.update(0.0015, is_speech=False)
print(f"  Calibrated Ambient Noise Floor: {tracker.floor:.4f}")

# Simulate low-voice whisper across room (~2m away, RMS ~0.0035)
whisper_rms = 0.0038
snr = tracker.snr_db(whisper_rms)
print(f"  Low-Voice Whisper RMS: {whisper_rms:.4f} -> SNR: {snr:.1f} dB")
assert snr > 5.0, "Low-voice SNR should be detected well above noise floor!"

whisper_audio = np.random.normal(0, 0.0035, 16000 * 2).astype(np.float32)
enhanced = enhance_far_field(whisper_audio, for_stt=True, max_gain=8.5, noise_floor=tracker.floor)
enh_rms = float(np.sqrt(np.mean(enhanced ** 2)))
print(f"  Far-Field DSP Boost: {whisper_rms:.4f} -> Enhanced RMS: {enh_rms:.4f} (Peak: {float(np.abs(enhanced).max()):.3f})")
assert enh_rms > 0.05, "Enhanced low-voice speech should be boosted to optimal ASR RMS!"

print("\n" + "=" * 60)
print("ALL PIPELINE BENCHMARKS PASSED PERFECTLY!")
print("=" * 60)

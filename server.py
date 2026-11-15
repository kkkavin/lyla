import os, sys, re, json, time, queue, struct, asyncio, threading
from collections import deque
from typing import Tuple, Dict, List, Optional
import numpy as np
import soundfile as sf
import torch
import ollama
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

# Ensure torch/lib DLLs are discoverable for CTranslate2 / Faster-Whisper CUDA acceleration on Windows
BASE = os.path.dirname(os.path.abspath(__file__))
torch_lib = os.path.join(BASE, "venv", "Lib", "site-packages", "torch", "lib")
if os.path.exists(torch_lib):
    try:
        os.add_dll_directory(torch_lib)
    except Exception:
        pass
    os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")

from kokoro import KPipeline
from resemblyzer import VoiceEncoder
from faster_whisper import WhisperModel
import tanglish_engine

# ================= CONFIG =================
LLM = "llama3.2:3b"
SR_IN = 16000
FRAME_SAMPLES = 512                            # Exact 32ms frame at 16kHz
FRAME_BYTES = FRAME_SAMPLES * 2
MAX_HISTORY = 12
SLEEP_TIMEOUT = 30.0
WHISPER_MODEL = "base"                         # Multilingual base model cached on CUDA float16
HOST, PORT = "127.0.0.1", 8000

VOICEPRINT_PATH = os.path.join(BASE, "owner_voice.npy")

# Timing and Frame Configuration (1 frame = 32ms)
PRE_ROLL_FRAMES       = 16                     # 512ms pre-roll captures unvoiced whisper onset
IDLE_START_FRAMES     = 3                      # 96ms of speech to start listening (rapid response)
END_SILENCE_FRAMES    = 20                     # 640ms pause after speech to trigger turn
MIN_UTTERANCE_FRAMES  = 12                     # 384ms minimum valid utterance
BARGE_SPEECH_FRAMES   = 5                      # 160ms of post-AEC3 owner speech to interrupt

# Neural VAD & Far-Field Sensitivity
VAD_PROB_IDLE         = 0.38                   # Calibrated for distant/low-voice speech
VAD_PROB_BARGE        = 0.62                   # High confidence gate for barge-in
MIN_SPEECH_RMS_FLOOR  = 0.0025                 # Allows faint whispers across room (~2m)
MIN_SNR_DB_SPEECH     = 2.2                    # Minimum dB over ambient noise floor

# Speaker Biometric Thresholds
SIMILARITY_NORMAL     = 0.52
OWNER_MARGIN_NORMAL   = 0.09
SIMILARITY_LOW_VOICE  = 0.44                   # Whisper harmonic drop compensation
OWNER_MARGIN_LOW_VOICE= 0.04
BARGE_OWNER_MIN       = 0.58
BARGE_MARGIN_MIN      = 0.04

torch.set_num_threads(max(2, (os.cpu_count() or 4) // 2))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Lyla Neural Engine Device:", DEVICE)

SYSTEM_RULES = (
    "\n\nCRITICAL CONVERSATIONAL RULES:\n"
    "1. The user (Kavin) is speaking across the room. If speech recognition slightly drops or mishears a word, "
    "infer what they meant naturally from context without complaining.\n"
    "2. SLEEP INTENT: If the user says go to sleep, take a rest, stop listening, stand by, or says goodbye, "
    "you MUST begin your reply with the exact tag [SLEEP] followed by a brief, natural goodbye.\n"
    "3. Keep every response to 1-2 punchy spoken lines. Sound warm, spontaneous, and conversational."
)
PERSONA = open(os.path.join(BASE, "persona.txt"), encoding="utf-8").read()

# ================= DYNAMIC NOISE FLOOR & SNR TRACKER =================
class NoiseFloorTracker:
    def __init__(self, alpha: float = 0.96, initial_floor: float = 0.002):
        self.floor = initial_floor
        self.alpha = alpha

    def update(self, frame_rms: float, is_speech: bool) -> float:
        if not is_speech and frame_rms > 0.0001:
            self.floor = self.alpha * self.floor + (1.0 - self.alpha) * frame_rms
            self.floor = max(0.0006, min(0.025, self.floor))
        return self.floor

    def snr_db(self, frame_rms: float) -> float:
        return float(20.0 * np.log10(max(1e-6, frame_rms) / max(1e-6, self.floor)))

# ================= NEURAL VAD (SILERO) =================
silero_model = None
try:
    from silero_vad import load_silero_vad
    silero_model = load_silero_vad(onnx=True)
    print("Loaded Silero Neural VAD (ONNX).")
except Exception:
    try:
        silero_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=True,
            trust_repo=True
        )
        print("Loaded Silero Neural VAD (TorchHub ONNX).")
    except Exception as e:
        print("Silero VAD fallback to spectral voice detector:", e)

def frame_speech_probability(frame_f32: np.ndarray, tracker: NoiseFloorTracker) -> Tuple[float, float, float]:
    """Returns (prob, rms, snr_db) for a 32ms frame with dynamic low-voice SNR detection."""
    rms = float(np.sqrt(np.mean(frame_f32 ** 2)))
    snr = tracker.snr_db(rms)

    # Dynamic low-voice energy floor gate
    if rms < MIN_SPEECH_RMS_FLOOR or snr < MIN_SNR_DB_SPEECH:
        tracker.update(rms, is_speech=False)
        return 0.0, rms, snr

    if silero_model is not None:
        with torch.no_grad():
            prob = float(silero_model(torch.from_numpy(frame_f32), SR_IN).item())
        # Formant-aware sensitivity boost for low-voice whispers
        if prob < VAD_PROB_IDLE and snr >= 5.0 and rms >= 0.0035:
            # Check spectral concentration in human vocal band (200Hz - 3600Hz)
            spec = np.abs(np.fft.rfft(frame_f32)) ** 2
            freqs = np.fft.rfftfreq(len(frame_f32), d=1.0 / SR_IN)
            voice_ratio = float(np.sum(spec[(freqs >= 200.0) & (freqs <= 3600.0)])) / (np.sum(spec) + 1e-12)
            if voice_ratio > 0.60:
                prob = max(prob, 0.45)
        tracker.update(rms, is_speech=(prob >= VAD_PROB_IDLE))
        return prob, rms, snr

    # Fallback spectral-harmonic detector
    spec = np.abs(np.fft.rfft(frame_f32 * np.hanning(len(frame_f32)))) ** 2
    freqs = np.fft.rfftfreq(len(frame_f32), d=1.0 / SR_IN)
    voice_band = spec[(freqs >= 180.0) & (freqs <= 3800.0)]
    total_e = float(np.sum(spec)) + 1e-12
    voice_ratio = float(np.sum(voice_band)) / total_e
    peak_ratio = float(np.max(voice_band)) / (float(np.sum(voice_band)) + 1e-12) if len(voice_band) else 1.0
    if voice_ratio > 0.58 and peak_ratio < 0.48:
        tracker.update(rms, is_speech=True)
        return min(1.0, voice_ratio), rms, snr
    tracker.update(rms, is_speech=False)
    return 0.0, rms, snr

# ================= SPEAKER & ASR MODELS =================
print("Loading voice encoder on CPU...")
voice_encoder = VoiceEncoder(device="cpu")

OWNER_EMBED = np.load(VOICEPRINT_PATH) if os.path.exists(VOICEPRINT_PATH) else None
LYLA_EMBED  = None

def resample_24k_to_16k(audio_24k: np.ndarray) -> np.ndarray:
    if len(audio_24k) == 0:
        return np.zeros(0, dtype=np.float32)
    idx = np.linspace(0, len(audio_24k) - 1, int(len(audio_24k) * SR_IN / 24000))
    return np.interp(idx, np.arange(len(audio_24k)), audio_24k).astype(np.float32)

def speech_bandpass(audio: np.ndarray) -> np.ndarray:
    n = len(audio)
    if n < 64:
        return audio
    freqs = np.fft.rfftfreq(n, d=1.0 / SR_IN)
    spec = np.fft.rfft(audio)
    mask = (freqs >= 150.0) & (freqs <= 4200.0)
    return np.fft.irfft(spec * mask, n=n).astype(np.float32)

def enhance_far_field(audio: np.ndarray, for_stt: bool = False, max_gain: float = 8.5, noise_floor: float = 0.002) -> np.ndarray:
    """Multi-stage acoustic DSP: Wiener spectral subtraction, tanh dynamic compression, and formant pre-emphasis."""
    if len(audio) == 0:
        return audio
    centered = audio - np.mean(audio)
    n = len(centered)
    if n < 64:
        return centered.astype(np.float32)

    # 1. FFT Bandpass filter (150Hz - 4200Hz)
    freqs = np.fft.rfftfreq(n, d=1.0 / SR_IN)
    spec = np.fft.rfft(centered)
    band_mask = (freqs >= 150.0) & (freqs <= 4200.0)
    filtered_spec = spec * band_mask

    # 2. Spectral noise suppression (subtract ambient room noise estimate)
    mag = np.abs(filtered_spec)
    phase = np.angle(filtered_spec)
    noise_est = noise_floor * np.sqrt(n) * 0.70
    cleaned_mag = np.maximum(mag - 1.20 * noise_est, 0.06 * mag)
    filtered = np.fft.irfft(cleaned_mag * np.exp(1j * phase), n=n)

    # 3. Dynamic Range Compression with soft-knee tanh
    rms = float(np.sqrt(np.mean(filtered ** 2)))
    if rms < 0.0008:
        return filtered.astype(np.float32)

    gain = min(max_gain, 0.095 / max(1e-5, rms))
    boosted = np.tanh((filtered * gain) * 1.15) / np.tanh(1.15)

    # 4. Formant consonant sharpening for STT
    if for_stt and len(boosted) > 2:
        boosted = np.append(boosted[0], boosted[1:] - 0.58 * boosted[:-1])

    peak = float(np.abs(boosted).max())
    if peak > 1e-3:
        boosted = boosted * (0.94 / peak)
    return boosted.astype(np.float32)

def verify_owner_voice(audio_16k: np.ndarray, max_gain: float = 7.5, noise_floor: float = 0.002):
    """Returns (owner_sim, lyla_sim, rms) with low-voice whisper compensation."""
    if len(audio_16k) < int(0.18 * SR_IN):
        return 0.0, 0.0, 0.0

    filtered = speech_bandpass(audio_16k - np.mean(audio_16k))
    rms = float(np.sqrt(np.mean(filtered ** 2)))
    if rms < MIN_SPEECH_RMS_FLOOR:
        return 0.0, 0.0, rms

    if OWNER_EMBED is None:
        return 1.0, 0.0, rms

    try:
        enhanced = enhance_far_field(filtered, for_stt=False, max_gain=max_gain, noise_floor=noise_floor)
        emb = voice_encoder.embed_utterance(enhanced)
        owner_sim = float(np.dot(OWNER_EMBED, emb))
        lyla_sim  = float(np.dot(LYLA_EMBED, emb)) if LYLA_EMBED is not None else 0.0
        return owner_sim, lyla_sim, rms
    except Exception:
        return 0.0, 0.0, rms

# Initialize CUDA Faster-Whisper with sub-200ms latency
print("Loading Faster-Whisper on CUDA (FP16)...")
whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_compute = "float16" if whisper_device == "cuda" else "int8"
_whisper = WhisperModel(WHISPER_MODEL, device=whisper_device, compute_type=whisper_compute)

def transcribe(audio_f32: np.ndarray) -> str:
    """Fast CUDA transcription with Tanglish acoustic bias and domain normalization."""
    segs, _ = _whisper.transcribe(
        audio_f32,
        language="en",
        beam_size=1,
        without_timestamps=True,
        vad_filter=False,
        condition_on_previous_text=False,
        initial_prompt=tanglish_engine.STT_INITIAL_PROMPT
    )
    raw_text = " ".join(s.text for s in segs).strip()
    return tanglish_engine.enhance_tanglish_stt(raw_text)

# Initialize Kokoro Neural TTS and Studio Voice Blending
print("Loading Kokoro-82M on", DEVICE, "...")
kokoro = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=DEVICE)

# Blend voices for studio-quality broadcast warmth
try:
    v_bella = kokoro.load_voice("af_bella").to(DEVICE)
    v_sarah = kokoro.load_voice("af_sarah").to(DEVICE)
    BLENDED_VOICE = (0.75 * v_bella + 0.25 * v_sarah).to(DEVICE)
    print("Created custom blended studio voice pack (af_bella + af_sarah).")
except Exception as e:
    print("Voice blend fallback to af_heart:", e)
    BLENDED_VOICE = kokoro.load_voice("af_heart").to(DEVICE)

def has_wake_word(text: str) -> bool:
    return bool(re.search(r"\blyla\b", text, re.I))

# Sub-clause regex for ultra-low TTFA (<450ms)
FIRST = re.compile(r"^(.{8,}?[,.!?…;:\-—])\s")
REST  = re.compile(r"^(.+?[.!?…—])\s")

history = [{"role": "system", "content": PERSONA}]
stt_lock, tts_lock, spk_lock = threading.Lock(), threading.Lock(), threading.Lock()

# ================= UNIFIED FULL-DUPLEX SESSION =================
class Session:
    def __init__(self, ws, loop):
        self.ws, self.loop = ws, loop
        self.turn = 0
        self.awake = False
        self.dialect = "tanglish"                      # Active dialect: tanglish, chennai, kongu, madurai, english
        self.sleep_timer = None
        self.sleep_after_playback = False

        # Pipeline activity flags
        self.processing_turn = False
        self.lyla_speaking = False
        self.enroll_mode = False
        self.enroll_frames = []

        # Acoustic noise floor & dynamic SNR
        self.tracker = NoiseFloorTracker()

        # VAD State Machine buffers
        self.pcm_remainder = bytearray()
        self.pre_roll = deque(maxlen=PRE_ROLL_FRAMES)
        self.speech_frames = []
        self.user_speaking = False
        self.speech_streak = 0
        self.silence_streak = 0

        # Barge-in verification state
        self.barge_frames = []
        self.barge_streak = 0
        self.checking_barge = False

        self.out = asyncio.Queue()

    @property
    def is_busy(self):
        return self.processing_turn or self.lyla_speaking

    def emit(self, kind, payload):
        self.loop.call_soon_threadsafe(self.out.put_nowait, (kind, payload))

    def drain(self):
        while not self.out.empty():
            try: self.out.get_nowait()
            except asyncio.QueueEmpty: break

    def cancel_sleep_timer(self):
        if self.sleep_timer:
            self.sleep_timer.cancel()
            self.sleep_timer = None

    def reset_sleep_timer(self):
        self.cancel_sleep_timer()
        if self.awake and not self.sleep_after_playback and not self.is_busy and not self.user_speaking:
            self.sleep_timer = self.loop.call_later(SLEEP_TIMEOUT, self.go_to_sleep)

    def go_to_sleep(self):
        if self.awake:
            self.awake = False
            self.sleep_after_playback = False
            self.sleep_timer = None
            print("[State] Lyla went to sleep.")
            self.emit("json", {"type": "state", "awake": False})

    def wake_up(self):
        self.sleep_after_playback = False
        if not self.awake:
            self.awake = True
            print("[State] Lyla woke up!")
            self.emit("json", {"type": "state", "awake": True})
        self.reset_sleep_timer()

ACTIVE = {"sess": None}

def verify_and_trigger_barge_in(sess: Session, candidate_frames: list[np.ndarray]):
    try:
        audio = np.concatenate(candidate_frames)
        with spk_lock:
            owner_sim, lyla_sim, rms = verify_owner_voice(audio, max_gain=6.0, noise_floor=sess.tracker.floor)

        margin = owner_sim - lyla_sim
        print(f"[Barge-in VAD] rms={rms:.4f} owner={owner_sim:.3f} lyla={lyla_sim:.3f} margin={margin:+.3f}")

        if owner_sim >= BARGE_OWNER_MIN and margin >= BARGE_MARGIN_MIN and sess.is_busy:
            print(">>> OWNER BARGE-IN! Stopping Lyla immediately! <<<")
            sess.processing_turn = False
            sess.lyla_speaking = False
            sess.sleep_after_playback = False
            sess.turn += 1
            sess.drain()

            # Seamlessly transition directly into USER_SPEAKING keeping pre-roll + candidate speech!
            sess.speech_frames = list(sess.pre_roll) + candidate_frames
            sess.user_speaking = True
            sess.silence_streak = 0
            sess.barge_frames = []
            sess.barge_streak = 0

            sess.emit("json", {"type": "interrupt"})
            sess.emit("json", {"type": "vad", "speaking": True})
            sess.emit("json", {"type": "turn", "id": sess.turn})
    finally:
        sess.checking_barge = False

def process_audio_bytes(sess: Session, chunk: bytes):
    """Processes incoming 16kHz PCM stream in exact 32ms frames with adaptive low-voice VAD."""
    if sess.enroll_mode:
        sess.enroll_frames.append(chunk)
        return

    sess.pcm_remainder.extend(chunk)
    while len(sess.pcm_remainder) >= FRAME_BYTES:
        raw_frame = bytes(sess.pcm_remainder[:FRAME_BYTES])
        del sess.pcm_remainder[:FRAME_BYTES]

        frame_f32 = np.frombuffer(raw_frame, dtype=np.int16).astype(np.float32) / 32768.0
        prob, rms, snr = frame_speech_probability(frame_f32, sess.tracker)

        # ---------------- MODE A: Lyla is speaking/generating (Full-Duplex Barge-In) ----------------
        if sess.is_busy:
            if prob >= VAD_PROB_BARGE:
                sess.barge_streak += 1
                sess.barge_frames.append(frame_f32)
                if len(sess.barge_frames) > 20:
                    sess.barge_frames = sess.barge_frames[-20:]

                if sess.barge_streak >= BARGE_SPEECH_FRAMES and not sess.checking_barge:
                    sess.checking_barge = True
                    sess.barge_streak = 0
                    candidate = list(sess.barge_frames)
                    threading.Thread(
                        target=verify_and_trigger_barge_in,
                        args=(sess, candidate),
                        daemon=True
                    ).start()
            else:
                if sess.barge_streak > 0:
                    sess.barge_streak = max(0, sess.barge_streak - 1)
                if sess.barge_streak == 0:
                    sess.barge_frames = []
                    sess.pre_roll.append(frame_f32)
            continue

        # ---------------- MODE B: Lyla is idle / listening ----------------
        is_speech = prob >= VAD_PROB_IDLE

        if not sess.user_speaking:
            sess.pre_roll.append(frame_f32)
            if is_speech:
                sess.speech_streak += 1
                if sess.speech_streak >= IDLE_START_FRAMES:
                    sess.user_speaking = True
                    sess.silence_streak = 0
                    sess.speech_frames = list(sess.pre_roll)
                    sess.pre_roll.clear()
                    sess.cancel_sleep_timer()
                    sess.turn += 1
                    if sess.awake:
                        sess.drain()
                    sess.emit("json", {"type": "vad", "speaking": True, "rms": round(rms, 4), "snr": round(snr, 1)})
                    sess.emit("json", {"type": "turn", "id": sess.turn})
            else:
                sess.speech_streak = 0
        else:
            sess.speech_frames.append(frame_f32)
            if is_speech:
                sess.silence_streak = 0
            else:
                sess.silence_streak += 1
                if sess.silence_streak >= END_SILENCE_FRAMES:
                    # Utterance complete
                    sess.user_speaking = False
                    sess.speech_streak = 0
                    sess.silence_streak = 0
                    frames_copy = sess.speech_frames
                    sess.speech_frames = []
                    sess.emit("json", {"type": "vad", "speaking": False})

                    if len(frames_copy) < MIN_UTTERANCE_FRAMES:
                        if sess.awake:
                            sess.reset_sleep_timer()
                        sess.emit("json", {"type": "ignored"})
                    else:
                        audio = np.concatenate(frames_copy)
                        sess.processing_turn = True
                        threading.Thread(
                            target=run_turn,
                            args=(sess, audio, sess.turn),
                            daemon=True
                        ).start()

def run_turn(sess: Session, audio: np.ndarray, turn: int):
    try:
        with spk_lock:
            owner_sim, lyla_sim, rms = verify_owner_voice(audio, max_gain=7.5, noise_floor=sess.tracker.floor)

        if owner_sim == 0.0:
            if turn == sess.turn: sess.processing_turn = False
            if sess.awake and not sess.is_busy:
                sess.loop.call_soon_threadsafe(sess.reset_sleep_timer)
            sess.emit("json", {"type": "ignored"})
            return

        margin = owner_sim - lyla_sim
        is_low_voice = (rms < 0.010)
        sim_threshold = SIMILARITY_LOW_VOICE if is_low_voice else SIMILARITY_NORMAL
        margin_threshold = OWNER_MARGIN_LOW_VOICE if is_low_voice else OWNER_MARGIN_NORMAL

        is_owner = (
            OWNER_EMBED is None
            or (owner_sim >= sim_threshold and margin >= margin_threshold)
        )
        print(f"[Voice verify] rms={rms:.4f} ({'LOW-VOICE' if is_low_voice else 'NORMAL'}) owner={owner_sim:.3f}, lyla={lyla_sim:.3f}, margin={margin:+.3f} -> {'OWNER' if is_owner else 'IGNORED'}")

        if not is_owner:
            if turn == sess.turn: sess.processing_turn = False
            if sess.awake and not sess.is_busy:
                sess.loop.call_soon_threadsafe(sess.reset_sleep_timer)
            sess.emit("json", {"type": "ignored"})
            return

        # Far-field acoustic enhancement for STT
        stt_audio = enhance_far_field(audio, for_stt=True, max_gain=8.5, noise_floor=sess.tracker.floor)

        t0 = time.time()
        with stt_lock:
            text = transcribe(stt_audio)
        t1 = time.time()
        print(f"[STT {round((t1-t0)*1000)}ms]: '{text}'")

        if len(text) < 2 or turn != sess.turn:
            if turn == sess.turn: sess.processing_turn = False
            if sess.awake and not sess.is_busy:
                sess.loop.call_soon_threadsafe(sess.reset_sleep_timer)
            sess.emit("json", {"type": "ignored"})
            return

        if not sess.awake:
            if has_wake_word(text):
                sess.loop.call_soon_threadsafe(sess.wake_up)
            else:
                print(f"[Ignored while asleep]: {text}")
                if turn == sess.turn: sess.processing_turn = False
                sess.emit("json", {"type": "ignored"})
                return

        if turn != sess.turn: return

        # Check for in-conversation dialect switch request
        switch_intent = tanglish_engine.detect_dialect_switch_intent(text)
        if switch_intent and switch_intent != sess.dialect:
            sess.dialect = switch_intent
            dialect_data = tanglish_engine.DIALECTS[sess.dialect]
            print(f"[Dialect Switch] Switched to {dialect_data['name']}!")
            sess.emit("json", {
                "type": "dialect",
                "dialect": sess.dialect,
                "name": dialect_data["name"],
                "tag": dialect_data["tag"]
            })

        sess.loop.call_soon_threadsafe(sess.cancel_sleep_timer)
        sess.lyla_speaking = True
        sess.sleep_after_playback = False
        sess.barge_streak = 0
        sess.barge_frames = []

        dialect_cfg = tanglish_engine.DIALECTS.get(sess.dialect, tanglish_engine.DIALECTS["tanglish"])
        sess.emit("json", {
            "type": "transcript",
            "turn": turn,
            "text": text,
            "dialect": sess.dialect,
            "dialect_tag": dialect_cfg["tag"]
        })

        # Dynamic System Prompt injecting current regional dialect rules
        turn_system_prompt = (
            f"{PERSONA}\n\n"
            f"ACTIVE REGIONAL DIALECT: {dialect_cfg['name']}\n"
            f"{dialect_cfg['prompt']}\n"
            f"{SYSTEM_RULES}"
        )

        history.append({"role": "user", "content": text})
        extra = len(history) - 1 - MAX_HISTORY
        if extra > 0: del history[1:1 + extra]

        # Use current dialect prompt for this turn
        turn_history = [{"role": "system", "content": turn_system_prompt}] + history[1:]

        sq = queue.Queue()
        tm = {"llm": None, "sent": False}
        current_speed = dialect_cfg.get("speed", 1.0)

        def tts_loop():
            while True:
                piece = sq.get()
                if piece is None: return
                if turn != sess.turn: continue
                clean = re.sub(r"\[.*?\]|[*_#`]", "", piece).strip()
                if not clean: continue

                # Apply conversational prosody and Tanglish IPA phoneme mapping
                formatted_piece = tanglish_engine.format_conversational_prosody(clean)
                phonemes = tanglish_engine.text_to_tanglish_phonemes(formatted_piece, kokoro.g2p)

                with tts_lock:
                    try:
                        out = KPipeline.infer(kokoro.model, phonemes, BLENDED_VOICE, speed=current_speed)
                        if turn == sess.turn and out and out.audio is not None:
                            arr = out.audio.detach().cpu().numpy()
                            pcm = (np.clip(arr, -1, 1) * 32767).astype(np.int16)
                            if not tm["sent"]:
                                tm["sent"] = True; now = time.time()
                                sess.emit("json", {
                                    "type": "timing", "turn": turn,
                                    "stt": round(t1 - t0, 3),
                                    "llm": round(tm["llm"] - t1, 3),
                                    "tts": round(now - tm["llm"], 3),
                                    "total": round(now - t0, 3)
                                })
                            sess.emit("bin", struct.pack("<I", turn) + pcm.tobytes())
                    except Exception as err:
                        print("TTS synthesis error:", err)

        tt = threading.Thread(target=tts_loop, daemon=True); tt.start()

        first, buf, full = True, "", ""
        tag_checked = False
        sleep_intent_detected = False

        for chunk in ollama.chat(model=LLM, messages=turn_history, stream=True, keep_alive="30m",
                                 options={"temperature": 0.72, "num_predict": 120, "num_ctx": 2048}):
            if turn != sess.turn: break
            tok = chunk["message"]["content"]
            buf += tok
            full += tok

            if not tag_checked:
                stripped = buf.lstrip()
                if len(stripped) < 8 and stripped.startswith("["):
                    continue
                tag_checked = True
                if stripped.upper().startswith("[SLEEP]"):
                    sleep_intent_detected = True
                    sess.sleep_after_playback = True
                    print("[Intent] Semantic sleep intent detected by LLM!")
                    buf = re.sub(r"^\s*\[SLEEP\]\s*", "", buf, flags=re.I)
                    full = re.sub(r"^\s*\[SLEEP\]\s*", "", full, flags=re.I)
                if buf:
                    sess.emit("json", {"type": "reply", "turn": turn, "text": buf})
            else:
                sess.emit("json", {"type": "reply", "turn": turn, "text": tok})

            while (m := (FIRST if first else REST).match(buf)):
                if first: tm["llm"] = time.time()
                sq.put(m.group(1)); buf = buf[m.end():]; first = False

        if turn == sess.turn:
            if not tag_checked:
                if buf.lstrip().upper().startswith("[SLEEP]"):
                    sleep_intent_detected = True
                    sess.sleep_after_playback = True
                    buf = re.sub(r"^\s*\[SLEEP\]\s*", "", buf, flags=re.I)
                    full = re.sub(r"^\s*\[SLEEP\]\s*", "", full, flags=re.I)
                if buf.strip():
                    sess.emit("json", {"type": "reply", "turn": turn, "text": buf})
            if buf.strip():
                if first: tm["llm"] = time.time()
                sq.put(buf.strip())

        sq.put(None); tt.join()
        if turn == sess.turn and full.strip():
            history.append({"role": "assistant", "content": full.strip()})

        if turn == sess.turn:
            sess.processing_turn = False
            sess.emit("json", {"type": "done", "turn": turn})
            if sleep_intent_detected and not sess.lyla_speaking:
                sess.loop.call_soon_threadsafe(sess.go_to_sleep)
    except Exception as e:
        print("run_turn error:", e)
        if turn == sess.turn:
            sess.processing_turn = False
            sess.lyla_speaking = False
        if sess.awake and not sess.is_busy:
            sess.loop.call_soon_threadsafe(sess.reset_sleep_timer)
        sess.emit("json", {"type": "error", "turn": turn, "text": str(e)})

# ================= FASTAPI WEB SERVER =================
app = FastAPI(title="Lyla Neural Voice Assistant")

@app.get("/")
async def index():
    return FileResponse(os.path.join(BASE, "static", "index.html"))

@app.get("/health")
async def health():
    return {
        "ok": True,
        "device": DEVICE,
        "enrolled": OWNER_EMBED is not None,
        "dialects": list(tanglish_engine.DIALECTS.keys())
    }

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    global OWNER_EMBED
    await ws.accept()

    loop = asyncio.get_running_loop()
    sess = Session(ws, loop)
    old = ACTIVE["sess"]; ACTIVE["sess"] = sess
    if old:
        old.turn += 1
        old.cancel_sleep_timer()
        try: await old.ws.close(code=4000)
        except Exception: pass

    async def sender():
        try:
            while True:
                kind, payload = await sess.out.get()
                if kind == "json": await ws.send_text(json.dumps(payload))
                else: await ws.send_bytes(payload)
        except Exception:
            pass
    send_task = asyncio.create_task(sender())
    sess.out.put_nowait(("json", {
        "type": "init",
        "awake": False,
        "enrolled": OWNER_EMBED is not None,
        "dialect": sess.dialect,
        "dialects": [
            {"id": k, "name": v["name"], "tag": v["tag"]} 
            for k, v in tanglish_engine.DIALECTS.items()
        ]
    }))

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect": break

            if msg.get("bytes") is not None:
                process_audio_bytes(sess, msg["bytes"])

            elif msg.get("text"):
                data = json.loads(msg["text"])
                mtype = data.get("type")

                if mtype == "set_dialect":
                    target_dialect = data.get("dialect")
                    if target_dialect in tanglish_engine.DIALECTS:
                        sess.dialect = target_dialect
                        dialect_cfg = tanglish_engine.DIALECTS[target_dialect]
                        print(f"[WebSocket] User selected dialect: {dialect_cfg['name']}")
                        sess.emit("json", {
                            "type": "dialect",
                            "dialect": sess.dialect,
                            "name": dialect_cfg["name"],
                            "tag": dialect_cfg["tag"]
                        })

                elif mtype == "enroll_start":
                    sess.enroll_mode = True
                    sess.enroll_frames = []
                    sess.user_speaking = False
                    sess.speech_frames = []

                elif mtype == "enroll_end":
                    sess.enroll_mode = False
                    raw = b"".join(sess.enroll_frames)
                    sess.enroll_frames = []
                    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    if len(audio) < 2.0 * SR_IN:
                        sess.emit("json", {"type": "enroll_result", "ok": False,
                                           "msg": "Too short. Please speak for the full 5 seconds."})
                        continue
                    with spk_lock:
                        enhanced = enhance_far_field(audio, for_stt=False, noise_floor=sess.tracker.floor)
                        OWNER_EMBED = voice_encoder.embed_utterance(enhanced)
                        np.save(VOICEPRINT_PATH, OWNER_EMBED)
                    print("[Enrollment] Saved room-distance owner voiceprint!")
                    sess.emit("json", {"type": "enroll_result", "ok": True})

                elif mtype == "playback_done":
                    sess.lyla_speaking = False
                    sess.barge_streak = 0
                    sess.barge_frames = []
                    if sess.sleep_after_playback:
                        sess.go_to_sleep()
                    elif sess.awake and not sess.user_speaking and not sess.processing_turn:
                        sess.reset_sleep_timer()
    except Exception:
        pass
    finally:
        sess.turn += 1
        sess.cancel_sleep_timer()
        send_task.cancel()
        if ACTIVE["sess"] is sess: ACTIVE["sess"] = None

def warmup():
    global LYLA_EMBED
    print("Warming up Lyla Neural Pipeline...")
    try:
        # Pre-synthesize reference greeting with blended voice pack
        ref_ps = tanglish_engine.text_to_tanglish_phonemes("Vanakkam Kavin, I am Lyla, your personal AI companion.", kokoro.g2p)
        out = KPipeline.infer(kokoro.model, ref_ps, BLENDED_VOICE, speed=1.0)
        if out and out.audio is not None:
            arr = out.audio.detach().cpu().numpy()
            lyla_16k = resample_24k_to_16k(arr)
            LYLA_EMBED = voice_encoder.embed_utterance(enhance_far_field(lyla_16k, for_stt=False))
            print("Extracted Lyla's reference voiceprint for self-echo rejection!")

        if silero_model is not None:
            dummy_tracker = NoiseFloorTracker()
            frame_speech_probability(np.zeros(FRAME_SAMPLES, dtype=np.float32), dummy_tracker)

        transcribe(np.zeros(SR_IN, dtype=np.float32))
        ollama.chat(model=LLM, messages=[{"role": "user", "content": "hi"}],
                    keep_alive="30m", options={"num_predict": 1, "num_ctx": 2048})
        print("Warm-up complete! All neural models cached in VRAM.")
    except Exception as e:
        print("Warm-up warning:", e)

if __name__ == "__main__":
    warmup()
    print(f"Lyla Live Voice Server listening on http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")

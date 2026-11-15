import os, re, json, time, queue, struct, asyncio, threading
from collections import deque
import numpy as np
import soundfile as sf
import torch
import ollama
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from kokoro import KPipeline
from resemblyzer import VoiceEncoder

# ================= CONFIG =================
LLM = "llama3.2:3b"
VOICE, SPEED = "af_heart", 1.0
SR_IN = 16000
FRAME_SAMPLES = 512                            # Exact 32ms frame at 16kHz (required by Silero VAD)
FRAME_BYTES = FRAME_SAMPLES * 2
MAX_HISTORY = 12
SLEEP_TIMEOUT = 30.0
STT_BACKEND = "parakeet"                       # "parakeet" or "whisper"
PARAKEET_MODEL = "nemo-parakeet-tdt-0.6b-v2"
WHISPER_MODEL = "small.en"
VOCAB = "Lyla, Laila, Kavin, Ollama, Whisper, Kokoro, Python"
HOST, PORT = "127.0.0.1", 8000
BASE = os.path.dirname(os.path.abspath(__file__))

VOICEPRINT_PATH = os.path.join(BASE, "owner_voice.npy")

# Gemini-Live-style frame timing (1 frame = 32ms)
PRE_ROLL_FRAMES       = 15                     # 480ms pre-roll so first syllable is never clipped
IDLE_START_FRAMES     = 4                      # 128ms of speech to start listening
END_SILENCE_FRAMES    = 22                     # 704ms pause after speech to send turn
MIN_UTTERANCE_FRAMES  = 14                     # 448ms minimum valid utterance
BARGE_SPEECH_FRAMES   = 6                      # 192ms of clean post-AEC3 owner speech to interrupt

# Neural VAD & Speaker Verification Thresholds
VAD_PROB_IDLE         = 0.50
VAD_PROB_BARGE        = 0.65
MIN_SPEECH_RMS        = 0.012
SIMILARITY_NORMAL     = 0.52
OWNER_MARGIN_NORMAL   = 0.10
BARGE_OWNER_MIN       = 0.58
BARGE_MARGIN_MIN      = 0.04

torch.set_num_threads(max(2, (os.cpu_count() or 4) // 2))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("TTS device:", DEVICE)

SYSTEM_RULES = (
    "\n\nIMPORTANT:\n"
    "1. The user is speaking from across the room, so speech recognition may occasionally "
    "drop or mishear a word. Infer what they most likely meant from context. Only if it truly "
    "makes no sense, say what you think you heard and ask casually in one short line.\n"
    "2. SLEEP INTENT: If the user tells you to go to sleep, take a nap, rest, stop listening, "
    "stand by, or indicates they are done talking to you for now (in any phrasing), you MUST "
    "begin your response with the exact tag [SLEEP] followed by a brief, natural goodbye "
    "(for example: '[SLEEP] Going to sleep. Just call my name when you need me.'). "
    "Do NOT use [SLEEP] if they are merely talking about feeling sleepy or tired themselves."
)
PERSONA = open(os.path.join(BASE, "persona.txt"), encoding="utf-8").read() + SYSTEM_RULES

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
        print("Silero VAD unavailable, using spectral-harmonic fallback VAD:", e)

def frame_speech_probability(frame_f32: np.ndarray) -> float:
    """Returns speech probability [0.0, 1.0] for a single 32ms (512-sample) 16kHz frame."""
    rms = float(np.sqrt(np.mean(frame_f32 ** 2)))
    if rms < MIN_SPEECH_RMS:
        return 0.0

    if silero_model is not None:
        with torch.no_grad():
            prob = float(silero_model(torch.from_numpy(frame_f32), SR_IN).item())
        return prob

    # Fallback spectral-harmonic speech check (rejects whistles, pure tones, and broadband noise)
    spec = np.abs(np.fft.rfft(frame_f32 * np.hanning(len(frame_f32)))) ** 2
    freqs = np.fft.rfftfreq(len(frame_f32), d=1.0 / SR_IN)
    voice_band = spec[(freqs >= 180.0) & (freqs <= 3600.0)]
    total_e = float(np.sum(spec)) + 1e-12
    voice_ratio = float(np.sum(voice_band)) / total_e
    peak_ratio = float(np.max(voice_band)) / (float(np.sum(voice_band)) + 1e-12) if len(voice_band) else 1.0
    if voice_ratio > 0.65 and peak_ratio < 0.45:
        return min(1.0, voice_ratio)
    return 0.0

# ================= SPEAKER & ASR MODELS =================
print("Loading voice encoder...")
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
    mask = (freqs >= 150.0) & (freqs <= 3800.0)
    return np.fft.irfft(spec * mask, n=n).astype(np.float32)

def enhance_far_field(audio: np.ndarray, for_stt: bool = False, max_gain: float = 6.0) -> np.ndarray:
    if len(audio) == 0:
        return audio
    filtered = speech_bandpass(audio - np.mean(audio))
    rms = float(np.sqrt(np.mean(filtered ** 2)))
    if rms < 0.002:
        return filtered

    gain = min(max_gain, 0.090 / rms)
    boosted = np.tanh((filtered * gain) * 1.12) / np.tanh(1.12)

    if for_stt and len(boosted) > 2:
        boosted = np.append(boosted[0], boosted[1:] - 0.55 * boosted[:-1])

    peak = float(np.abs(boosted).max())
    if peak > 1e-3:
        boosted = boosted * (0.92 / peak)
    return boosted.astype(np.float32)

def verify_owner_voice(audio_16k: np.ndarray, max_gain: float = 6.0):
    """Returns (owner_sim, lyla_sim, rms)."""
    if len(audio_16k) < int(0.18 * SR_IN):
        return 0.0, 0.0, 0.0

    filtered = speech_bandpass(audio_16k - np.mean(audio_16k))
    rms = float(np.sqrt(np.mean(filtered ** 2)))
    if rms < MIN_SPEECH_RMS:
        return 0.0, 0.0, rms

    if OWNER_EMBED is None:
        return 1.0, 0.0, rms

    try:
        enhanced = enhance_far_field(filtered, for_stt=False, max_gain=max_gain)
        emb = voice_encoder.embed_utterance(enhanced)
        owner_sim = float(np.dot(OWNER_EMBED, emb))
        lyla_sim  = float(np.dot(LYLA_EMBED, emb)) if LYLA_EMBED is not None else 0.0
        return owner_sim, lyla_sim, rms
    except Exception:
        return 0.0, 0.0, rms

if STT_BACKEND == "parakeet":
    import onnx_asr
    _asr = onnx_asr.load_model(PARAKEET_MODEL)

    def transcribe(audio):
        try:
            return str(_asr.recognize(audio)).strip()
        except Exception:
            sf.write("_tmp.wav", audio, SR_IN)
            return str(_asr.recognize("_tmp.wav")).strip()
else:
    from faster_whisper import WhisperModel
    _w_device = "cuda" if torch.cuda.is_available() else "cpu"
    _w_compute = "float16" if _w_device == "cuda" else "int8"
    _w = WhisperModel(WHISPER_MODEL, device=_w_device, compute_type=_w_compute)

    def transcribe(audio):
        segs, _ = _w.transcribe(
            audio,
            language="en",
            beam_size=5,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt="Vocabulary: " + VOCAB
        )
        return " ".join(s.text for s in segs).strip()

kokoro = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=DEVICE)

LYLA_PATTERN = re.compile(
    r"\b(lyla|lylah|layla|laylah|lila|lilah|leila|leilah|laila|leyla|lyler|liler|lie\s?la|la\s?la|lala|isla)\b",
    re.I
)

FIXES = {
    LYLA_PATTERN.pattern: "Lyla",
    r"\bo\s?llama\b": "Ollama"
}

def fix_vocab(t):
    for pat, rep in FIXES.items():
        t = re.sub(pat, rep, t, flags=re.I)
    return t

def has_wake_word(text):
    return bool(re.search(r"\blyla\b", text, re.I))

FIRST = re.compile(r"^(.{12,}?[,.!?…;:])\s")
REST  = re.compile(r"^(.+?[.!?…])\s")

history = [{"role": "system", "content": PERSONA}]
stt_lock, tts_lock, spk_lock = threading.Lock(), threading.Lock(), threading.Lock()

# ================= UNIFIED FULL-DUPLEX SESSION =================
class Session:
    def __init__(self, ws, loop):
        self.ws, self.loop = ws, loop
        self.turn = 0
        self.awake = False
        self.sleep_timer = None
        self.sleep_after_playback = False

        # Pipeline activity flags
        self.processing_turn = False
        self.lyla_speaking = False
        self.enroll_mode = False
        self.enroll_frames = []

        # Unified VAD State Machine buffers
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
            owner_sim, lyla_sim, rms = verify_owner_voice(audio, max_gain=5.0)

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
    """Processes incoming 16kHz PCM stream in exact 32ms (512-sample) frames."""
    if sess.enroll_mode:
        sess.enroll_frames.append(chunk)
        return

    sess.pcm_remainder.extend(chunk)
    while len(sess.pcm_remainder) >= FRAME_BYTES:
        raw_frame = bytes(sess.pcm_remainder[:FRAME_BYTES])
        del sess.pcm_remainder[:FRAME_BYTES]

        frame_f32 = np.frombuffer(raw_frame, dtype=np.int16).astype(np.float32) / 32768.0
        prob = frame_speech_probability(frame_f32)

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
                    sess.emit("json", {"type": "vad", "speaking": True})
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
                    # Utterance complete!
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
            owner_sim, lyla_sim, rms = verify_owner_voice(audio, max_gain=6.5)

        if owner_sim == 0.0:
            if turn == sess.turn:
                sess.processing_turn = False
            if sess.awake and not sess.is_busy:
                sess.loop.call_soon_threadsafe(sess.reset_sleep_timer)
            sess.emit("json", {"type": "ignored"})
            return

        margin = owner_sim - lyla_sim
        is_owner = (
            OWNER_EMBED is None
            or (owner_sim >= SIMILARITY_NORMAL and margin >= OWNER_MARGIN_NORMAL)
        )
        print(f"[Voice verify] rms={rms:.4f} owner={owner_sim:.3f}, lyla={lyla_sim:.3f}, margin={margin:+.3f} -> {'OWNER' if is_owner else 'IGNORED'}")

        if not is_owner:
            if turn == sess.turn:
                sess.processing_turn = False
            if sess.awake and not sess.is_busy:
                sess.loop.call_soon_threadsafe(sess.reset_sleep_timer)
            sess.emit("json", {"type": "ignored"})
            return

        stt_audio = enhance_far_field(audio, for_stt=True, max_gain=7.5)

        t0 = time.time()
        with stt_lock:
            text = fix_vocab(transcribe(stt_audio))
        t1 = time.time()
        print(f"[STT]: '{text}'")

        if len(text) < 2 or turn != sess.turn:
            if turn == sess.turn:
                sess.processing_turn = False
            if sess.awake and not sess.is_busy:
                sess.loop.call_soon_threadsafe(sess.reset_sleep_timer)
            sess.emit("json", {"type": "ignored"})
            return

        if not sess.awake:
            if has_wake_word(text):
                sess.loop.call_soon_threadsafe(sess.wake_up)
            else:
                print(f"[Ignored while asleep]: {text}")
                if turn == sess.turn:
                    sess.processing_turn = False
                sess.emit("json", {"type": "ignored"})
                return

        if turn != sess.turn:
            return

        sess.loop.call_soon_threadsafe(sess.cancel_sleep_timer)
        sess.lyla_speaking = True
        sess.sleep_after_playback = False
        sess.barge_streak = 0
        sess.barge_frames = []
        sess.emit("json", {"type": "transcript", "turn": turn, "text": text})

        history.append({"role": "user", "content": text})
        extra = len(history) - 1 - MAX_HISTORY
        if extra > 0: del history[1:1 + extra]

        sq = queue.Queue()
        tm = {"llm": None, "sent": False}

        def tts_loop():
            while True:
                piece = sq.get()
                if piece is None: return
                if turn != sess.turn: continue
                clean = re.sub(r"\[.*?\]|[*_#`]", "", piece).strip()
                if not clean: continue
                with tts_lock:
                    for _, _, a in kokoro(clean, voice=VOICE, speed=SPEED):
                        if turn != sess.turn: break
                        if a is None: continue
                        arr = a.detach().cpu().numpy()
                        pcm = (np.clip(arr, -1, 1) * 32767).astype(np.int16)
                        if not tm["sent"]:
                            tm["sent"] = True; now = time.time()
                            sess.emit("json", {"type": "timing", "turn": turn,
                                               "stt": round(t1 - t0, 2),
                                               "llm": round(tm["llm"] - t1, 2),
                                               "tts": round(now - tm["llm"], 2),
                                               "total": round(now - t0, 2)})
                        sess.emit("bin", struct.pack("<I", turn) + pcm.tobytes())

        tt = threading.Thread(target=tts_loop, daemon=True); tt.start()

        first, buf, full = True, "", ""
        tag_checked = False
        sleep_intent_detected = False

        for chunk in ollama.chat(model=LLM, messages=history, stream=True, keep_alive="30m",
                                 options={"temperature": 0.7, "num_predict": 150,
                                          "num_ctx": 2048}):
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

# ================= WEB APP =================
app = FastAPI()

@app.get("/")
async def index():
    return FileResponse(os.path.join(BASE, "static", "index.html"))

@app.get("/health")
async def health():
    return {"ok": True, "device": DEVICE, "enrolled": OWNER_EMBED is not None}

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
        "enrolled": OWNER_EMBED is not None
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

                if mtype == "enroll_start":
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
                        enhanced = enhance_far_field(audio, for_stt=False)
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
    print("Warming up...")
    try:
        chunks = [a.detach().cpu().numpy() for _, _, a in kokoro(
            "Hello Kavin, I am Lyla, your personal voice assistant ready to help you today.", voice=VOICE) if a is not None]
        if chunks:
            lyla_16k = resample_24k_to_16k(np.concatenate(chunks))
            LYLA_EMBED = voice_encoder.embed_utterance(enhance_far_field(lyla_16k, for_stt=False))
            print("Extracted Lyla's own voiceprint for self-echo rejection!")

        if silero_model is not None:
            frame_speech_probability(np.zeros(FRAME_SAMPLES, dtype=np.float32))
        transcribe(np.zeros(SR_IN, dtype=np.float32))
        ollama.chat(model=LLM, messages=[{"role": "user", "content": "hi"}],
                    keep_alive="30m", options={"num_predict": 1, "num_ctx": 2048})
    except Exception as e:
        print("Warm-up warning:", e)

if __name__ == "__main__":
    warmup()
    print("Connect to the server!!")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
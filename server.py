import os, re, json, time, queue, struct, asyncio, threading
import numpy as np
import soundfile as sf
import torch
import ollama
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from kokoro import KPipeline
from resemblyzer import VoiceEncoder, preprocess_wav

# ================= CONFIG =================
LLM = "llama3.2:3b"
VOICE, SPEED = "af_heart", 1.0
SR_IN = 16000
MAX_HISTORY = 12
SLEEP_TIMEOUT = 30.0
STT_BACKEND = "parakeet"                       # "parakeet" or "whisper"
PARAKEET_MODEL = "nemo-parakeet-tdt-0.6b-v2"
WHISPER_MODEL = "small.en"
VOCAB = "Lyla, Kavin, Ollama, Whisper, Kokoro, Python"
HOST, PORT = "127.0.0.1", 8000
BASE = os.path.dirname(os.path.abspath(__file__))

VOICEPRINT_PATH = os.path.join(BASE, "owner_voice.npy")
SIMILARITY_NORMAL = 0.54
SIMILARITY_BARGE_IN = 0.45

torch.set_num_threads(max(2, (os.cpu_count() or 4) // 2))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("TTS device:", DEVICE)

SYSTEM_RULES = (
    "\n\nIMPORTANT:\n"
    "1. The user's words come from speech recognition and may contain mishearings. "
    "Infer what they most likely meant from context. Only if it truly makes no sense, "
    "say what you think you heard and ask casually in one short line.\n"
    "2. SLEEP INTENT: If the user tells you to go to sleep, take a nap, rest, stop listening, "
    "stand by, or indicates they are done talking to you for now (in any phrasing), you MUST "
    "begin your response with the exact tag [SLEEP] followed by a brief, natural goodbye "
    "(for example: '[SLEEP] Going to sleep. Just call my name when you need me.'). "
    "Do NOT use [SLEEP] if they are merely talking about feeling sleepy or tired themselves."
)
PERSONA = open(os.path.join(BASE, "persona.txt"), encoding="utf-8").read() + SYSTEM_RULES

# ================= MODELS =================
print("Loading voice encoder...")
voice_encoder = VoiceEncoder(device="cpu")

def load_owner_embed():
    if os.path.exists(VOICEPRINT_PATH):
        return np.load(VOICEPRINT_PATH)
    return None

OWNER_EMBED = load_owner_embed()

def get_voice_similarity(audio_16k: np.ndarray) -> float:
    if OWNER_EMBED is None or len(audio_16k) < int(0.25 * SR_IN):
        return 1.0
    try:
        wav = preprocess_wav(audio_16k, source_sr=SR_IN)
        if len(wav) < int(0.22 * SR_IN):
            return 0.0
        emb = voice_encoder.embed_utterance(wav)
        return float(np.dot(OWNER_EMBED, emb))
    except Exception:
        return 0.0

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
    _w = WhisperModel(WHISPER_MODEL, compute_type="int8")

    def transcribe(audio):
        segs, _ = _w.transcribe(audio, language="en", beam_size=5,
                                condition_on_previous_text=False,
                                initial_prompt="Vocabulary: " + VOCAB)
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

def normalize(a):
    peak = float(np.abs(a).max())
    return a * (0.9 / peak) if peak > 1e-3 else a

FIRST = re.compile(r"^(.{12,}?[,.!?…;:])\s")
REST  = re.compile(r"^(.+?[.!?…])\s")

history = [{"role": "system", "content": PERSONA}]
stt_lock, tts_lock, spk_lock = threading.Lock(), threading.Lock(), threading.Lock()

# ================= SESSION =================
class Session:
    def __init__(self, ws, loop):
        self.ws, self.loop = ws, loop
        self.turn = 0
        self.frames = []
        self.recording = False
        self.enroll_mode = False
        self.lyla_speaking = False
        self.sleep_after_playback = False
        self.bytes_since_check = 0
        self.checking_barge_in = False
        self.out = asyncio.Queue()
        self.awake = False
        self.sleep_timer = None

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
        if self.awake and not self.sleep_after_playback:
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

def sliding_window_barge_in(sess, window_bytes):
    try:
        audio = np.frombuffer(window_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        with spk_lock:
            sim = get_voice_similarity(audio)
        print(f"[Sliding barge-in] sim={sim:.3f} (need >= {SIMILARITY_BARGE_IN})")

        if sim >= SIMILARITY_BARGE_IN and sess.lyla_speaking:
            print(">>> OWNER DETECTED! Interrupting Lyla immediately! <<<")
            sess.lyla_speaking = False
            sess.sleep_after_playback = False
            sess.turn += 1
            sess.drain()
            keep_bytes = int(0.75 * SR_IN * 2)
            raw = b"".join(sess.frames)
            sess.frames = [raw[-keep_bytes:]] if len(raw) > keep_bytes else [raw]
            sess.recording = True
            sess.emit("json", {"type": "interrupt"})
            sess.emit("json", {"type": "turn", "id": sess.turn})
    finally:
        sess.checking_barge_in = False

def run_turn(sess, audio, turn):
    try:
        with spk_lock:
            sim = get_voice_similarity(audio)
        print(f"[Voice verify] sim={sim:.3f} (need >= {SIMILARITY_NORMAL})")

        if sim < SIMILARITY_NORMAL:
            print("[Rejected] Audio did not match owner's voiceprint.")
            if sess.awake and not sess.lyla_speaking:
                sess.loop.call_soon_threadsafe(sess.reset_sleep_timer)
            sess.emit("json", {"type": "ignored"})
            return

        t0 = time.time()
        with stt_lock:
            text = fix_vocab(transcribe(audio))
        t1 = time.time()

        if len(text) < 2 or turn != sess.turn:
            if sess.awake:
                sess.loop.call_soon_threadsafe(sess.reset_sleep_timer)
            sess.emit("json", {"type": "ignored"})
            return

        # Wake-word check when sleeping
        if not sess.awake:
            if has_wake_word(text):
                sess.loop.call_soon_threadsafe(sess.wake_up)
            else:
                print(f"[Ignored while asleep]: {text}")
                sess.emit("json", {"type": "ignored"})
                return

        if turn != sess.turn:
            return

        sess.loop.call_soon_threadsafe(sess.cancel_sleep_timer)
        sess.lyla_speaking = True
        sess.sleep_after_playback = False
        sess.bytes_since_check = 0
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
                        pcm = (np.clip(a.detach().cpu().numpy(), -1, 1) * 32767).astype(np.int16)
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

            # Hold initial tokens briefly until we verify whether the LLM emitted [SLEEP]
            if not tag_checked:
                stripped = buf.lstrip()
                if len(stripped) < 8 and stripped.startswith("["):
                    continue
                tag_checked = True
                if stripped.upper().startswith("[SLEEP]"):
                    sleep_intent_detected = True
                    sess.sleep_after_playback = True
                    print("[Intent] Semantic sleep intent detected by LLM!")
                    # Strip [SLEEP] tag from both stream buffer and full text
                    buf = re.sub(r"^\s*\[SLEEP\]\s*", "", buf, flags=re.I)
                    full = re.sub(r"^\s*\[SLEEP\]\s*", "", full, flags=re.I)
                if buf:
                    sess.emit("json", {"type": "reply", "turn": turn, "text": buf})
            else:
                sess.emit("json", {"type": "reply", "turn": turn, "text": tok})

            while (m := (FIRST if first else REST).match(buf)):
                if first: tm["llm"] = time.time()
                sq.put(m.group(1)); buf = buf[m.end():]; first = False

        # Flush any remaining buffered text (even if total reply was < 8 chars)
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
            sess.emit("json", {"type": "done", "turn": turn})
            # Fallback: if audio playback already finished or no audio was generated, sleep right away
            if sleep_intent_detected and not sess.lyla_speaking:
                sess.loop.call_soon_threadsafe(sess.go_to_sleep)
    except Exception as e:
        print("run_turn error:", e)
        sess.lyla_speaking = False
        if sess.awake:
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

    WINDOW_BYTES = int(0.48 * SR_IN * 2)
    STEP_BYTES   = int(0.20 * SR_IN * 2)

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect": break

            if msg.get("bytes") is not None:
                chunk = msg["bytes"]
                if sess.enroll_mode and sess.recording:
                    sess.frames.append(chunk)
                    continue

                if sess.recording:
                    sess.frames.append(chunk)

                    if sess.lyla_speaking:
                        raw = b"".join(sess.frames)
                        if len(raw) > int(1.2 * SR_IN * 2):
                            raw = raw[-int(1.2 * SR_IN * 2):]
                            sess.frames = [raw]

                        sess.bytes_since_check += len(chunk)
                        if len(raw) >= WINDOW_BYTES and sess.bytes_since_check >= STEP_BYTES and not sess.checking_barge_in:
                            sess.bytes_since_check = 0
                            sess.checking_barge_in = True
                            window = raw[-WINDOW_BYTES:]
                            threading.Thread(target=sliding_window_barge_in,
                                             args=(sess, window), daemon=True).start()

            elif msg.get("text"):
                data = json.loads(msg["text"])
                mtype = data.get("type")

                if mtype == "enroll_start":
                    sess.enroll_mode = True
                    sess.frames = []
                    sess.recording = True

                elif mtype == "enroll_end":
                    sess.recording = False
                    sess.enroll_mode = False
                    raw = b"".join(sess.frames); sess.frames = []
                    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    if len(audio) < 1.5 * SR_IN:
                        sess.emit("json", {"type": "enroll_result", "ok": False,
                                           "msg": "Too short. Please speak for at least 3 seconds."})
                        continue
                    with spk_lock:
                        wav = preprocess_wav(normalize(audio), source_sr=SR_IN)
                        OWNER_EMBED = voice_encoder.embed_utterance(wav)
                        np.save(VOICEPRINT_PATH, OWNER_EMBED)
                    print("[Enrollment] Saved owner voiceprint!")
                    sess.emit("json", {"type": "enroll_result", "ok": True})

                elif mtype == "start":
                    if sess.enroll_mode:
                        continue
                    sess.cancel_sleep_timer()
                    sess.frames = []
                    sess.recording = True
                    sess.bytes_since_check = 0
                    if not sess.lyla_speaking:
                        sess.turn += 1
                        if sess.awake:
                            sess.drain()
                        sess.emit("json", {"type": "turn", "id": sess.turn})

                elif mtype == "end" and sess.recording:
                    if sess.enroll_mode:
                        continue
                    sess.recording = False
                    if sess.lyla_speaking:
                        sess.frames = []
                        continue
                    raw = b"".join(sess.frames); sess.frames = []
                    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    if len(audio) < 0.3 * SR_IN:
                        if sess.awake:
                            sess.reset_sleep_timer()
                        sess.emit("json", {"type": "ignored"})
                        continue
                    threading.Thread(target=run_turn, args=(sess, normalize(audio), sess.turn),
                                     daemon=True).start()

                elif mtype == "playback_done":
                    sess.lyla_speaking = False
                    if sess.sleep_after_playback:
                        sess.go_to_sleep()
                    elif sess.awake and not sess.recording:
                        sess.reset_sleep_timer()
    except Exception:
        pass
    finally:
        sess.turn += 1
        sess.cancel_sleep_timer()
        send_task.cancel()
        if ACTIVE["sess"] is sess: ACTIVE["sess"] = None

def warmup():
    print("Warming up...")
    try:
        list(kokoro("Hello there.", voice=VOICE))
        transcribe(np.zeros(SR_IN, dtype=np.float32))
        voice_encoder.embed_utterance(np.zeros(SR_IN, dtype=np.float32))
        ollama.chat(model=LLM, messages=[{"role": "user", "content": "hi"}],
                    keep_alive="30m", options={"num_predict": 1, "num_ctx": 2048})
    except Exception as e:
        print("Warm-up warning:", e)

if __name__ == "__main__":
    warmup()
    print("Connect to the server!!")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
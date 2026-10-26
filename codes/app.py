"""LeLa: agentic 1-on-1 Kannada tutor (FastAPI + WebSocket).

Pipeline per student turn:  audio/text -> ASR -> LessonSession (LLM + guardrails) -> TTS -> browser.
The lesson logic lives in tutor.py; this file only wires models and transport.

Config (env vars):
  LELA_LLM_URL      vLLM OpenAI endpoint     (default http://127.0.0.1:8001/v1/chat/completions)
  LELA_LLM_MODEL    model name served by vLLM (default meta-llama/Meta-Llama-3.1-8B-Instruct)
  LELA_ASR_KN       "indicconformer" | "whisper"   (Kannada ASR backend, default indicconformer)
  LELA_ASR_EN       faster-whisper size for English questions (default "small"; reused if LELA_ASR_KN=whisper)
  LELA_TTS          "mms" (default, Meta MMS-TTS Kannada: fast) | "parler" (AI4Bharat Indic Parler-TTS: nicer, slow)
                    | "indicf5" (voice clone of prompts/kannada_teacher.wav)
  LELA_TTS_RATE     MMS speaking rate (default 0.85: a little slower than normal, for a learner)
  LELA_TTS_VOICE    Parler voice description (default: Anu, slow and clear)
  LELA_LLM_KANNADA  0 (default): LeLa speaks only hand-checked Kannada; 1: also speak the LLM's own Kannada
  LELA_CONFORMER_DEVICE  "auto" (GPU, falls back to CPU if onnxruntime's CUDA/cuDNN fails) | "cpu"
  LELA_DISABLE_CUDNN   default 1 (cuDNN off for PyTorch, as in your original; IndicF5 fails with
                       CUDNN_STATUS_NOT_INITIALIZED otherwise). Set 0 once torch/cuDNN versions are aligned.
"""

import asyncio
import io
import json
import os
import re
import threading
import traceback
from contextlib import asynccontextmanager

import httpx
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response

from translit import transliterate
from tutor import RESPONSE_SCHEMA, LessonSession, kannada_only, student_display

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_URL = os.getenv("LELA_LLM_URL", "http://127.0.0.1:8001/v1/chat/completions")
LLM_MODEL = os.getenv("LELA_LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
ASR_KN = os.getenv("LELA_ASR_KN", "indicconformer")
ASR_EN = os.getenv("LELA_ASR_EN", "small")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Only affects PyTorch (IndicF5). Whisper (CTranslate2) and IndicConformer (ONNX) don't use this flag.
if os.getenv("LELA_DISABLE_CUDNN", "1") == "1":
    torch.backends.cudnn.enabled = False

REF_AUDIO_PATH = os.path.join(BASE_DIR, "prompts", "kannada_teacher.wav")
# Must be the exact words spoken in kannada_teacher.wav, or IndicF5 output degrades.
REF_AUDIO_TEXT = os.getenv("LELA_REF_TEXT", "ನಮಸ್ಕಾರ, ಎಲ್ಲರಿಗೂ ಸುಸ್ವಾಗತ.")
TTS_BACKEND = os.getenv("LELA_TTS", "mms")
TTS_VOICE = os.getenv(
    "LELA_TTS_VOICE",
    "Anu speaks slowly and clearly at a slightly slow pace, with a clear, close-sounding recording of very high quality.",
)
TTS_SAMPLE_RATE = 24000  # IndicF5 outputs 24 kHz; Parler / MMS overwrite this from their config at load time.

with open(os.path.join(BASE_DIR, "lessons.json"), encoding="utf-8") as f:
    LESSON = json.load(f)

M: dict = {}                      # loaded models
ASR_LOCK = threading.Lock()       # GPU models are shared by all sessions; run one call at a time
TTS_LOCK = threading.Lock()
TTS_CACHE: dict[str, bytes] = {}  # lesson prompts repeat, so cache their audio
TTS_LAST_ERROR = {"error": None}


def load_conformer():
    """IndicConformer builds its own onnxruntime sessions and always asks for CUDA when torch sees a GPU.
    If onnxruntime's CUDA provider can't start (e.g. CUDNN_STATUS_NOT_INITIALIZED from a cuDNN
    version clash or a full GPU), build the remaining sessions on CPU instead of crashing."""
    import onnxruntime as ort
    from transformers import AutoModel

    orig_session = ort.InferenceSession
    state = {"cpu": os.getenv("LELA_CONFORMER_DEVICE", "auto") == "cpu", "providers": set()}

    def session(path, *args, providers=None, **kwargs):
        if not state["cpu"]:
            try:
                sess = orig_session(path, *args, providers=providers, **kwargs)
                state["providers"].add(sess.get_providers()[0])
                return sess
            except Exception as e:
                print(f"[Init] IndicConformer: onnxruntime GPU session failed, using CPU ({str(e).splitlines()[0][:160]})")
                state["cpu"] = True
        state["providers"].add("CPUExecutionProvider")
        return orig_session(path, *args, providers=["CPUExecutionProvider"], **kwargs)

    ort.InferenceSession = session  # the model's remote code looks this up at call time
    try:
        model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)
    finally:
        ort.InferenceSession = orig_session
    print(f"[Init] IndicConformer running on: {', '.join(sorted(state['providers']))}")
    return model


def load_models() -> None:
    from faster_whisper import WhisperModel
    from transformers import AutoModel

    compute = "float16" if DEVICE == "cuda" else "int8"
    if ASR_KN != "whisper":
        try:
            M["conformer"] = load_conformer()
        except Exception:
            traceback.print_exc()
            print("[Init] !!! IndicConformer failed to load; using Whisper large-v3 for Kannada instead")
    if "conformer" in M:
        M["whisper"] = WhisperModel(ASR_EN, device=DEVICE, compute_type=compute)
    else:
        M["whisper"] = WhisperModel("large-v3", device=DEVICE, compute_type=compute)
    load_tts()


def load_tts() -> None:
    global TTS_SAMPLE_RATE, SENTENCE_GAP
    if TTS_BACKEND == "parler":
        # pip install git+https://github.com/huggingface/parler-tts.git  (gated: accept terms on HF)
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(DEVICE)
        M["tts"] = model
        M["tts_tok"] = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
        M["tts_desc_tok"] = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
        TTS_SAMPLE_RATE = model.config.sampling_rate
    elif TTS_BACKEND == "mms":
        # Meta MMS (VITS). Small and fast, but CC-BY-NC 4.0: personal / non-commercial use only.
        from transformers import AutoTokenizer, VitsModel

        M["tts"] = VitsModel.from_pretrained("facebook/mms-tts-kan").to(DEVICE).eval()
        M["tts"].speaking_rate = float(os.getenv("LELA_TTS_RATE", "0.85"))  # <1 = slower
        M["tts_tok"] = AutoTokenizer.from_pretrained("facebook/mms-tts-kan")  # takes Kannada script directly
        TTS_SAMPLE_RATE = M["tts"].config.sampling_rate
    else:
        from transformers import AutoModel

        M["tts"] = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True).to(DEVICE)
    SENTENCE_GAP = np.zeros(int(0.25 * TTS_SAMPLE_RATE), dtype=np.float32)
    print(f"[Init] TTS: {TTS_BACKEND} at {TTS_SAMPLE_RATE} Hz")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print("[Init] Loading ASR + TTS models...")
    await asyncio.to_thread(load_models)
    if TTS_BACKEND == "indicf5":
        if not os.path.exists(REF_AUDIO_PATH):
            raise RuntimeError(f"Missing TTS reference clip: {REF_AUDIO_PATH}")
        ref_secs = sf.info(REF_AUDIO_PATH).duration
        print(f"[Init] TTS reference: {ref_secs:.1f} s clip, transcript '{REF_AUDIO_TEXT}'")
        if not 2 <= ref_secs <= 12:
            print("[Init] !!! The reference clip should be about 3-10 s long; trim it, or speech may drop words.")
    M["http"] = httpx.AsyncClient(timeout=30.0)
    # TTS self-test: fails loudly at startup instead of silently on every turn,
    # and warms the cache with the opening greeting.
    wav = await asyncio.to_thread(synthesize, LESSON["steps"][0]["tutor_kn"])
    if wav:
        print(f"[Init] TTS self-test OK ({len(wav) // 2 / TTS_SAMPLE_RATE:.1f} s of audio)")
    else:
        print(f"[Init] !!! TTS self-test FAILED: {TTS_LAST_ERROR['error']}  (the tutor will be silent)")
    print("[Init] Ready.")
    yield
    await M["http"].aclose()


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------- blocking model calls (run in threads)

def decode_audio(data: bytes, rate: int = 16000) -> np.ndarray:
    """Browser recording (webm/opus, mp4/aac, wav) -> float32 mono at `rate`, via PyAV.
    Not faster_whisper.audio.decode_audio: faster-whisper 1.2.x passes an argument PyAV 19+ removed."""
    import av

    resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
    chunks = []
    with av.open(io.BytesIO(data), mode="r") as container:
        for frame in container.decode(audio=0):
            chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
    chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]  # flush
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def transcribe(audio_bytes: bytes, lang: str) -> str:
    """Any browser format -> text. lang is "kn" or "en"."""
    try:
        audio = decode_audio(audio_bytes)
    except Exception as e:
        print(f"[ASR] decode failed: {e}")
        return ""
    if len(audio) < 16000 * 0.3:  # under 0.3 s: a click, not speech
        return ""
    with ASR_LOCK:
        if lang == "kn" and "conformer" in M:
            wav = torch.from_numpy(audio).unsqueeze(0)
            text = M["conformer"](wav, "kn", "rnnt")
            return (text[0] if isinstance(text, (list, tuple)) else str(text)).strip()
        segments, _ = M["whisper"].transcribe(
            audio, language=lang, beam_size=5, vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(s.text for s in segments).strip()


SENTENCE_GAP = np.zeros(int(0.25 * TTS_SAMPLE_RATE), dtype=np.float32)  # rebuilt in load_tts()
SENT_CACHE: dict[str, np.ndarray] = {}


def split_sentences(text: str) -> list[str]:
    """'ತುಂಬಾ ಒಳ್ಳೆಯದು! ನೀವು ಹೇಗಿದ್ದೀರಿ?' -> ['ತುಂಬಾ ಒಳ್ಳೆಯದು!', 'ನೀವು ಹೇಗಿದ್ದೀರಿ?']"""
    return [p.strip() for p in re.findall(r"[^.!?।]+[.!?।]*", text) if p.strip(" .!?।")]


def _run_tts(sentence: str):
    if TTS_BACKEND == "parler":
        desc = M["tts_desc_tok"](TTS_VOICE, return_tensors="pt").to(DEVICE)
        prompt = M["tts_tok"](sentence, return_tensors="pt").to(DEVICE)
        torch.manual_seed(0)  # Parler samples; a fixed seed keeps the voice the same between lines
        return M["tts"].generate(
            input_ids=desc.input_ids, attention_mask=desc.attention_mask,
            prompt_input_ids=prompt.input_ids, prompt_attention_mask=prompt.attention_mask,
        )
    if TTS_BACKEND == "mms":
        tok = M["tts_tok"]
        text = sentence
        if getattr(tok, "is_uroman", False):  # some MMS languages expect romanized input
            import uroman  # pip install uroman

            text = uroman.Uroman().romanize_string(sentence)
        inputs = tok(text, return_tensors="pt").to(DEVICE)
        torch.manual_seed(0)  # VITS samples durations; a fixed seed makes each line sound the same every time
        return M["tts"](**inputs).waveform
    return M["tts"](sentence, ref_audio_path=REF_AUDIO_PATH, ref_text=REF_AUDIO_TEXT)


def _tts_one(sentence: str) -> np.ndarray:
    """One sentence -> float32 audio in [-1, 1]. Caller holds TTS_LOCK."""
    if sentence in SENT_CACHE:
        return SENT_CACHE[sentence]
    with torch.no_grad():
        try:
            audio = _run_tts(sentence)
        except RuntimeError as e:
            if "CUDNN" not in str(e).upper() or not torch.backends.cudnn.enabled:
                raise
            print("[TTS] cuDNN error; disabling cuDNN for PyTorch and retrying")
            torch.backends.cudnn.enabled = False
            audio = _run_tts(sentence)
    if isinstance(audio, tuple):
        audio = audio[0]
    if hasattr(audio, "cpu"):
        audio = audio.cpu().numpy()
    audio = np.asarray(audio).squeeze()
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"IndicF5 returned empty/invalid audio (shape {audio.shape})")
    peak = float(np.abs(audio).max())
    if peak > 1.0:  # float output outside [-1, 1] would clip to noise in the WAV
        audio = audio / peak
    audio = audio.astype(np.float32)
    if len(SENT_CACHE) < 512:
        SENT_CACHE[sentence] = audio
    return audio


def synthesize(text: str) -> bytes:
    """Kannada text -> WAV bytes. Each sentence is synthesized separately and joined with a short
    pause: given several sentences at once, F5-style TTS can drop or reorder them."""
    text = kannada_only(text)
    if not text:
        return b""
    if text in TTS_CACHE:
        return TTS_CACHE[text]
    try:
        parts = []
        with TTS_LOCK:
            for sentence in split_sentences(text):
                if parts:
                    parts.append(SENTENCE_GAP)
                parts.append(_tts_one(sentence))
        buf = io.BytesIO()
        sf.write(buf, np.concatenate(parts), TTS_SAMPLE_RATE, format="WAV")
        wav = buf.getvalue()
    except Exception as e:
        TTS_LAST_ERROR["error"] = f"{type(e).__name__}: {e}"
        print(f"[TTS] synthesis failed for '{text}':")
        traceback.print_exc()
        return b""
    TTS_LAST_ERROR["error"] = None
    if len(TTS_CACHE) < 256:
        TTS_CACHE[text] = wav
    return wav


async def call_llm(messages: list[dict]) -> str:
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 400,
        # vLLM structured output: the reply is forced to match the schema.
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "lela_turn", "schema": RESPONSE_SCHEMA},
        },
    }
    resp = await M["http"].post(LLM_URL, json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------- transport

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.get("/tts_test")
async def tts_test(text: str = "ನಮಸ್ಕಾರ! ನೀವು ಹೇಗಿದ್ದೀರಿ?"):
    """Open http://localhost:7800/tts_test in the browser: plays the WAV, or shows the TTS error."""
    wav = await asyncio.to_thread(synthesize, text)
    if not wav:
        return JSONResponse({"ok": False, "error": TTS_LAST_ERROR["error"], "tts": TTS_BACKEND}, 500)
    return Response(wav, media_type="audio/wav")


async def send_turn(ws: WebSocket, turn) -> None:
    # Text first so the student can read while audio is synthesized.
    await ws.send_text(json.dumps(turn.to_json(), ensure_ascii=False))
    audio = await asyncio.to_thread(synthesize, turn.kn)
    if audio:
        await ws.send_bytes(audio)
    await ws.send_text(json.dumps({"type": "turn_end", "tts_error": None if audio else TTS_LAST_ERROR["error"]}))


@app.websocket("/ws/tutor")
async def websocket_tutor(ws: WebSocket):
    await ws.accept()
    session = LessonSession(LESSON, call_llm, llm_kannada=os.getenv("LELA_LLM_KANNADA", "0") == "1")
    next_audio_lang = "kn"
    try:
        await send_turn(ws, session.opening())  # tutor opens the class with a greeting
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            if msg.get("text") is not None:
                data = json.loads(msg["text"])
                if data.get("type") == "audio_lang":      # next audio clip is "kn" or "en"
                    next_audio_lang = "en" if data.get("lang") == "en" else "kn"
                    continue
                if data.get("type") != "text" or not data.get("text", "").strip():
                    continue
                student_text = data["text"].strip()
                # Typed input: Kannada script if it contains any, else treat as English.
                lang = "kn" if kannada_only(student_text) else "en"
            else:
                lang, next_audio_lang = next_audio_lang, "kn"
                student_text = await asyncio.to_thread(transcribe, msg["bytes"], lang)
                print(f"[ASR:{lang}] '{student_text}'")
                if not student_text:
                    retry_kn = "ದಯವಿಟ್ಟು ಮತ್ತೊಮ್ಮೆ ಹೇಳಿ."
                    await ws.send_text(json.dumps({
                        "type": "tutor",
                        "student_display": "(silence / inaudible)",
                        "teacher_kannada": retry_kn,
                        "romanized": transliterate(retry_kn),
                        "english_help": "Please say that again. (Hold the button while you speak.)",
                        "state": session.step["id"], "action": "repeat",
                    }, ensure_ascii=False))
                    audio = await asyncio.to_thread(synthesize, retry_kn)
                    if audio:
                        await ws.send_bytes(audio)
                    await ws.send_text(json.dumps({"type": "turn_end", "tts_error": None if audio else TTS_LAST_ERROR["error"]}))
                    continue
                # Show what was heard right away (English letters for Kannada), before the LLM answers.
                await ws.send_text(json.dumps({"type": "heard", "student_display": student_display(student_text, lang)},
                                              ensure_ascii=False))

            turn = await session.handle_student(student_text, lang)
            await send_turn(ws, turn)
    except WebSocketDisconnect:
        pass
    finally:
        print("[WS] session closed")

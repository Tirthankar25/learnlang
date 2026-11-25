"""LeLa: agentic 1-on-1 Kannada tutor (FastAPI + WebSocket).

Pipeline per student turn:  audio/text -> ASR -> LessonSession (LLM + guardrails) -> TTS -> browser.
The lesson logic lives in tutor.py (tutor persona: AuTut) and flow.py (the four-part lesson flow:
concept intro -> guided repetition -> micro-roleplay -> mastery check); this file only wires models
and transport. At the end of a lesson AuTut asks "next lesson, repeat, or stop?": the turn carries
"choices", and once the student picks (by voice, typing or a button) the turn carries "choice" and
the page opens the next lesson, restarts this one, or goes back to the lesson list.

Config (env vars):
  LELA_LLM_URL      vLLM OpenAI endpoint     (default http://127.0.0.1:8001/v1/chat/completions)
  LELA_LLM_MODEL    model name served by vLLM (default hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4)
  LELA_ASR_KN       "indicconformer" | "whisper"   (Kannada ASR backend, default indicconformer)
  LELA_ASR_EN       faster-whisper size for English questions (default "small"; reused if LELA_ASR_KN=whisper)
  LELA_TTS          "indictts" (default: AI4Bharat Indic-TTS Kannada on CPU: free, offline, clear pronunciation; see indictts.py)
                    | "mms" (Meta MMS-TTS Kannada: fast) | "parler" (AI4Bharat Indic Parler-TTS: nicer, slow)
                    | "indicf5" (voice clone of prompts/kannada_teacher.wav)
                    | "sarvam" (Sarvam Bulbul cloud API: most natural; falls back to MMS when offline)
  LELA_INDICTTS_DIR     folder with fastpitch/ and hifigan/ (default ./models/indictts/kn)
  LELA_INDICTTS_SPEAKER "female" (default) | "male"
  LELA_TTS_CACHE_DIR    where Sarvam audio is saved so each line is paid for once (default ./tts_cache)
  SARVAM_API_KEY    Sarvam API key (dashboard.sarvam.ai), needed for LELA_TTS=sarvam
  LELA_SARVAM_MODEL   default bulbul:v3 (bulbul:v4-flash is cheaper and faster)
  LELA_SARVAM_SPEAKER default shubh (bulbul:v3 voices include anand, kavitha, roopa, vijay, gokul, mani)
  LELA_SARVAM_PACE    default 0.85 (0.5 to 2.0; below 1 is slower, for a learner)
  LELA_TRANSLATE    "sarvam" (default when SARVAM_API_KEY is set): English questions are answered by the LLM in
                    English, then translated to Kannada by Sarvam for LeLa to speak | "llm": use the LLM's own Kannada
  LELA_EN_ANSWERS   "repeat" (default): an English answer is said back in Kannada and the student is asked to
                    say it in Kannada before moving on | "accept": said back in Kannada, then the lesson moves on
  LELA_SARVAM_TRANSLATE_MODEL  default sarvam-translate:v1 (mayura:v1 also works)
  LELA_TTS_RATE     MMS speaking rate (default 0.85: a little slower than normal, for a learner)
  LELA_TTS_PAD      seconds of silence added after each spoken line, half as much before (default 0.4)
  LELA_TTS_PAUSE    seconds AuTut pauses inside a turn, e.g. before "let's go back to where we left off" (default 0.9)
  LELA_TTS_VOICE    Parler voice description (default: Anu, slow and clear)
  LELA_LLM_JSON_SCHEMA  1 (default): ask vLLM for schema-forced JSON; 0 turns it off
  LELA_LLM_KANNADA  0 (default): LeLa speaks only hand-checked Kannada; 1: also speak the LLM's own Kannada
  LELA_CONFORMER_DEVICE  "auto" (GPU, falls back to CPU if onnxruntime's CUDA/cuDNN fails) | "cpu"
  LELA_EN_TTS       "tinyvoca" (default): at the start of a lesson AuTut repeats its opening in English with the
                    tiny-voca voice (CPU, see english_tts.py); skipped if not installed | "off"
  LELA_TINYVOCA_DIR folder with tiny-voca.pt and phonemizer/ (default ./models/tiny-voca)
  LELA_EN_VOICE     tiny-voca voice: "M2" (male, default) or "F2" (female)
  LELA_EN_SPEED     tiny-voca speed, 0.25 to 4 (default 0.8: slower than normal, for a learner)
  LELA_EN_PITCH     tiny-voca pitch shift (default -0.1, a little deeper; 0.058 is about one semitone, 0 = normal)
  LELA_EN_VOLUME    tiny-voca loudness, 0 to 1 (default 0.7: a little quieter than the Kannada voice)
  LELA_EN_GAP       seconds of silence between the Kannada and the English (default 1.0)
  LELA_LOG_PARTIAL  1: log each live-preview transcription with its size and time (to measure ASR speed)
  LELA_LESSONS_DIR  folder of lesson1.json, lesson2.json, ... (default ./lessons)
  LELA_PROGRESS_DIR folder for each learner's progress file (default ./progress)
  LELA_DISABLE_CUDNN   default 1 (cuDNN off for PyTorch, as in your original; IndicF5 fails with
                       CUDNN_STATUS_NOT_INITIALIZED otherwise). Set 0 once torch/cuDNN versions are aligned.
"""

import asyncio
import base64
import hashlib
import io
import json
import os
import re
import threading
import time
import traceback
from typing import Optional
from contextlib import asynccontextmanager

import httpx
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response

from progress import ProgressStore, lesson_summary, load_lessons, user_key
from translit import has_kannada, transliterate
from tutor import PAUSE, RESPONSE_SCHEMA, LessonSession, kannada_only, student_display

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_URL = os.getenv("LELA_LLM_URL", "http://127.0.0.1:8001/v1/chat/completions")
LLM_MODEL = os.getenv("LELA_LLM_MODEL", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
ASR_KN = os.getenv("LELA_ASR_KN", "indicconformer")
ASR_EN = os.getenv("LELA_ASR_EN", "small")
LOG_PARTIAL = os.getenv("LELA_LOG_PARTIAL", "0") == "1"
EN_TTS_BACKEND = os.getenv("LELA_EN_TTS", "tinyvoca")
EN_GAP_SECS = float(os.getenv("LELA_EN_GAP", "1.0"))
EN_VOICE = os.getenv("LELA_EN_VOICE", "M2")
EN_SPEED = float(os.getenv("LELA_EN_SPEED", "0.8"))
EN_PITCH = float(os.getenv("LELA_EN_PITCH", "-0.1"))
EN_VOLUME = float(os.getenv("LELA_EN_VOLUME", "0.7"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Only affects PyTorch (IndicF5). Whisper (CTranslate2) and IndicConformer (ONNX) don't use this flag.
if os.getenv("LELA_DISABLE_CUDNN", "1") == "1":
    torch.backends.cudnn.enabled = False

REF_AUDIO_PATH = os.path.join(BASE_DIR, "prompts", "kannada_teacher.wav")
# Must be the exact words spoken in kannada_teacher.wav, or IndicF5 output degrades.
REF_AUDIO_TEXT = os.getenv("LELA_REF_TEXT", "ನಮಸ್ಕಾರ, ಎಲ್ಲರಿಗೂ ಸುಸ್ವಾಗತ.")
TTS_BACKEND = os.getenv("LELA_TTS", "indictts")
TTS_VOICE = os.getenv(
    "LELA_TTS_VOICE",
    "Anu speaks slowly and clearly at a slightly slow pace, with a clear, close-sounding recording of very high quality.",
)
TTS_SAMPLE_RATE = 24000  # IndicF5 outputs 24 kHz; Parler / MMS overwrite this from their config at load time.
SARVAM_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_MODEL = os.getenv("LELA_SARVAM_MODEL", "bulbul:v3")
SARVAM_SPEAKER = os.getenv("LELA_SARVAM_SPEAKER", "shubh")
SARVAM_PACE = float(os.getenv("LELA_SARVAM_PACE", "0.85"))
SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"
SARVAM_TRANSLATE_MODEL = os.getenv("LELA_SARVAM_TRANSLATE_MODEL", "sarvam-translate:v1")
TRANSLATE_BACKEND = os.getenv("LELA_TRANSLATE", "sarvam" if SARVAM_KEY else "llm")
SARVAM_RATE = 24000  # requested from Sarvam; LeLa's output rate when LELA_TTS=sarvam
# Sarvam audio is saved here so each line is paid for once, not once per server run.
# Files are keyed by voice settings + text, so changing speaker/pace/model makes new ones.
INDICTTS_DIR = os.getenv("LELA_INDICTTS_DIR", os.path.join(BASE_DIR, "models", "indictts", "kn"))
INDICTTS_SPEAKER = os.getenv("LELA_INDICTTS_SPEAKER", "female")
SARVAM_CACHE_DIR = os.getenv("LELA_TTS_CACHE_DIR", os.path.join(BASE_DIR, "tts_cache"))

LESSONS = load_lessons(os.getenv("LELA_LESSONS_DIR", os.path.join(BASE_DIR, "lessons")))
LESSONS_BY_ID = {l["id"]: l for l in LESSONS}
PROGRESS = ProgressStore(os.getenv("LELA_PROGRESS_DIR", os.path.join(BASE_DIR, "progress")), LESSONS)

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
    load_english_tts()


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
        load_mms()
        TTS_SAMPLE_RATE = M["mms"].config.sampling_rate
    elif TTS_BACKEND == "sarvam":
        # Cloud voice; MMS stays loaded (~0.2 GB) so the tutor still speaks when the API is unreachable.
        load_mms()
        TTS_SAMPLE_RATE = SARVAM_RATE
        if not SARVAM_KEY:
            print("[Init] !!! LELA_TTS=sarvam but SARVAM_API_KEY is not set: every line will use MMS")
        else:
            print(f"[Init] Sarvam TTS: {SARVAM_MODEL}, speaker {SARVAM_SPEAKER}, pace {SARVAM_PACE}")
    elif TTS_BACKEND == "indictts":
        # Free, offline AI4Bharat Kannada voice on the CPU (no GPU memory). Setup steps: indictts.py
        from indictts import IndicTTS

        M["tts"] = IndicTTS(INDICTTS_DIR, INDICTTS_SPEAKER)
        TTS_SAMPLE_RATE = M["tts"].sample_rate
        print(f"[Init] Indic-TTS Kannada ({M['tts'].speaker} voice) on CPU from {INDICTTS_DIR}")
    else:
        from transformers import AutoModel

        M["tts"] = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True).to(DEVICE)
    SENTENCE_GAP = np.zeros(int(0.25 * TTS_SAMPLE_RATE), dtype=np.float32)
    print(f"[Init] TTS: {TTS_BACKEND} at {TTS_SAMPLE_RATE} Hz")


def load_mms() -> None:
    # Meta MMS (VITS). Small and fast, but CC-BY-NC 4.0: personal / non-commercial use only.
    from transformers import AutoTokenizer, VitsModel

    M["mms"] = VitsModel.from_pretrained("facebook/mms-tts-kan").to(DEVICE).eval()
    M["mms"].speaking_rate = float(os.getenv("LELA_TTS_RATE", "0.85"))  # <1 = slower
    M["mms_tok"] = AutoTokenizer.from_pretrained("facebook/mms-tts-kan")  # takes Kannada script directly


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
    wav = await asyncio.to_thread(synthesize, LESSONS[0]["steps"][0]["tutor_kn"])
    if wav:
        print(f"[Init] TTS self-test OK ({len(wav) // 2 / TTS_SAMPLE_RATE:.1f} s of audio)")
    else:
        print(f"[Init] !!! TTS self-test FAILED: {TTS_LAST_ERROR['error']}  (the tutor will be silent)")
    if M.get("en_tts"):  # same for the English voice, with the first lesson's opening line
        en_wav = await asyncio.to_thread(synthesize_en, LESSONS[0]["steps"][0]["tutor_en"])
        print(f"[Init] English voice self-test {'OK' if en_wav else 'FAILED (see [TTS:en] above)'}")
    if TRANSLATE_BACKEND == "sarvam" and SARVAM_KEY:
        print(f"[Init] English questions: answered in Kannada via Sarvam translate ({SARVAM_TRANSLATE_MODEL})")
    else:
        print("[Init] English questions: answered with the LLM's own Kannada (set SARVAM_API_KEY for better Kannada)")
    titles = ", ".join(f"{l['num']}. {l.get('title', l['id'])}" for l in LESSONS)
    print(f"[Init] Lessons: {titles}")
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
        try:
            for frame in container.decode(audio=0):
                chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
        except av.error.FFmpegError:
            # A live preview is a recording cut mid-stream: keep the frames decoded so far.
            if not chunks:
                raise
    chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]  # flush
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def transcribe(audio_bytes: bytes, lang: str, partial: bool = False) -> Optional[str]:
    """Any browser format -> text. lang is "kn" or "en".
    partial: a live preview of a recording still in progress. It never waits for the GPU: if
    another learner's ASR call is running it returns None at once, so previews can't slow grading."""
    try:
        audio = decode_audio(audio_bytes)
    except Exception as e:
        if not partial:
            print(f"[ASR] decode failed: {e}")
        return ""
    if len(audio) < 16000 * 0.3:  # under 0.3 s: a click, not speech
        return ""
    if not ASR_LOCK.acquire(blocking=not partial):
        return None
    try:
        if lang == "kn" and "conformer" in M:
            wav = torch.from_numpy(audio).unsqueeze(0)
            text = M["conformer"](wav, "kn", "rnnt")
            return (text[0] if isinstance(text, (list, tuple)) else str(text)).strip()
        segments, _ = M["whisper"].transcribe(
            audio, language=lang, beam_size=1 if partial else 5, vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(s.text for s in segments).strip()
    finally:
        ASR_LOCK.release()


def load_english_tts() -> None:
    """tiny-voca for AuTut's spoken English. Optional: without it the English is only shown."""
    if EN_TTS_BACKEND != "tinyvoca":
        print("[Init] English voice off (LELA_EN_TTS)")
        return
    try:
        from english_tts import EnglishTTS

        M["en_tts"] = EnglishTTS(os.getenv("LELA_TINYVOCA_DIR", os.path.join(BASE_DIR, "models", "tiny-voca")),
                                 EN_VOICE, EN_SPEED, EN_PITCH, EN_VOLUME)
        print(f"[Init] English voice: tiny-voca {M['en_tts'].voice}, speed {EN_SPEED}, "
              f"pitch {EN_PITCH:+}, volume {EN_VOLUME} on CPU")
    except Exception as e:
        print(f"[Init] English voice unavailable, English is shown but not spoken: {e}")


def synthesize_en(text: str) -> bytes:
    """English -> WAV bytes with tiny-voca, or b"" if it isn't loaded or fails."""
    tts = M.get("en_tts")
    if not tts:
        return b""
    from english_tts import speakable
    t0 = time.perf_counter()
    try:
        audio = tts(text)
    except Exception as e:
        traceback.print_exc()
        print(f"[TTS:en] failed: {e!r}  line='{speakable(text)[:80]}'")
        return b""
    if not len(audio):
        print(f"[TTS:en] nothing could be read aloud from '{speakable(text)[:80]}'")
        return b""
    print(f"[TTS:en] {len(audio) / tts.sample_rate:.1f} s of English in {time.perf_counter() - t0:.1f} s")
    buf = io.BytesIO()
    sf.write(buf, audio, tts.sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


SENTENCE_GAP = np.zeros(int(0.25 * TTS_SAMPLE_RATE), dtype=np.float32)  # rebuilt in load_tts()
SENT_CACHE: dict[str, np.ndarray] = {}
TAIL_PAD_SECS = float(os.getenv("LELA_TTS_PAD", "0.4"))  # silence after each line (half as much before)


def split_sentences(text: str) -> list[str]:
    """'ತುಂಬಾ ಒಳ್ಳೆಯದು! ನೀವು ಹೇಗಿದ್ದೀರಿ?' -> ['ತುಂಬಾ ಒಳ್ಳೆಯದು!', 'ನೀವು ಹೇಗಿದ್ದೀರಿ?']"""
    return [p.strip() for p in re.findall(r"[^.!?।]+[.!?।]*", text) if p.strip(" .!?।")]


def end_sentence(text: str) -> str:
    """'ಹೌದು, ಇದು ನನ್ನ ಪುಸ್ತಕ' -> 'ಹೌದು, ಇದು ನನ್ನ ಪುಸ್ತಕ.' Practice phrases reach the TTS with their
    full stop stripped, and without one the models treat the line as unfinished and swallow the
    last vowel (ಪುಸ್ತಕ comes out as 'pustak'). A trailing comma becomes a full stop too."""
    text = text.rstrip(" ,")
    return text if not text or text[-1] in ".!?।" else text + "."


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
        return _run_mms(sentence)
    if TTS_BACKEND == "indictts":
        return M["tts"](sentence)
    if TTS_BACKEND == "sarvam":
        return _run_sarvam(sentence)
    return M["tts"](sentence, ref_audio_path=REF_AUDIO_PATH, ref_text=REF_AUDIO_TEXT)


def _run_mms(sentence: str):
    tok = M["mms_tok"]
    text = sentence
    if getattr(tok, "is_uroman", False):  # some MMS languages expect romanized input
        import uroman  # pip install uroman

        text = uroman.Uroman().romanize_string(sentence)
    inputs = tok(text, return_tensors="pt").to(DEVICE)
    torch.manual_seed(0)  # VITS samples durations; a fixed seed makes each line sound the same every time
    return M["mms"](**inputs).waveform


def _run_sarvam(sentence: str) -> np.ndarray:
    """Sarvam Bulbul REST call -> float32 mono at SARVAM_RATE. Raises on any failure (caller falls back)."""
    if not SARVAM_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set")
    r = httpx.post(
        SARVAM_URL,
        headers={"api-subscription-key": SARVAM_KEY},
        json={
            "text": sentence,
            "language_code": "kn-IN",
            "model": SARVAM_MODEL,
            "speaker": SARVAM_SPEAKER,
            "pace": SARVAM_PACE,
            "speech_sample_rate": SARVAM_RATE,
            "output_audio_codec": "wav",
        },
        timeout=20,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Sarvam HTTP {r.status_code}: {r.text[:300]}")
    audio, rate = sf.read(io.BytesIO(base64.b64decode(r.json()["audios"][0])), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return _resample(audio, rate, SARVAM_RATE)


def _sarvam_cache_path(sentence: str) -> str:
    key = f"{SARVAM_MODEL}|{SARVAM_SPEAKER}|{SARVAM_PACE}|{SARVAM_RATE}|{sentence}"
    return os.path.join(SARVAM_CACHE_DIR, hashlib.sha256(key.encode("utf-8")).hexdigest()[:24] + ".wav")


def _sarvam_cached(sentence: str) -> np.ndarray:
    """Sarvam audio from the disk cache, or from the API (then saved). A bad cache file is ignored."""
    path = _sarvam_cache_path(sentence)
    if os.path.exists(path):
        try:
            audio, _ = sf.read(path, dtype="float32")
            return audio
        except Exception as e:
            print(f"[TTS] unreadable cache file {path} ({e}); asking Sarvam again")
    audio = _run_sarvam(sentence)
    try:
        os.makedirs(SARVAM_CACHE_DIR, exist_ok=True)
        tmp = path + ".tmp"
        sf.write(tmp, audio, SARVAM_RATE, format="WAV", subtype="FLOAT")
        os.replace(tmp, path)  # never leave a half-written file behind
    except Exception as e:
        print(f"[TTS] could not save Sarvam audio to {SARVAM_CACHE_DIR}: {e}")
    return audio


def _resample(audio: np.ndarray, orig: int, new: int) -> np.ndarray:
    if orig == new:
        return audio
    import torchaudio

    t = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    return torchaudio.functional.resample(t, orig, new).numpy()


def _run_local(run, sentence: str):
    with torch.no_grad():
        try:
            return run(sentence)
        except RuntimeError as e:
            if "CUDNN" not in str(e).upper() or not torch.backends.cudnn.enabled:
                raise
            print("[TTS] cuDNN error; disabling cuDNN for PyTorch and retrying")
            torch.backends.cudnn.enabled = False
            return run(sentence)


def _tts_one(sentence: str) -> tuple[np.ndarray, bool]:
    """One sentence -> (float32 audio in [-1, 1], cacheable). Caller holds TTS_LOCK.
    With LELA_TTS=sarvam, a failed API call falls back to MMS; that audio isn't cached,
    so the next time the line is spoken Sarvam is tried again."""
    if sentence in SENT_CACHE:
        return SENT_CACHE[sentence], True
    cacheable = True
    if TTS_BACKEND == "sarvam":
        try:
            audio = _sarvam_cached(sentence)
        except Exception as e:
            print(f"[TTS] Sarvam failed ({type(e).__name__}: {e}); using MMS for this line")
            audio = _run_local(_run_mms, sentence)
            audio = audio.cpu().numpy().squeeze().astype(np.float32)
            audio = _resample(audio, M["mms"].config.sampling_rate, SARVAM_RATE)
            cacheable = False
    else:
        audio = _run_local(_run_tts, sentence)
    if isinstance(audio, tuple):
        audio = audio[0]
    if hasattr(audio, "cpu"):
        audio = audio.cpu().numpy()
    audio = np.asarray(audio).squeeze()
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"{TTS_BACKEND} returned empty/invalid audio (shape {audio.shape})")
    peak = float(np.abs(audio).max())
    if peak > 1.0:  # float output outside [-1, 1] would clip to noise in the WAV
        audio = audio / peak
    audio = audio.astype(np.float32)
    if cacheable and len(SENT_CACHE) < 512:
        SENT_CACHE[sentence] = audio
    return audio, cacheable


def synthesize(text: str) -> bytes:
    """Kannada text -> WAV bytes. Sarvam, MMS and Indic-TTS get the whole line in one call: cut into
    one-word pieces like 'ನಮಸ್ಕಾರ!', they clip the last syllable and lose the sentence's
    intonation. IndicF5 / Parler still get one sentence at a time (joined with a short pause),
    because given several sentences at once F5-style TTS can drop or reorder them."""
    text = kannada_only(text)
    if not text:
        return b""
    text = end_sentence(text)
    if text in TTS_CACHE:
        return TTS_CACHE[text]
    try:
        parts = []
        cacheable = True
        with TTS_LOCK:
            pieces = [text] if TTS_BACKEND in ("sarvam", "mms", "indictts") else split_sentences(text)
            for sentence in pieces:
                if parts:
                    parts.append(SENTENCE_GAP)
                audio, ok = _tts_one(sentence)
                parts.append(audio)
                cacheable = cacheable and ok
        # Quiet padding at both ends: some speakers / Bluetooth outputs cut the first and last
        # ~0.2 s of a clip, which swallows a final syllable like the ಕಿ of ಶಿಕ್ಷಕಿ.
        pad = np.zeros(int(TAIL_PAD_SECS * TTS_SAMPLE_RATE), dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, np.concatenate([pad[: len(pad) // 2], *parts, pad]), TTS_SAMPLE_RATE, format="WAV")
        wav = buf.getvalue()
    except Exception as e:
        TTS_LAST_ERROR["error"] = f"{type(e).__name__}: {e}"
        print(f"[TTS] synthesis failed for '{text}':")
        traceback.print_exc()
        return b""
    TTS_LAST_ERROR["error"] = None
    if cacheable and len(TTS_CACHE) < 256:
        TTS_CACHE[text] = wav
    return wav


LLM_JSON_SCHEMA = {"on": os.getenv("LELA_LLM_JSON_SCHEMA", "1") == "1"}


async def call_llm(messages: list[dict]) -> str:
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 400,
    }
    if LLM_JSON_SCHEMA["on"]:
        # vLLM structured output: the reply is forced to match the schema.
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "lela_turn", "schema": RESPONSE_SCHEMA},
        }
    resp = await M["http"].post(LLM_URL, json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def translate_en_kn(text: str) -> str:
    """Sarvam translate: English -> Kannada. Raises on failure (tutor.py falls back to the LLM's Kannada)."""
    resp = await M["http"].post(
        SARVAM_TRANSLATE_URL,
        headers={"api-subscription-key": SARVAM_KEY},
        json={
            "input": text,
            "source_language_code": "en-IN",
            "target_language_code": "kn-IN",
            "model": SARVAM_TRANSLATE_MODEL,
            "mode": "formal",
        },
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Sarvam translate HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json()["translated_text"]


async def call_llm_text(messages: list[dict]) -> str:
    """Plain-text chat (no JSON schema): used to answer the student's side questions."""
    payload = {"model": LLM_MODEL, "messages": messages, "temperature": 0.3, "max_tokens": 120}
    resp = await M["http"].post(LLM_URL, json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------- transport

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.get("/api/lessons")
async def api_lessons(user: str = ""):
    """Lesson list for the picker; with ?user=<name>, that learner's progress card on each lesson."""
    out = {"lessons": [lesson_summary(l) for l in LESSONS]}
    if user_key(user):
        out["progress"] = PROGRESS.cards(user)
        out["suggested"] = PROGRESS.next_lesson(user)
    return out


@app.post("/api/reset")
async def api_reset(user: str = "", lesson: str = ""):
    """Wipe one lesson's progress for this learner (step 0, no scores). The page asks to confirm first."""
    if not user_key(user) or lesson not in LESSONS_BY_ID:
        return JSONResponse({"ok": False, "error": "unknown learner or lesson"}, 400)
    return {"ok": True, "card": await asyncio.to_thread(PROGRESS.reset, user, lesson)}


@app.get("/tts_test")
async def tts_test(text: str = "ನಮಸ್ಕಾರ! ನೀವು ಹೇಗಿದ್ದೀರಿ?"):
    """Open http://localhost:7800/tts_test in the browser: plays the WAV, or shows the TTS error."""
    wav = await asyncio.to_thread(synthesize, text)
    if not wav:
        return JSONResponse({"ok": False, "error": TTS_LAST_ERROR["error"], "tts": TTS_BACKEND}, 500)
    return Response(wav, media_type="audio/wav")


@app.get("/tts_test_en")
async def tts_test_en(text: str = "Welcome to LeLa. Namaskara means hello."):
    """Open http://localhost:7800/tts_test_en in the browser: plays tiny-voca's English, or says why not."""
    if not M.get("en_tts"):
        return JSONResponse({"ok": False, "error": "English voice not loaded (see the [Init] lines in the log)"}, 500)
    wav = await asyncio.to_thread(synthesize_en, text)
    if not wav:
        return JSONResponse({"ok": False, "error": "tiny-voca produced no audio (see [TTS:en] in the log)"}, 500)
    return Response(wav, media_type="audio/wav")


PAUSE_SECS = float(os.getenv("LELA_TTS_PAUSE", "0.9"))  # silence where AuTut pauses inside a turn (tutor.PAUSE)


def silence_wav(secs: float) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(secs * TTS_SAMPLE_RATE), dtype=np.float32), TTS_SAMPLE_RATE, format="WAV")
    return buf.getvalue()


async def send_turn(ws: WebSocket, turn) -> None:
    # Text first so the student can read while audio is synthesized.
    await ws.send_text(json.dumps(turn.to_json(), ensure_ascii=False))
    # A turn can hold pauses (tutor.PAUSE, a line break): "answer / pause / now you say it", or
    # "let's go back to where we left off / pause / the question". Each piece is its own clip;
    # the page plays the clips in order, with a clip of silence between them.
    state = {"ok": True, "sent": False}

    async def say_kn(text: str, gap: float) -> None:
        first = True
        for piece in (p for p in text.split(PAUSE) if kannada_only(p)):
            audio = await asyncio.to_thread(synthesize, piece)
            state["ok"] = state["ok"] and bool(audio)
            if audio:
                if state["sent"]:
                    await ws.send_bytes(silence_wav(gap if first else PAUSE_SECS))
                await ws.send_bytes(audio)
                state["sent"], first = True, False

    async def say_en(text: str, gap: float) -> None:
        # Kannada words inside the English are said by the Kannada voice (tiny-voca can't say them).
        if not (text and M.get("en_tts")):
            return
        from english_tts import segments

        first = True
        for lang, piece in segments(text, getattr(turn, "speak_en_kn", None)):
            audio = await asyncio.to_thread(synthesize_en if lang == "en" else synthesize, piece)
            if not audio:
                continue
            if first and state["sent"]:
                await ws.send_bytes(silence_wav(gap))
            await ws.send_bytes(audio)
            state["sent"], first = True, False

    speech = getattr(turn, "speech", None)
    if speech and not M.get("en_tts"):
        speech = [(lang, text) for lang, text in speech if lang == "kn"] or [("kn", turn.kn)]
    if speech:
        # In order, with a pause between parts: e.g. lesson start = English, pause, Kannada;
        # an English question = the answer in English, pause, back to the lesson in Kannada.
        for lang, text in speech:
            await (say_en if lang == "en" else say_kn)(text, EN_GAP_SECS)
    else:
        await say_kn(turn.kn, PAUSE_SECS)
        if getattr(turn, "speak_en", None):
            await say_en(turn.speak_en, EN_GAP_SECS)
    ok = state["ok"]
    # turn_end comes last, so the mic stays locked until everything has played.
    await ws.send_text(json.dumps({"type": "turn_end", "tts_error": None if ok else TTS_LAST_ERROR["error"]}))


async def send_progress(ws: WebSocket, lesson_id: str, card: dict) -> None:
    await ws.send_text(json.dumps({"type": "progress", "lesson": lesson_id, "card": card}, ensure_ascii=False))


@app.websocket("/ws/tutor")
async def websocket_tutor(ws: WebSocket):
    # /ws/tutor?user=<name>&lesson=<lesson id>&restart=1
    await ws.accept()  # accept first so a refusal reaches the page with its reason
    user = ws.query_params.get("user", "").strip()[:60]
    if not user_key(user):
        await ws.close(code=4400, reason="Enter your name first")
        return
    lesson_id = ws.query_params.get("lesson") or PROGRESS.next_lesson(user)
    if lesson_id not in LESSONS_BY_ID:
        await ws.close(code=4404, reason=f"No lesson {lesson_id}")
        return
    lesson = LESSONS_BY_ID[lesson_id]
    # A new learner (nothing started yet) first hears AuTut introduce itself.
    new_learner = all(c.get("status") == "not_started" for c in PROGRESS.cards(user).values())
    card = PROGRESS.start(user, lesson_id, restart=ws.query_params.get("restart") == "1")
    print(f"[WS] {user}: lesson {lesson['num']} ({lesson_id}) from step {card['step']}")
    session = LessonSession(
        lesson, call_llm, step_idx=card["step"], results=dict(card["results"]),
        has_next=LESSONS.index(lesson) + 1 < len(LESSONS),
        llm_kannada=os.getenv("LELA_LLM_KANNADA", "0") == "1",
        chat=call_llm_text,
        english_answers=os.getenv("LELA_EN_ANSWERS", "repeat"),
        translate=translate_en_kn if (TRANSLATE_BACKEND == "sarvam" and SARVAM_KEY) else None,
    )
    next_audio_lang = "kn"
    next_is_partial = False   # the next audio clip is a live preview, not the student's answer
    try:
        await send_progress(ws, lesson_id, card)
        # Tutor opens the class with a greeting (a new learner: AuTut introduces itself first).
        groups = len({re.match(r"\d+", str(l.get("num", ""))).group() for l in LESSONS
                      if re.match(r"\d+", str(l.get("num", "")))})
        await send_turn(ws, session.opening(welcome_lessons=groups if new_learner else 0))
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            if msg.get("text") is not None:
                data = json.loads(msg["text"])
                if data.get("type") == "audio_lang":      # next audio clip is "kn" or "en"
                    next_audio_lang = "en" if data.get("lang") == "en" else "kn"
                    continue
                if data.get("type") == "partial_audio":   # next clip: the recording so far, for a preview
                    next_is_partial = True
                    next_audio_lang = "en" if data.get("lang") == "en" else "kn"
                    continue
                if data.get("type") != "text" or not data.get("text", "").strip():
                    continue
                student_text = data["text"].strip()
                # Typed input: Kannada script if it contains any, else treat as English.
                lang = "kn" if has_kannada(student_text) else "en"
            elif next_is_partial:
                # Live preview while the student is still speaking: show it, nothing else.
                next_is_partial = False
                t0 = time.perf_counter()
                text = await asyncio.to_thread(transcribe, msg["bytes"], next_audio_lang, True)
                if LOG_PARTIAL:
                    print(f"[ASR:partial] {len(msg['bytes'])} B in {time.perf_counter() - t0:.2f}s: '{text}'")
                await ws.send_text(json.dumps({"type": "partial", "skipped": text is None,
                                               "student_display": student_display(text, next_audio_lang) if text else ""},
                                              ensure_ascii=False))
                continue
            else:
                lang, next_audio_lang = next_audio_lang, "kn"
                t0 = time.perf_counter()
                student_text = await asyncio.to_thread(transcribe, msg["bytes"], lang)
                print(f"[ASR] {len(msg['bytes'])} B transcribed in {time.perf_counter() - t0:.2f}s")
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
            card = await asyncio.to_thread(PROGRESS.update, user, lesson_id, session)
            await send_progress(ws, lesson_id, card)
            await send_turn(ws, turn)
    except WebSocketDisconnect:
        pass
    finally:
        print("[WS] session closed")

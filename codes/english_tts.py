"""English voice for AuTut: tiny-voca (https://huggingface.co/n4ze3m/tiny-voca), run on the CPU.

Used for the English line AuTut says after the Kannada one at the start of a lesson.
Small (4.5M parameters, ~18 MB checkpoint), about 70x faster than real time on 4 CPU threads,
no GPU memory. Apache-2.0. English only, two voices: "M2" (male, LeLa's default) and "F2" (female), 24 kHz.
Kannada words inside the English are said by the Kannada voice instead (see segments()).

Setup (once, in the lela folder with the venv active). --no-deps keeps pip from touching the
torch / numpy versions vLLM needs (tiny-voca works with torch>=2.4 and numpy>=1.24):
    pip install --no-deps git+https://github.com/n4ze3m/dizione git+https://github.com/n4ze3m/tiny-voca
    hf download n4ze3m/tiny-voca --local-dir models/tiny-voca
        (or: huggingface-cli download n4ze3m/tiny-voca --local-dir models/tiny-voca)
    -> models/tiny-voca/tiny-voca.pt and models/tiny-voca/phonemizer/

If it isn't installed, AuTut just skips the spoken English (the English stays on screen).
"""

from __future__ import annotations

import os
import re
import unicodedata

import numpy as np

_KANNADA = re.compile(r"[ಀ-೿]+")
_EMPTY_PARENS = re.compile(r"\(\s*\)")
_PARENS = re.compile(r"\(\s*([^()]*?)\s*\)")


def speakable(text: str) -> str:
    """AuTut's on-screen English -> a line the English voice can read: no Kannada script,
    no diacritics ("Namaskāra" -> "Namaskara"), "x = y" read as "x, meaning y", one line."""
    text = _KANNADA.sub("", text or "")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _EMPTY_PARENS.sub("", text)
    text = _PARENS.sub(r"\1", text)                 # "Say (sari) when ready" -> "Say sari when ready"
    text = text.replace(" = ", ", meaning ").replace("=", ", meaning ")
    text = re.sub(r"\s*\n+\s*", ". ", text)
    text = re.sub(r"\.\s*\.", ".", text)
    text = re.sub(r"([?!])\s*[.;]", r"\1", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _find_checkpoint(model_dir: str) -> tuple[str, list[str]]:
    """The tiny-voca .pt file, or "" plus every folder searched. Looks in model_dir (and any
    sub-folder), ./models/tiny-voca and ./tiny-voca from where the app was started, and the
    Hugging Face cache (where `hf download` without --local-dir puts it)."""
    here = os.path.dirname(os.path.abspath(__file__))
    dirs = [model_dir, os.path.join(here, "models", "tiny-voca"), os.path.join(os.getcwd(), "models", "tiny-voca"),
            os.path.join(here, "tiny-voca"), os.path.join(os.getcwd(), "tiny-voca")]
    try:
        from huggingface_hub import snapshot_download
        dirs.append(snapshot_download("n4ze3m/tiny-voca", local_files_only=True))
    except Exception:
        pass
    seen = []
    for d in dirs:
        d = os.path.abspath(d)
        if d in seen:
            continue
        seen.append(d)
        found = sorted(os.path.join(root, f) for root, _, files in os.walk(d) for f in files if f.endswith(".pt"))
        if found:  # prefer tiny-voca.pt, else the first .pt there
            named = [f for f in found if os.path.basename(f) == "tiny-voca.pt"]
            return (named or found)[0], seen
    return "", seen


def _find_phonemizer(start: str) -> str:
    """The phonemizer/ data folder next to (or above) the checkpoint."""
    d = os.path.dirname(start)
    for _ in range(3):
        if os.path.isdir(os.path.join(d, "phonemizer")):
            return os.path.join(d, "phonemizer")
        d = os.path.dirname(d)
    return ""


_KN_RUN = re.compile(r"[\u0C80-\u0CFF][\u0C80-\u0CFF\s.,!?]*(?:\s*\([^()]*\))?")


def segments(text: str, phrases: dict | None = None) -> list[tuple[str, str]]:
    """Split AuTut's English into [("en", words) | ("kn", Kannada script)], so Kannada words are
    spoken by the Kannada voice and only real English goes to tiny-voca (which mangles Kannada).
    Kannada script in the text ("Say ಸರಿ (sari)") becomes a "kn" piece and its "(roman)" is dropped;
    a romanized phrase from `phrases` ({"Namaskāra": "ನಮಸ್ಕಾರ"}) is swapped for its Kannada."""
    text = re.sub(r"\s*\n+\s*", ". ", text or "")
    spans = []   # (start, end, Kannada to say, English to say right after it)
    for m in _KN_RUN.finditer(text):
        kn, _, paren = m.group(0).partition("(")
        gloss = paren.rstrip(") ").split(",", 1)[1].strip() if "," in paren else ""   # "(nanna, my)" -> "my"
        spans.append((m.start(), m.end(), kn.strip(" .,!?"), gloss))
    for roman, kn in sorted((phrases or {}).items(), key=lambda kv: -len(kv[0])):
        if not roman.strip():
            continue
        for m in re.finditer(r"(?<![\w])" + re.escape(roman.strip(" .,!?")) + r"(?![\w])", text, re.IGNORECASE):
            if not any(a < m.end() and m.start() < b for a, b, _, _ in spans):
                spans.append((m.start(), m.end(), kn, ""))
    out, at = [], 0
    for a, b, kn, gloss in sorted(spans):
        if a > at:
            out.append(("en", text[at:a]))
        out.append(("kn", kn))
        if gloss:
            out.append(("en", gloss + ","))
        at = b
    out.append(("en", text[at:]))
    from translit import transliterate

    merged = []
    for lang, piece in out:
        if lang == "en":
            piece = piece.replace("(", " ").replace(")", " ")
            if merged and merged[-1][0] == "kn":   # "ಇವರು ivaru": the Kannada voice already said it
                roman = speakable(transliterate(merged[-1][1])).lower().strip(" .,!?")
                head = speakable(piece)
                if roman and head.lower().startswith(roman):
                    piece = head[len(roman):]
        piece = speakable(piece) if lang == "en" else piece.strip()
        if lang == "en" and not re.search(r"[A-Za-z0-9]", piece):
            continue
        if lang == "kn" and not piece:
            continue
        merged.append((lang, piece))
    return merged


class EnglishTTS:
    def __init__(self, model_dir: str, voice: str = "M2", speed: float = 0.8,
                 pitch_shift: float = -0.1, volume: float = 0.7):
        """pitch_shift: added to log pitch (0.058 is about one semitone; -0.1 is a little deeper).
        volume: loudness relative to tiny-voca's normal level (0.7 = quieter than the Kannada voice)."""
        ckpt, searched = _find_checkpoint(model_dir)
        if not ckpt:
            raise FileNotFoundError("no tiny-voca .pt checkpoint in " + ", ".join(searched)
                                    + " (set LELA_TINYVOCA_DIR to the folder hf download wrote; see english_tts.py)")
        phon = _find_phonemizer(ckpt)
        if phon:
            os.environ.setdefault("DIZIONE_DATA", phon)  # must be set before dizione loads
        print(f"[Init] tiny-voca checkpoint {ckpt}" + (f", phonemizer {phon}" if phon else ", no phonemizer/ folder found"))
        from tiny_voca.infer import load_model, match_frontend
        from tiny_voca.text import FRONTEND_VARIABLE

        if FRONTEND_VARIABLE not in os.environ:
            match_frontend(ckpt)                        # the phonemizer the checkpoint was trained with
        self.model = load_model(ckpt, "cpu")
        self.voice = voice if voice in self.model.config.speakers else self.model.config.speakers[0]
        self.speed, self.pitch_shift, self.volume = speed, pitch_shift, volume
        self.sample_rate = self.model.config.sample_rate

    def __call__(self, text: str) -> np.ndarray:
        """English text -> float32 mono at self.sample_rate (empty if nothing could be read)."""
        import torch
        from tiny_voca.infer import speak_text
        from tiny_voca.text import NothingToPronounce

        line = speakable(text)
        if not line:
            return np.zeros(0, dtype=np.float32)
        try:
            with torch.inference_mode():
                wave, _ = speak_text(self.model, line, self.voice, self.speed, self.pitch_shift)
        except NothingToPronounce:
            return np.zeros(0, dtype=np.float32)
        return (wave.numpy() * self.volume).astype(np.float32)

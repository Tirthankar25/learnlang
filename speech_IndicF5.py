import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel

# 1. Load IndicF5
repo_id = "ai4bharat/IndicF5"
tts_model = AutoModel.from_pretrained(repo_id, trust_remote_code=True).to(
    "cuda"
)

# 2. Reference audio prompt (a 3-5 second clear clip of your ideal Kannada teacher)
REF_AUDIO = "prompts/kannada_teacher_warm_f.wav"
REF_TEXT = "ನಮಸ್ಕಾರ, ಎಲ್ಲರಿಗೂ ಸುಸ್ವಾಗತ."  # Ground truth transcript of the reference clip


def synthesize_kannada(text_to_speak: str) -> bytes:
  # Generate audio using flow-matching
  audio = tts_model(
      text_to_speak, ref_audio_path=REF_AUDIO, ref_text=REF_TEXT
  )

  if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0

  # Serialize to 24kHz WAV bytes for the WebSocket stream
  import io

  buffer = io.BytesIO()
  sf.write(buffer, np.array(audio, dtype=np.float32), 24000, format="WAV")
  return buffer.getvalue()
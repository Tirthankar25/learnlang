"""AI4Bharat Indic-TTS Kannada voice (FastPitch + HiFi-GAN), run on the CPU with Coqui TTS.

Free and offline. Trained on native speakers reading Kannada script (IndicTTS database, IIT Madras),
so conjuncts like ಕ್ಷ and word endings come out right. Two voices: "female" and "male".

Setup (once, in the lela folder with the venv active):
    printf 'numpy==1.26.4\ntransformers==4.57.6\ntorch==2.4.0\n' > /tmp/lela-pins.txt
    pip install coqui-tts six -c /tmp/lela-pins.txt
    mkdir -p models/indictts && cd models/indictts
    wget https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/kn.zip
    unzip kn.zip && rm kn.zip && cd ../..     # -> models/indictts/kn/{fastpitch,hifigan}/  (~1.6 GB)

The pins stop pip from upgrading numpy / transformers / torch, which vLLM 0.6.2 needs as they are
(a plain `pip install coqui-tts` pulls numpy 2 and transformers 5). coqui-tts forgets to list `six`.
License: the checkpoints are CC-BY 4.0 (credit AI4Bharat); free for any use with attribution.
"""

import json
import os

import numpy as np


class IndicTTS:
    def __init__(self, model_dir: str, speaker: str = "female", use_cuda: bool = False):
        from TTS.utils.synthesizer import Synthesizer  # pip install coqui-tts

        fp, hg = os.path.join(model_dir, "fastpitch"), os.path.join(model_dir, "hifigan")
        for f in ("best_model.pth", "config.json", "speakers.pth"):
            if not os.path.exists(os.path.join(fp, f)):
                raise FileNotFoundError(f"Indic-TTS Kannada model not found: {os.path.join(fp, f)} (see indictts.py for setup)")
        speakers = os.path.join(fp, "speakers.pth")

        # The released config points at the authors' own path for speakers.pth; write a copy that
        # points at this one (next to the original, so nothing outside the model folder changes).
        with open(os.path.join(fp, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["speakers_file"] = speakers
        cfg.setdefault("model_args", {})["speakers_file"] = speakers
        cfg_path = os.path.join(fp, "config.lela.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=1)

        self.synth = Synthesizer(
            tts_checkpoint=os.path.join(fp, "best_model.pth"),
            tts_config_path=cfg_path,
            tts_speakers_file=speakers,
            vocoder_checkpoint=os.path.join(hg, "best_model.pth"),
            vocoder_config=os.path.join(hg, "config.json"),
            use_cuda=use_cuda,
        )
        self.sample_rate = int(self.synth.output_sample_rate)
        names = list(getattr(self.synth.tts_model.speaker_manager, "name_to_id", {}) or [])
        if names and speaker not in names:
            print(f"[Init] Indic-TTS has no speaker '{speaker}' (has {names}); using {names[0]}")
            speaker = names[0]
        self.speaker = speaker

    def __call__(self, text: str) -> np.ndarray:
        wav = self.synth.tts(text, speaker_name=self.speaker, split_sentences=False)
        return np.asarray(wav, dtype=np.float32)

# LeLa code review

Reviewed: `app.py`, `index.html`, `run_all.sh`, `start_vllm.sh` (the `prompts/kannada_teacher.wav` clip was not attached).
Design target: an agentic 1-on-1 Kannada tutor that leads the class, opens with a greeting, coaches the student to greet back, lets the student ask for clarification, and uses English as support.

## Verdict

The plumbing is a good start: browser mic → WebSocket → ASR → LLM → TTS → browser playback works end to end, the greeting is spoken first, and IndicF5 is wired up the way its model card shows. But **the tutor can't really teach yet**. The LLM sees one student sentence at a time, with no memory of what it asked and no lesson plan, so it can't lead, coach a missed greeting, or know when to move on. Clarification questions in English are also forced through Kannada ASR, so they come out garbled.

Corrected drafts sit next to this file (details at the end). None of them have been run on a GPU; the lesson logic was tested with a fake LLM (`test_tutor.py`, all cases pass).

---

## 1. Must fix

### 1.1 Live Hugging Face token in `start_vllm.sh` (security)
Line 4 hardcodes `HF_TOKEN="hf_btVx..."`. Anyone who has seen the file can use your HF account. **Revoke it** at huggingface.co → Settings → Access Tokens, and check git history if the file was ever committed or pushed. Use `hf auth login` once (it caches the token in `~/.cache/huggingface`) or export `HF_TOKEN` from an untracked `.env`. The comment above it ("uncomment") also doesn't match: the line is active.

### 1.2 The LLM has no conversation history and no lesson state (design)
Each request is `system prompt + "Student said: '...'"`. Consequences:
- The model doesn't know what LeLa just asked, so it can't judge whether "ನಮಸ್ಕಾರ" was the right reply or coach "say ನಮಸ್ಕಾರ back" when the student says something else.
- `next_state` is returned but never fed back, so the "state machine" is decorative; every turn starts from zero.
- There is no curriculum, so after the greeting the 8B model improvises a different lesson each time.
- No attempt counting, so a student who can't get a phrase can loop forever.

**Fix (in the drafts):** the server owns the lesson. `lessons.json` holds the steps (what LeLa asks, the goal, accepted answers, a coaching hint in Kannada/romanized/English). `tutor.py`'s `LessonSession` keeps the current step, attempt count and the last 12 lines of dialogue and sends all of that to the LLM. The LLM is the agent inside a step: it picks `advance`, `retry` (coach the student) or `explain` (answer a clarification, then re-ask). Code guardrails sit around it: a fuzzy match against the expected phrase forces `advance` even if the LLM is too strict; after `max_attempts` LeLa models the answer and moves on; if the LLM is down or returns junk, a deterministic hint is used.

This hybrid is deliberate. A small open model leading a whole lesson freely drifts; a fixed plan with an LLM judging and phrasing each turn stays on track and still feels conversational. You can make it more agentic later (let the LLM pick the next step or insert a review step) once the basic loop is solid.

### 1.3 Clarification in English can't work (design)
Whisper is forced to `language="kn"`, so "what does hegiddiri mean?" gets transliterated or hallucinated into Kannada script. The design explicitly lets the student ask for clarity, and that will mostly happen in English.
**Fix:** the draft UI has two hold-to-talk buttons ("answer in Kannada" and "ask in English") plus a text box. The client tells the server which language the next clip is in, and English goes to Whisper with `language="en"`. Automatic language detection per clip is possible but is unreliable on 1 to 2 second clips, so an explicit button is more robust for now.

### 1.4 Blocking ASR/TTS inside the async handler (performance, correctness)
`transcribe_audio_chunk` and `synthesize_speech` are synchronous GPU calls made directly in `async def websocket_tutor`. Each one freezes the whole event loop for seconds: other sessions, the `/` page and even WebSocket pings stall. Two sessions would also call the same GPU models concurrently.
**Fix:** `await asyncio.to_thread(...)` plus one `threading.Lock` per model (draft `app.py`). Repeated lesson prompts are cached so the opening greeting isn't re-synthesized for every connection.

### 1.5 `run_all.sh` never stops vLLM
`trap "kill $VLLM_PID" EXIT` comes after the blocking `uvicorn` line, so it is only registered once uvicorn has already exited, and on Ctrl-C the script dies before reaching it. vLLM is left holding ~55% of GPU memory, and the next start fails. Also: if vLLM crashes during startup, the health loop spins forever; relative paths break if run from another directory.
**Fix:** draft registers the trap first, starts vLLM in its own process group (`setsid`) and kills the group, checks the PID is still alive while waiting, times out after ~10 minutes and prints the tail of `vllm.log`, and `cd`s to the script directory.

### 1.6 Microphone won't work when you open the server from another machine
`getUserMedia` only works in a secure context: `https://` or `http://localhost`. If you browse to `http://<ubuntu-ip>:7800`, the mic request fails. Either use an SSH tunnel (`ssh -L 7800:localhost:7800 user@server`, then open `http://localhost:7800`) or put Caddy/nginx with TLS in front. The draft binds uvicorn to 127.0.0.1 by default for the tunnel case; set `LELA_HOST=0.0.0.0` behind a TLS proxy.

---

## 2. Should fix

### Backend (`app.py`)
- **JSON fallback marks the student as passed.** On malformed LLM output the code invents `"passed": True, "next_state": "DRILL"`, so garbage output silently "passes" the student. The draft's fallback uses the fuzzy match and otherwise gives the step's hint.
- **Use structured output instead of parsing fences.** `raw_content.split("```")[1]` breaks on preamble text or nested backticks. vLLM supports `response_format: {"type": "json_schema", ...}`, which forces valid JSON matching a schema. The draft uses it and keeps a tolerant parser as a backup.
- **Silent `continue` on LLM errors.** The student hears nothing and the UI stays on "Processing..." forever. The draft always answers (fallback hint) and sends a `turn_end` message so the UI unlocks.
- **TTS input isn't sanitized.** If the LLM slips Latin letters, digits or parentheses into `kannada_speech`, IndicF5 mispronounces them. The draft strips everything outside the Kannada block before TTS.
- **`english_support` mixes romanization and translation in one string.** The draft splits them into `say_roman` and `say_en` and shows both, which is much easier for a beginner to read.
- **`torch.backends.cudnn.enabled = False` globally** slows every conv layer (IndicF5's vocoder included) to hide a version clash. The usual cause is faster-whisper's CTranslate2 and PyTorch expecting different cuDNN versions. Align them (recent CTranslate2 4.5+ builds expect cuDNN 9, which matches current PyTorch CUDA 12 wheels) and drop the line. The draft keeps it behind `LELA_DISABLE_CUDNN=1` in case you still need it.
- **Models load at import time.** Fine for one worker, but use FastAPI's `lifespan` (draft does) so startup errors are clear, and the missing `prompts/kannada_teacher.wav` is caught at startup rather than as a TTS error on the first turn.
- **ASR guard for tiny clips.** A click on the mic button produces a near-empty clip; the draft skips clips under 0.3 s and enables Whisper's VAD filter, which cuts hallucinated "thank you"-type output on silence.

### Frontend (`index.html`)
- **Touch fires twice.** `ontouchstart/ontouchend` are followed by emulated `mousedown/mouseup`, so a tap on a phone starts a second recording and sends an empty WAV. Releasing the mouse outside the button never fires `onmouseup`, so recording sticks on. The draft uses pointer events with `setPointerCapture`.
- **A new 16 kHz `AudioContext` is created per recording and never closed.** Browsers cap the number of live contexts (Chrome warns around 6), so recording breaks after a handful of turns. Firefox also refuses to connect a 48 kHz mic stream into a 16 kHz context. `ScriptProcessorNode` is deprecated. The draft uses `MediaRecorder` and lets the server decode and resample (faster-whisper's `decode_audio` handles webm/opus and Safari's mp4).
- **Student can talk over the tutor.** Nothing disables the mic during playback, so LeLa's own voice can end up in the answer. The draft locks input while the tutor is thinking or speaking.
- **Overlapping audio.** Two clips arriving close together play on top of each other; the draft queues them.
- **Status badge stays on "Processing..."** after a reply; the draft shows "Your turn" when the turn ends.
- Forcing the output `AudioContext` to 24 kHz is harmless but unnecessary; `decodeAudioData` resamples to the context rate anyway.

### Scripts
- `start_vllm.sh` binds vLLM to `0.0.0.0`, exposing an unauthenticated LLM endpoint to your network. Only `app.py` needs it: bind `127.0.0.1`.
- `--trust-remote-code` isn't needed for Llama 3.1; drop it so no remote code runs.
- `--enforce-eager` disables CUDA graphs and makes generation slower. Fine while debugging; try without it once stable.

---

## 3. Model choices

### ASR: Whisper large-v3 vs AI4Bharat IndicConformer
Whisper large-v3 is weak on Kannada (Kannada was a small slice of its training data, and it tends to hallucinate on short or accented clips). For a learner's short, imperfect Kannada utterances, an Indic-specific model should do much better.
- **`ai4bharat/indic-conformer-600m-multilingual`** (MIT, gated: accept the terms on HF): one model for 22 Indian languages including Kannada, 16 kHz mono input, CTC or RNNT decoding: `model(wav, "kn", "rnnt")`. It needs `onnxruntime-gpu`; the model card pins `onnx/onnxruntime 1.20.1`. The draft uses it by default (`LELA_ASR_KN=indicconformer`) and keeps Whisper for English questions (a `small` Whisper is plenty for English).
- The model card now points to a newer **Indic-Transcribe** model as having better quality; worth a look once the pipeline works.
- **Recommendation:** record 20 to 30 of your own practice answers (the actual phrases from the lesson, spoken as a learner) and compare WER/CER of Whisper large-v3 vs IndicConformer CTC vs RNNT before deciding. Your learner speech is what matters, not benchmark scores.
- Not verified: whether the IndicConformer wrapper returns a string or a list, and whether it runs on GPU via ONNX automatically. The draft handles both return types; check GPU use with `nvidia-smi` on first run.

### TTS: IndicF5 at 24 kHz is correct
The IndicF5 model card writes its output with `samplerate=24000`, and F5-TTS's Vocos vocoder is 24 kHz, so `sf.write(..., 24000)` is right. Two things to check yourself:
- **`REF_AUDIO_TEXT` must be the exact transcript of `kannada_teacher.wav`.** F5 aligns the reference audio against this text; a mismatch causes skipped or garbled words. Keep the clip clean, single-speaker, roughly 5 to 10 seconds. The draft lets you set it with `LELA_REF_TEXT`.
- The card's terms say only clone voices you have explicit permission to use, so make sure the teacher clip is your own recording or licensed.
- The card also mentions a newer **Indic-Speak** model as a successor; optional.

### LLM: Llama 3.1 8B is a weak fit for Kannada
Llama 3.1 8B's Kannada script output is often ungrammatical or mixes in English. Since LeLa's Kannada is what the student learns from, this matters most. Candidates to compare, all servable by vLLM: an Indic-focused model such as Sarvam's (`sarvamai/sarvam-m`), Gemma 3 12B/27B (instruction-tuned, broad multilingual coverage) or a recent Qwen instruct model. Test each on ~20 lesson turns and have a Kannada speaker judge the output. The draft reads the model name from `LELA_LLM_MODEL`, so swapping is one env var. Larger models need more than the 55% GPU memory budget alongside ASR+TTS; check your GPU size.

The fixed lesson lines in `lessons.json` are written by hand rather than by the LLM, so the core teaching phrases stay correct no matter which model you use. **Please have a Kannada speaker check `lessons.json`**: I used standard forms (e.g. ನೀವು ಹೇಗಿದ್ದೀರಿ, ನಾನು ಚೆನ್ನಾಗಿದ್ದೇನೆ) and accepted common colloquial variants, but it deserves a native check.

---

## 4. Corrected drafts in this folder

| File | What changed |
|---|---|
| `app.py` | Models load in `lifespan`; ASR/TTS run in threads with per-model locks; IndicConformer for Kannada, Whisper for English; vLLM JSON-schema output; text input; TTS cache and sanitizing; always replies and sends `turn_end`. |
| `tutor.py` | New. Lesson controller: step/attempt state, dialogue history, LLM prompt and schema, `advance/retry/explain` actions, fuzzy-match and max-attempt guardrails, deterministic fallback. No GPU deps. |
| `lessons.json` | New. Lesson 1: greeting (with greet-back coaching), how are you, name, where from, goodbye. Each step has Kannada, romanized and English text plus a hint. |
| `index.html` | Same look. Pointer events, `MediaRecorder`, "ask in English" button and text box, romanized line, input locked while the tutor speaks, audio queue, proper status. |
| `run_all.sh` | Trap first, process-group cleanup, crash/timeout detection, checks for the reference clip. |
| `start_vllm.sh` | No token, binds 127.0.0.1, no `--trust-remote-code`, model and memory via env vars. |
| `test_tutor.py` | Runs the lesson logic with a fake LLM: ignored greeting gets coached, ASR-noisy ನಮಸ್ಕಾರ passes, English clarification re-asks without costing an attempt, LLM outage falls back, attempts cap, lesson ends. `python3 test_tutor.py` from this folder. |

**Install additions** for the drafts: `pip install onnxruntime-gpu==1.20.2 onnx==1.20.1 torchaudio` (the card says onnxruntime-gpu 1.20.1, which isn't on PyPI) (for IndicConformer), plus your existing `faster-whisper`, `transformers`, `fastapi`, `uvicorn[standard]`, `httpx`, `soundfile`. Accept the gated-model terms for `ai4bharat/indic-conformer-600m-multilingual` and `ai4bharat/IndicF5` on Hugging Face.

**Untested on real models:** vLLM's `response_format` JSON-schema support depends on your vLLM version (supported in recent releases; if yours rejects it, delete that key and the tolerant parser still works). IndicConformer's exact return type and IndicF5 latency are unverified here.

Sources: [IndicF5 model card](https://huggingface-gw-kz.just-ai.com/ai4bharat/IndicF5), [IndicConformer 600M model card](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual), [AI4Bharat IndicConformer collection](https://huggingface.co/collections/ai4bharat/indicconformer).

---

## Update: tutor not speaking, transcript in English letters

**Why the tutor went silent (most likely):** the first draft turned your `torch.backends.cudnn.enabled = False` line into an opt-in setting. IndicF5 needs it on your machine (that's the `CUDNN_STATUS_NOT_INITIALIZED` error your original comment mentions), so every synthesis call failed, the error was caught, and the browser got text but no audio. That was my mistake. Now:
- cuDNN is off for PyTorch by default again (`LELA_DISABLE_CUDNN=0` turns it back on). It only affects IndicF5; Whisper and IndicConformer don't use it.
- If a cuDNN error still happens, TTS turns cuDNN off and retries automatically.
- On startup the server synthesizes the greeting once and prints `TTS self-test OK` or `TTS self-test FAILED: <reason>`.
- `http://localhost:7800/tts_test` plays a test sentence in the browser, or shows the exact TTS error.
- If audio fails during a lesson, the reason now appears in red under the tutor text instead of failing silently.
- Other checks if it is still silent: `prompts/kannada_teacher.wav` must sit next to `app.py`; install `uvicorn[standard]` (plain `uvicorn` has no WebSocket support and the page can't connect at all).

**Also fixed:** faster-whisper 1.2.x's audio decoder breaks with PyAV 19+ (`unexpected keyword argument 'metadata_errors'`), which would make every spoken answer come back as "silence". `app.py` now decodes with PyAV directly.

**Transcript display:** when you speak (or type) Kannada, "You said" shows it in English letters, e.g. ನನ್ನ ಹೆಸರು ತೀರ್ಥಂಕರ್ → *Nanna hesaru tīrthankar*. English is shown as is. The text appears as soon as speech recognition finishes, before the tutor answers. The tutor's own pronunciation line now comes from the same transliterator (`translit.py`), so every romanized line is spelled one consistent way: long vowels with a macron (ā ē ī ō ū), retroflex sounds with a dot (ṭ ḍ ṇ ḷ), and ch/sh instead of c/ś. The LLM no longer writes romanization, which was unreliable.

**Tested without a GPU:** a stub server with stand-in models (a fake IndicF5 that throws the cuDNN error first, a fake ASR) driven by headless Chromium with a fake microphone. The greeting audio plays, a typed ನಮಸ್ಕಾರ shows as *Namaskāra*, and a hold-to-speak recording is decoded, shown as *Nanna hesaru tīrthankar*, and answered with audio. Real models are still untested.

**Python 3.10:** fine. All files compile and `test_tutor.py` plus the browser smoke test pass on Python 3.10.20, with no 3.11+ syntax or library features. `requirements.txt` now pins `onnxruntime-gpu==1.20.2` (1.20.1 doesn't exist on PyPI) and `transformers<5`.

## Update: startup crash in IndicConformer (CUDNN_STATUS_NOT_INITIALIZED)

onnxruntime's CUDA provider couldn't create a cuDNN handle. This is the same error IndicF5 hit in PyTorch, so the GPU's cuDNN setup is the common cause, not the model. The two usual reasons are a cuDNN version clash (PyTorch loads its own cuDNN into the process first, and onnxruntime-gpu 1.20 needs cuDNN 9 with CUDA 12) or the GPU being too full to allocate the handle.

**Code fix:** `app.py` now catches this. If onnxruntime's GPU session fails, IndicConformer loads on CPU (a few short clips per minute are fine on CPU), and the log says which device it ended up on. If IndicConformer fails completely, Whisper large-v3 handles Kannada instead of the server crashing. `LELA_CONFORMER_DEVICE=cpu` skips the GPU attempt.

**To get it on the GPU later**, run these in the venv and compare:
```
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())"
python -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())"
echo $LD_LIBRARY_PATH; ldconfig -p | grep -i cudnn
```
- If torch reports CUDA 11.x or cuDNN 8.x, onnxruntime-gpu 1.20 can't use the GPU with it; stay on CPU or move torch to a CUDA 12 build.
- If `LD_LIBRARY_PATH` points at a system cuDNN (e.g. `/usr/local/cuda/lib64`) that differs from torch's, unset it for the LeLa process.
- If `nvidia-smi` shows the GPU nearly full once vLLM is up, lower vLLM's share: `LELA_VLLM_MEM=0.45 ./run_all.sh`.

**After `nvidia-smi` (two A100 40GB, GPU0 only 23.6 GB used):** memory wasn't the problem, so the error is a cuDNN version clash. You asked to use a single GPU, so `run_all.sh` keeps vLLM, ASR and TTS together on GPU0 (`LELA_GPU` picks another), with vLLM capped at 55% so the speech models fit in the remaining ~18 GB. Run `CUDA_VISIBLE_DEVICES=0 python check_gpu.py` in the venv: it prints the torch, onnxruntime and cuDNN versions, which cuDNN library each actually loads, and runs a tiny cuDNN convolution in both PyTorch and onnxruntime to show which one fails.

**`check_gpu.py` result:** memory and drivers are fine. PyTorch itself fails a basic cuDNN convolution, and torch reports cuDNN 9.20 while pip says `nvidia-cudnn-cu12 9.1.0.70` is installed. So the venv's cuDNN folder mixes files from two cuDNN versions (something installed 9.20 over torch's 9.1), and cuDNN refuses to start when its sub-libraries don't match. That explains both the original IndicF5 error and the IndicConformer crash. `onnxruntime` 1.23.2 (CPU, pulled in by faster-whisper) and `onnxruntime-gpu` 1.20.2 are also both installed and overwrite each other.

Clean fix, in the venv:
```
pip uninstall -y onnxruntime onnxruntime-gpu nvidia-cudnn-cu12
rm -rf ~/.virtualenvs/lelaenv/lib/python3.10/site-packages/nvidia/cudnn
pip install --no-cache-dir nvidia-cudnn-cu12==9.5.1.17
pip install --no-cache-dir --no-deps onnxruntime-gpu==1.20.2
CUDA_VISIBLE_DEVICES=0 python check_gpu.py
```
Both convolutions should then say OK. pip will warn that torch wants 9.1.0.70 and faster-whisper wants `onnxruntime`; both warnings are harmless. If PyTorch still fails, reinstall with `nvidia-cudnn-cu12==9.1.0.70` instead (torch's exact pin); IndicConformer then falls back to CPU if onnxruntime can't use it. Once PyTorch passes, start LeLa with `LELA_DISABLE_CUDNN=0` so IndicF5 uses cuDNN and runs faster. Avoid `pip install -U faster-whisper` later, which would bring the CPU onnxruntime back.

## Update: switched the default TTS to Indic Parler-TTS

IndicF5 clones the voice in `prompts/kannada_teacher.wav` and guesses how long each line should last from that clip. When the guess is off, it clips words (ಒಳ್ಳೆಯದು) or stretches them (ನೀವು) and skips the rest. That matched what you heard, even after sentence splitting.

`app.py` now supports three voices through `LELA_TTS`:
- `parler` (new default): [AI4Bharat Indic Parler-TTS](https://huggingface.co/ai4bharat/indic-parler-tts), Apache 2.0, officially supports Kannada (speakers Anu, Suresh, Chetan, Vidya). No reference clip, so there's no length guessing. The voice is described in words (`LELA_TTS_VOICE`); the default asks Anu to speak slowly and clearly for a learner. The model card also points to a newer AI4Bharat model, Indic-Speak, as better still; I couldn't find a public release of it to wire in.
- `mms`: [Meta MMS-TTS Kannada](https://huggingface.co/facebook/mms-tts-kan). Small, fast, very stable, more robotic. CC-BY-NC 4.0, so personal use only.
- `indicf5`: the old voice clone, kept for comparison.

Install: `pip install git+https://github.com/huggingface/parler-tts.git` (it pins `transformers==4.46.1`, a small step down from your 4.46.3) and accept the model's terms on Hugging Face. Then `./run_all.sh` as usual; the log prints `[Init] TTS: parler at <rate> Hz`. Try `http://localhost:7800/tts_test` first.

**Default voice is now MMS (`LELA_TTS=mms`)**, since Parler-TTS is too slow for a conversation. MMS-TTS Kannada (VITS, 36M parameters) takes Kannada script directly and generates a sentence in a fraction of a second. It speaks at 0.85x speed by default for a learner (`LELA_TTS_RATE`, 1.0 = normal), and a fixed seed makes each line sound the same every time. License is CC-BY-NC 4.0: fine for personal study, not for a commercial product. No extra install beyond `transformers`; the model downloads on first start.


## Update: Sarvam Bulbul voice (`LELA_TTS=sarvam`)

LeLa can now speak with [Sarvam's Bulbul](https://docs.sarvam.ai/api/models/bulbul) cloud voice, which sounds much more natural than MMS. MMS stays loaded (~0.2 GB GPU), so if the API is unreachable or the key is missing, that line is spoken with MMS instead and the log prints `[TTS] Sarvam failed (...); using MMS for this line`. Fallback audio isn't cached, so Sarvam is tried again next time.

Setup: sign up at [dashboard.sarvam.ai](https://dashboard.sarvam.ai) (₹100 free credit), create an API key, then:

```
export SARVAM_API_KEY=...        # keep it out of git
LELA_TTS=sarvam ./run_all.sh
```

Open `http://localhost:7800/tts_test` to hear it. Optional settings: `LELA_SARVAM_SPEAKER` (default `shubh`; also try `anand`, `kavitha`, `roopa`, `vijay`), `LELA_SARVAM_PACE` (default 0.85, slower than normal for a learner), `LELA_SARVAM_MODEL` (default `bulbul:v3`; `bulbul:v4-flash` is about a third of the price). Audio comes back at 24 kHz. Lesson lines are cached in memory, so each line costs once per server run.

Tested here with a stubbed API and stubbed MMS (Sarvam audio is used and cached; with the network down every sentence falls back to MMS at the right sample rate). Not yet tested against the live Sarvam API.

---

## Update: numbered lessons, per-learner progress and a lesson picker (2026-11-06)

- **Lessons** now live in `lessons/lesson1.json`, `lessons/lesson2.json`, ... The number in the file name sets the order. `lesson1.json` is the old `lessons.json` unchanged (plus a one-line `summary_en`), so delete the old `lessons.json`. `lesson2.json` (numbers one to five) and `lesson3.json` (food and drink) are new drafts: **please have a Kannada speaker check them.** To add a lesson, drop in the next `lessonN.json` in the same format and restart.
- **Progress** (new `progress.py`): each learner gets `progress/<name>.json` with one card per lesson: status, where to resume, how each step went (right first time / after coaching / shown by LeLa), times finished, best first-try score, last played. Learners identify themselves by name only, remembered in the browser. That suits a home machine, not a server open to the internet.
- **UI** (`index.html`): enter your name, then pick a lesson from cards that show your progress. *Continue* resumes an unfinished lesson where you left off ("ಮತ್ತೆ ಸ್ವಾಗತ!", welcome back); *Start over* and *Practise again* start from step 1. During a lesson a bar shows the step; at the end you get *Next lesson* / *All lessons*.
- **Server** (`app.py`): `GET /api/lessons?user=<name>` lists lessons and that learner's cards; the WebSocket is now `/ws/tutor?user=<name>&lesson=<id>[&restart=1]`. `LELA_LESSONS_DIR` and `LELA_PROGRESS_DIR` move the folders.
- `test_tutor.py` checks every lesson file, resume, per-learner separation and completion counting. Tested in a browser with stand-in models (no GPU).

## Update: TTS clipping the end of short sentences

LeLa used to cut every line into sentences and voice each one separately, so "ನಮಸ್ಕಾರ!" went to the voice as a one-word request. Sarvam and MMS tend to clip the last syllable of such tiny inputs (heard as "Namaska") and lose the sentence's intonation. With `LELA_TTS=sarvam` or `mms`, `synthesize()` now sends the whole line in one call. IndicF5 and Parler still go sentence by sentence, since F5-style models drop or reorder sentences when given several. Restart LeLa after updating, because audio is cached in memory per run.

## Update: free offline voice (`LELA_TTS=indictts`) and a Sarvam disk cache

**Indic-TTS.** [AI4Bharat Indic-TTS](https://github.com/AI4Bharat/Indic-TTS) Kannada (FastPitch + HiFi-GAN, CC-BY 4.0) now runs as `LELA_TTS=indictts`. It runs on the CPU, so it takes no GPU memory from vLLM. Tested here on CPU: about 4 s to load, then about 1 s for the 4 s greeting. It has `female` (default) and `male` voices (`LELA_INDICTTS_SPEAKER`). Setup is three commands, listed at the top of `indictts.py`. Install coqui-tts with the pins shown there: a plain `pip install coqui-tts` upgrades numpy to 2.x and transformers to 5.x, which would break vLLM 0.6.2. Model files go in `models/indictts/kn/` (about 1.6 GB, `LELA_INDICTTS_DIR` to change). Sample clips: `samples/indictts_female_greeting.wav`, `samples/indictts_male_greeting.wav`.

**Sarvam disk cache.** Sarvam audio is now saved in `tts_cache/` (`LELA_TTS_CACHE_DIR`), keyed by model, speaker, pace and text. Each fixed lesson line is paid for once, ever, not once per server run. Changing the speaker or pace makes new files. Deleting the folder is safe.

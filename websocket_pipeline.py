import asyncio
import json
import torch
from fastapi import FastAPI, WebSocket
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# In-memory session tracking
sessions = {}


@app.websocket("/ws/tutor/{session_id}")
async def tutor_endpoint(websocket: WebSocket, session_id: str):
  await websocket.accept()

  # 1. Start session with LeLa's initial greeting
  initial_payload = {
      "kannada_speech": "ನಮಸ್ಕಾರ! ನೀವು ಹೇಗಿದ್ದೀರಾ?",
      "english_support": "Namaskara! Neevu hegiddira? (Hello! How are you?)",
      "state": "GREETING",
  }
  # Generate initial TTS and push to client
  audio_bytes = generate_tts(initial_payload["kannada_speech"])
  await websocket.send_bytes(audio_bytes)
  await websocket.send_text(json.dumps(initial_payload))

  sessions[session_id] = {"state": "GREETING"}

  try:
    while True:
      # Receive recorded speech from student
      data = await websocket.receive()
      if "bytes" in data:
        pcm_audio = data["bytes"]

        # 2. Run ASR
        student_transcript = run_asr(pcm_audio)  # IndicConformer transcription

        # 3. Step the Tutor Agent
        current_state = sessions[session_id]["state"]
        agent_decision = step_tutor_agent(
            student_transcript=student_transcript, current_state=current_state
        )

        # Update state
        sessions[session_id]["state"] = agent_decision["next_state"]

        # 4. Generate Kannada Audio via TTS
        reply_audio = generate_tts(agent_decision["kannada_speech"])

        # 5. Send Audio and UI Metadata back to React
        await websocket.send_bytes(reply_audio)
        await websocket.send_text(
            json.dumps({
                "student_said": student_transcript,
                "teacher_kannada": agent_decision["kannada_speech"],
                "english_help": agent_decision["english_support"],
                "passed": agent_decision["passed"],
            })
        )

  except Exception as e:
    print(f"Connection ended: {e}")
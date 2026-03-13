import os
import json
import asyncio
import logging
import collections
import numpy as np
import httpx
import edge_tts
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from shared_utils import N8N_WEBHOOK_URL, normalize_username

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount static files for the frontend
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# VAD configuration
VAD_AGGRESSIVENESS = 3  # 0, 1, 2, or 3 (3 is most aggressive)
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

# Audio configuration
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30  # webrtcvad supports 10, 20, or 30ms
PADDING_DURATION_MS = 300  # Duration of silence to trigger end of speech
CHUNK_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000) * 2  # 16-bit mono

import whisper
from concurrent.futures import ThreadPoolExecutor

# Load Whisper model (base is a good balance for testing)
whisper_model = whisper.load_model("base")
executor = ThreadPoolExecutor(max_workers=1)

async def transcribe_audio(audio_bytes: bytes):
    """Transcribe audio bytes using Whisper in a separate thread."""
    def _transcribe():
        # Convert PCM bytes to float32 numpy array
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        result = whisper_model.transcribe(audio_data, fp16=False)
        return result["text"].strip()
    
    return await asyncio.get_event_loop().run_in_executor(executor, _transcribe)

async def get_bot_response(text: str):
    """Call the n8n webhook asynchronously."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(N8N_WEBHOOK_URL, json={"message": text}, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                item = data[0]
                return item.get("output") or item.get("outpput") or str(item)
            if isinstance(data, dict):
                return data.get("output") or data.get("outpput") or str(data)
            return str(data)
        except Exception as e:
            logger.error(f"Error calling n8n: {e}")
            return "I'm sorry, I'm having trouble connecting to my brain."

async def stream_tts_to_ws(websocket: WebSocket, text: str):
    """Stream TTS audio chunks to the WebSocket."""
    communicate = edge_tts.Communicate(text, "en-US-GuyNeural")
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            await websocket.send_bytes(chunk["data"])
    # Send a completion signal
    await websocket.send_json({"type": "tts_complete"})

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    # webrtcvad needs exactly 10, 20 or 30ms
    # 16000Hz * 30ms = 480 samples = 960 bytes
    FRAME_SIZE = 960
    
    pending_audio: bytes = b""
    audio_buffer = collections.deque(maxlen=int(PADDING_DURATION_MS / FRAME_DURATION_MS))
    is_speaking = False
    collected_audio: list[bytes] = []
    
    try:
        while True:
            message = await websocket.receive()
            
            if "text" in message:
                try:
                    text_data = str(message.get("text", ""))
                    data = json.loads(text_data)
                    if data.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                except Exception:
                    pass
                continue
                
            if "bytes" in message:
                chunk_bytes: bytes = message["bytes"]
                pending_audio += chunk_bytes
                
                # Process all complete 30ms frames
                while len(pending_audio) >= FRAME_SIZE:
                    frame = pending_audio[:FRAME_SIZE]
                    pending_audio = pending_audio[FRAME_SIZE:]
                    
                    is_speech = vad.is_speech(frame, SAMPLE_RATE)
                    audio_buffer.append(is_speech)
                    
                    if is_speech:
                        if not is_speaking:
                            logger.info("Speech started")
                            is_speaking = True
                            await websocket.send_json({"type": "status", "value": "listening"})
                        collected_audio.append(frame)
                    else:
                        if is_speaking:
                            if not any(audio_buffer):
                                logger.info("Speech ended")
                                is_speaking = False
                                await websocket.send_json({"type": "status", "value": "thinking"})
                                
                                full_audio = b"".join(collected_audio)
                                collected_audio = []
                                
                                # Transcribe
                                raw_user_text = await transcribe_audio(full_audio)
                                user_text: str = str(raw_user_text)
                                logger.info(f"User: {user_text}")
                                
                                if not user_text or len(user_text) < 2:
                                    await websocket.send_json({"type": "status", "value": "idle"})
                                    continue

                                await websocket.send_json({"type": "transcript", "value": user_text})
                                
                                # Get LLM response
                                raw_bot_text = await get_bot_response(user_text)
                                bot_text: str = str(raw_bot_text)
                                logger.info(f"Bot: {bot_text}")
                                await websocket.send_json({"type": "bot_text", "value": bot_text})
                                
                                # Stream TTS
                                await stream_tts_to_ws(websocket, bot_text)
                            else:
                                collected_audio.append(frame)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

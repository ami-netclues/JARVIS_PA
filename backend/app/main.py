from __future__ import annotations
import logging
from typing import Any, List, Optional
import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from app.settings import settings
from app.auth import auth_manager
import tempfile
import os
import subprocess
import numpy as np
import edge_tts


logger = logging.getLogger("jarvis-backend")


class MessageRequest(BaseModel):
    message: str = Field(default="")
    sessionId: str | None = None
    meta: dict[str, Any] | None = None


class MessageResponse(BaseModel):
    replyText: str


class RegisterRequest(BaseModel):
    name: str
    pcmData: List[float]
    password: str = Field(default="")


class VerifyRequest(BaseModel):
    name: str
    pcmData: List[float]


class AuthResponse(BaseModel):
    success: bool
    message: str
    score: Optional[float] = None
    authenticated: Optional[bool] = None

# FFmpeg conversion removed in favor of direct librosa processing in endpoints to avoid 'FFmpeg not found' errors.



import sys
from logging.handlers import RotatingFileHandler

# Setup logging to both file and console
LOG_FILE = "app.log"
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=2*1024*1024, backupCount=2)
file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
root_logger.handlers = [file_handler, console_handler]

app = FastAPI(title="Jarvis Voice Backend", default_response_class=ORJSONResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    logger.info("Backend starting. Transcribe enabled=%s", settings.enable_transcribe)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/message", response_model=MessageResponse)
async def message(req: MessageRequest) -> MessageResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    if not settings.n8n_webhook_url:
        # Helpful error if you forgot to configure the webhook URL.
        # raise HTTPException(
        #     status_code=500,
        #     detail="N8N_WEBHOOK_URL is not configured on the server. "
        #     "Set it in the backend 'env' file or hosting environment.",
        # )
        return MessageResponse(replyText=req.message.strip())

    payload: dict[str, Any] = {
        "message": req.message,
        # "sessionId": req.sessionId,
        # "meta": req.meta or {"source": "web"},
    }

    def _extract_reply_text(value: Any) -> str:
        """
        Normalize common n8n webhook responses into a single text string.

        Examples handled:
        - {"replyText": "..."}
        - {"output": "..."}
        - [{"output": "..."}]
        - {"text": "..."} / {"message": "..."}
        - plain string
        """
        if value is None:
            return ""

        if isinstance(value, str):
            return value

        if isinstance(value, list):
            if not value:
                return ""
            # n8n commonly returns an array of items; take the first one.
            return _extract_reply_text(value[0])

        if isinstance(value, dict):
            for key in ("replyText", "output", "message", "text", "result", "response"):
                v = value.get(key)
                if isinstance(v, str) and v.strip():
                    return v
                if v is not None and not isinstance(v, (dict, list)):
                    s = str(v).strip()
                    if s:
                        return s

            # If it's a single-key dict, try its only value.
            if len(value) == 1:
                return _extract_reply_text(next(iter(value.values())))

        return str(value)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(settings.n8n_webhook_url, json=payload)
            r.raise_for_status()


            if r.headers.get("content-type", "").startswith("application/json"):
                try:
                    data = r.json()
                except ValueError:
                    data = None  # or handle gracefully
            else:
                data = r.text
            # data: Any = r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text


    except httpx.HTTPError as e:
        logger.exception("n8n webhook call failed")
        raise HTTPException(status_code=502, detail=f"n8n webhook call failed: {e}") from e

    reply_text = _extract_reply_text(data).strip()
    if not reply_text:
        reply_text = "No reply from n8n."
    return MessageResponse(replyText=reply_text)


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict[str, str]:
    if not settings.enable_transcribe:
        raise HTTPException(status_code=404, detail="transcription not enabled")

    content_type = (file.content_type or "").lower()
    if not (content_type.startswith("audio/") or content_type in {"application/octet-stream"}):
        raise HTTPException(status_code=400, detail=f"unsupported content-type: {file.content_type}")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty file")

    # Lazy import so the app can run without Whisper deps installed.
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Whisper deps not installed: {e}") from e

    # NOTE: This is a minimal implementation. For production, consider:
    # - persistent model instance (global) + worker processes
    # - file size limits
    # - authentication / rate limiting
    import io
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model.transcribe(io.BytesIO(audio_bytes), language="en")
    _ = info

    text = " ".join(seg.text.strip() for seg in segments).strip()
    return {"text": text}


# --- Voice Auth Endpoints ---

@app.get("/api/auth/profiles")
async def get_profiles():
    try:
        logger.info("[AUTH] Getting profiles list")
        print("[AUTH] Getting profiles list")
        profiles = auth_manager.get_profiles()
        logger.info(f"[AUTH] Found {len(profiles)} profiles: {profiles}")
        print(f"[AUTH] Found {len(profiles)} profiles: {profiles}")
        return {"profiles": profiles}
    except Exception as e:
        logger.error(f"[AUTH] Exception in get_profiles: {e}")
        print(f"[AUTH] Exception in get_profiles: {e}")
        return {"profiles": []}


@app.delete("/api/auth/profiles/{name}", response_model=AuthResponse)
async def delete_profile(name: str, password: str = ""):
    if password != "J@rv!s#AI2026":
        return AuthResponse(success=False, message="Invalid admin password")
    
    success = auth_manager.delete_profile(name)
    if success:
        return AuthResponse(success=True, message=f"Profile '{name}' deleted")
    else:
        return AuthResponse(success=False, message="Profile not found")


@app.post("/api/auth/register", response_model=AuthResponse)
async def register(file: UploadFile = File(...), user_id: str = Form(...), password: str = Form(default="")):
    logger.info(f"[AUTH] Register request for user: {user_id}")
    print(f"[AUTH] Register request for user: {user_id}")
    
    if password != "J@rv!s#AI2026":
        logger.info(f"[AUTH] Invalid admin password for registration: {user_id}")
        return AuthResponse(success=False, message="Invalid admin password")
    
    try:
        # Read file content
        file_content = await file.read()
        logger.info(f"[AUTH] File read, size: {len(file_content)} bytes")
        print(f"[AUTH] File read, size: {len(file_content)} bytes")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".upload") as temp_input:
            temp_input.write(file_content)
            temp_input_path = temp_input.name
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_output:
            temp_output_path = temp_output.name
            
        try:
            logger.info(f"[AUTH] Calling auth_manager.register_profile for: {user_id}")
            print(f"[AUTH] Calling auth_manager.register_profile for: {user_id}")
            # Pass the raw temp file directly to auth_manager, which now handles librosa.load internally
            success = auth_manager.register_profile(user_id, temp_input_path)
            logger.info(f"[AUTH] Register result: {success}")
            print(f"[AUTH] Register result: {success}")
            
            if success:
                return AuthResponse(success=True, message=f"Profile '{user_id}' registered successfully")
            else:
                return AuthResponse(success=False, message="Failed to register profile (maybe name exists or limit reached)")
        except Exception as e:
            logger.error(f"[AUTH] Exception in auth_manager.register_profile: {e}")
            print(f"[AUTH] Exception in auth_manager.register_profile: {e}")
            return AuthResponse(success=False, message=f"Registration error: {str(e)}")
        finally:
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
    except Exception as e:
        logger.error(f"[AUTH] General exception in register endpoint: {e}")
        print(f"[AUTH] General exception in register endpoint: {e}")
        return AuthResponse(success=False, message=f"Server error: {str(e)}")


@app.post("/api/auth/verify", response_model=AuthResponse)
async def verify(file: UploadFile = File(...), user_id: str = Form(...)):
    logger.info(f"[AUTH] Verify request for user: {user_id}")
    print(f"[AUTH] Verify request for user: {user_id}")
    
    try:
        # Read file content
        file_content = await file.read()
        logger.info(f"[AUTH] File read, size: {len(file_content)} bytes")
        print(f"[AUTH] File read, size: {len(file_content)} bytes")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".upload") as temp_input:
            temp_input.write(file_content)
            temp_input_path = temp_input.name
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_output:
            temp_output_path = temp_output.name
            
        try:
            logger.info(f"[AUTH] Calling auth_manager.verify_voice for: {user_id}")
            print(f"[AUTH] Calling auth_manager.verify_voice for: {user_id}")
            # Pass the raw temp file directly to auth_manager, which now handles librosa.load internally
            authenticated, score = auth_manager.verify_voice(user_id, temp_input_path)
            logger.info(f"[AUTH] Verify result: authenticated={authenticated}, score={score}")
            print(f"[AUTH] Verify result: authenticated={authenticated}, score={score}")
            
            if authenticated:
                return AuthResponse(success=True, message="Authentication successful", score=score, authenticated=True)
            else:
                return AuthResponse(success=False, message=f"Voice verification failed (Score: {score:.3f})", score=score, authenticated=False)
        except Exception as e:
            logger.error(f"[AUTH] Exception in auth_manager.verify_voice: {e}")
            print(f"[AUTH] Exception in auth_manager.verify_voice: {e}")
            return AuthResponse(success=False, message=f"Verification error: {str(e)}")
        finally:
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
    except Exception as e:
        logger.error(f"[AUTH] General exception in verify endpoint: {e}")
        print(f"[AUTH] General exception in verify endpoint: {e}")
        return AuthResponse(success=False, message=f"Server error: {str(e)}")


# --- TTS WebSocket Endpoint ---

VOICE = "en-US-AvaNeural"  # High-quality neural voice


@app.websocket("/ws/tts")
async def stream_tts(websocket: WebSocket):
    logger.info("[TTS] WebSocket endpoint entered.")
    print("[TTS] WebSocket endpoint entered.")
    await websocket.accept()
    try:
        while True:
            logger.info("[TTS] Waiting for text from frontend...")
            print("[TTS] Waiting for text from frontend...")
            text = await websocket.receive_text()
            logger.info(f"[TTS] Received text: {text[:50]}...")
            print(f"[TTS] Received text: {text[:50]}...")
            if not text.strip():
                logger.info("[TTS] Empty text received, skipping.")
                print("[TTS] Empty text received, skipping.")
                continue

            # Call n8n webhook for reply
            try:
                logger.info("[TTS] Calling n8n webhook...")
                logger.info(f"calling: {settings.n8n_webhook_url}")
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.post(settings.n8n_webhook_url, json={"message": text})
                    r.raise_for_status()
                    content_type = r.headers.get("content-type", "")
                    if content_type.startswith("application/json"):
                        try:
                            data = r.json()
                        except Exception as e:
                            logger.info(f"[TTS] Failed to parse JSON from n8n: {e}")
                            print(f"[TTS] Failed to parse JSON from n8n: {e}")
                            data = r.text
                    else:
                        data = r.text
            except Exception as e:
                logger.info(f"[TTS] n8n webhook call failed: {e}")
                print(f"[TTS] n8n webhook call failed: {e}")
                data = None

            # Extract reply text
            def _extract_reply_text(value):
                if value is None:
                    return ""
                if isinstance(value, str):
                    return value
                if isinstance(value, list):
                    if not value:
                        return ""
                    return _extract_reply_text(value[0])
                if isinstance(value, dict):
                    for key in ("replyText", "output", "message", "text", "result", "response"):
                        v = value.get(key)
                        if isinstance(v, str) and v.strip():
                            return v
                        if v is not None and not isinstance(v, (dict, list)):
                            s = str(v).strip()
                            if s:
                                return s
                    if len(value) == 1:
                        return _extract_reply_text(next(iter(value.values())))
                return str(value)

            reply_text = _extract_reply_text(data).strip()
            logger.info(f"[TTS] Extracted reply text: {reply_text[:50]}...")
            print(f"[TTS] Extracted reply text: {reply_text[:50]}...")
            if not reply_text:
                reply_text = "No reply from n8n."

            # Send reply text first
            try:
                logger.info("[TTS] Sending reply text to frontend...")
                print("[TTS] Sending reply text to frontend...")
                await websocket.send_json({"text": reply_text})
            except Exception as e:
                logger.info(f"[TTS] Failed to send JSON to frontend: {e}")
                print(f"[TTS] Failed to send JSON to frontend: {e}")
                continue

            # Synthesize audio using edge-tts
            try:
                logger.info(f"[TTS] Synthesizing audio for text: {reply_text[:50]}...")
                print(f"[TTS] Synthesizing audio for text: {reply_text[:50]}...")
                communicate = edge_tts.Communicate(reply_text, VOICE)
                audio_data = bytearray()
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_data.extend(chunk["data"])
                # Save audio for verification
                if audio_data:
                    try:
                        with open("tts_debug_output.mp3", "wb") as f:
                            f.write(audio_data)
                        logger.info(f"[TTS] Saved debug audio, {len(audio_data)} bytes.")
                        print(f"[TTS] Saved debug audio, {len(audio_data)} bytes.")
                    except Exception as e:
                        logger.info(f"[TTS] Failed to save debug audio: {e}")
                        print(f"[TTS] Failed to save debug audio: {e}")
                    logger.info("[TTS] Sending audio to frontend...")
                    print("[TTS] Sending audio to frontend...")
                    await websocket.send_bytes(bytes(audio_data))
            except Exception as e:
                logger.info(f"[TTS] Failed to synthesize or send audio: {e}")
                print(f"[TTS] Failed to synthesize or send audio: {e}")
    except Exception as e:
        logger.info(f"[TTS] Connection closed: {e}")
        print(f"[TTS] Connection closed: {e}")


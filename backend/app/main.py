from __future__ import annotations
import logging
from typing import Any, List, Optional
import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from app.settings import settings
from app.auth import auth_manager
import tempfile
import os
import subprocess
import numpy as np


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
        raise HTTPException(
            status_code=500,
            detail="N8N_WEBHOOK_URL is not configured on the server. "
            "Set it in the backend 'env' file or hosting environment.",
        )

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
            data: Any = r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
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
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_bytes, language="en")
    _ = info

    text = " ".join(seg.text.strip() for seg in segments).strip()
    return {"text": text}


# --- Voice Auth Endpoints ---

@app.get("/api/auth/profiles")
async def get_profiles():
    return {"profiles": auth_manager.get_profiles()}


@app.delete("/api/auth/profiles/{name}", response_model=AuthResponse)
async def delete_profile(name: str, password: str = ""):
    if password != "pass123":
        return AuthResponse(success=False, message="Invalid admin password")
    
    success = auth_manager.delete_profile(name)
    if success:
        return AuthResponse(success=True, message=f"Profile '{name}' deleted")
    else:
        return AuthResponse(success=False, message="Profile not found")


@app.post("/api/auth/register", response_model=AuthResponse)
async def register(file: UploadFile = File(...), user_id: str = Form(...), password: str = Form(default="")):
    if password != "pass123":
        return AuthResponse(success=False, message="Invalid admin password")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".upload") as temp_input:
        temp_input.write(await file.read())
        temp_input_path = temp_input.name
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_output:
        temp_output_path = temp_output.name
        
    try:
        # Pass the raw temp file directly to auth_manager, which now handles librosa.load internally
        success = auth_manager.register_profile(user_id, temp_input_path)
        if success:
            return AuthResponse(success=True, message=f"Profile '{user_id}' registered successfully")
        else:
            return AuthResponse(success=False, message="Failed to register profile (maybe name exists or limit reached)")
    finally:
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)


@app.post("/api/auth/verify", response_model=AuthResponse)
async def verify(file: UploadFile = File(...), user_id: str = Form(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".upload") as temp_input:
        temp_input.write(await file.read())
        temp_input_path = temp_input.name
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_output:
        temp_output_path = temp_output.name
        
    try:
        # Pass the raw temp file directly to auth_manager, which now handles librosa.load internally
        authenticated, score = auth_manager.verify_voice(user_id, temp_input_path)
        if authenticated:
            return AuthResponse(success=True, message="Authentication successful", score=score, authenticated=True)
        else:
            return AuthResponse(success=False, message=f"Voice verification failed (Score: {score:.3f})", score=score, authenticated=False)
    finally:
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)


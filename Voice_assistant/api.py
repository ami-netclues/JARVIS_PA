import os
import asyncio
import threading
import sys
import edge_tts
import io
import numpy as np
import requests
import warnings
import re
import tempfile
import wave
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# Lazy imports to handle broken environments
# from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
# from faster_whisper import WhisperModel
from database import VoiceDatabase
import json

# Disable unnecessary logs
warnings.filterwarnings("ignore")
# os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" - Removed pygame usage

# --- CONFIGURATION ---
N8N_WEBHOOK_URL = "https://v1netclues.app.n8n.cloud/webhook/2a842b2d-2345-4fc1-a399-1838ac2c1da8"
FS = 16000
AUTH_THRESHOLD = 0.75
DB_PATH = r"c:\Users\darshil.patel\Documents\GitHub\JARVIS_PA\Voice_match\database\voice_profiles.db"

# --- GLOBALS & MODELS ---
db = VoiceDatabase(DB_PATH)
f_ext = None
v_mod = None
vad_mod = None
stt_model = None

app_state = None # Defined later to avoid circular issues

# if not pygame.mixer.get_init():
#     pygame.mixer.init() - Removed pygame usage

# --- AUDIO HELPERS (Production Ready) ---
async def generate_speech(text):
    """Generates audio using edge-tts and returns bytes."""
    try:
        # Using a more "Jarvis-like" British voice with a slight speed boost
        communicate = edge_tts.Communicate(text, "en-GB-RyanNeural", rate="+5%")
        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]
        return audio_bytes
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def load_all_models():
    global f_ext, v_mod, vad_mod, stt_model
    print("Loading models...")
    try:
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
        from faster_whisper import WhisperModel
        import torch
        
        f_ext = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        v_mod = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
        vad_mod, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        stt_model = WhisperModel("base", device="cpu", compute_type="int8")
        print("Models loaded successfully.")
    except (ImportError, Exception) as e:
        print(f"WARNING: ML Models failed to load due to environment error: {e}")
        print("Running in DEMO/MOCK mode without Voice Verification or STT.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app_state.main_loop = asyncio.get_running_loop()
    print(f"DEBUG: Main event loop initialized: {app_state.main_loop}")
    # loading models lazily/in thread
    threading.Thread(target=load_all_models, daemon=True).start()
    yield
    # Shutdown
    pass

app = FastAPI(title="JARVIS PA API", lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "online", "message": "JARVIS PA API is running", "port": 8000}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request, call_next):
    print(f"DEBUG: Incoming {request.method} request to {request.url.path}")
    response = await call_next(request)
    print(f"DEBUG: Response status: {response.status_code}")
    return response

# --- STATE ---
class AppState:
    def __init__(self):
        self.continuous_listening = False
        self.last_audio_file = None
        self.stop_listening_event = threading.Event()
        self.active_websockets = []
        self.enrollment_data = []
        self.main_loop = None

app_state = AppState()

# --- AUDIO HELPERS ---
async def speak_text_ws(text, websocket: WebSocket):
    """Generates audio and streams it via WebSocket in buffered chunks."""
    try:
        # Using en-GB-RyanNeural for the British/Jarvis feel
        communicate = edge_tts.Communicate(text, "en-GB-RyanNeural", rate="+5%")
        
        chunk_buffer = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunk_buffer += chunk["data"]
                # Send once we have a decent chunk (e.g., 32KB) for the browser to decode efficiently
                if len(chunk_buffer) >= 32768:
                    if websocket.client_state.value == 1:
                        await websocket.send_bytes(chunk_buffer)
                    chunk_buffer = b""
                    
        # Send remaining bytes
        if chunk_buffer and websocket.client_state.value == 1:
            await websocket.send_bytes(chunk_buffer)
        
        if websocket.client_state.value == 1:
            await websocket.send_json({"type": "tts_complete"})
            
    except Exception as e:
        print(f"WebSocket TTS streaming error: {e}")

def get_embedding(audio_data):
    global f_ext, v_mod
    if f_ext is None or v_mod is None:
        return None
    try:
        import torch
        waveform = torch.tensor(audio_data).float().unsqueeze(0)
        inputs = f_ext(waveform.squeeze().numpy(), sampling_rate=FS, return_tensors="pt", padding=True)
        with torch.no_grad():
            embeddings = v_mod(**inputs).embeddings
        return torch.nn.functional.normalize(embeddings, dim=-1)
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def verify_voice(audio_data, profiles, threshold=None):
    if not profiles:
        return None
    
    if threshold is None:
        threshold = AUTH_THRESHOLD
        
    current_emb = get_embedding(audio_data)
    if current_emb is None:
        return None
        
    best_score = -1
    best_user = None
    
    # Try using torch for similarity if available
    try:
        import torch
        cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        current_emb_torch = current_emb if torch.is_tensor(current_emb) else torch.tensor(current_emb)
        
        for profile in profiles:
            emb = profile['embedding']
            if not torch.is_tensor(emb):
                emb = torch.tensor(emb)
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            score = cosine_sim(emb, current_emb_torch).item()
            if score > best_score:
                best_score = score
                best_user = profile['user_name']
    except (ImportError, Exception):
        # Fallback to pure numpy
        print("Using numpy fallback for cosine similarity")
        cur_emb_np = current_emb.numpy().flatten() if hasattr(current_emb, 'numpy') else np.array(current_emb).flatten()
        
        for profile in profiles:
            emb_np = np.array(profile['embedding']).flatten()
            # Cosine similarity formula: (A . B) / (||A|| * ||B||)
            norm_a = np.linalg.norm(cur_emb_np)
            norm_b = np.linalg.norm(emb_np)
            if norm_a > 0 and norm_b > 0:
                sim = np.dot(cur_emb_np, emb_np) / (norm_a * norm_b)
            else:
                sim = 0
            if sim > best_score:
                best_score = sim
                best_user = profile['user_name']
            
    print(f"DEBUG: Auth matching - Best Score: {best_score:.4f}, User: {best_user}, Required: {threshold}")
    if best_score >= threshold:
        return best_user
    return None

def broadcast_message(msg: dict):
    if not app_state.main_loop:
        print("Error: Main loop not initialized for broadcasting")
        return
        
    async def _broadcast():
        print(f"Broadcasting: {msg.get('type')}")
        for ws in app_state.active_websockets[:]:
            try:
                await ws.send_json(msg)
            except Exception as e:
                print(f"Broadcast failed for a socket: {e}")
                if ws in app_state.active_websockets:
                    app_state.active_websockets.remove(ws)
                    
    asyncio.run_coroutine_threadsafe(_broadcast(), app_state.main_loop)

def extract_response(resp):
    if isinstance(resp, list) and len(resp) > 0:
        raw = resp[0].get("output") or resp[0].get("outpput") or str(resp[0])
    elif isinstance(resp, dict):
        raw = resp.get("output") or resp.get("outpput") or str(resp)
    else:
        raw = str(resp)
        
    # Simple Repetition Filter (Whisper Hallucination Guard)
    words = raw.split()
    if len(words) > 10:
        # Check if the first 3 words are repeated more than 5 times
        triplet = " ".join(words[:2])
        if raw.count(triplet) > 5:
            return words[0] + "... (Repeated output filtered)"
    return raw

async def process_n8n_request_async(text: str, websocket: WebSocket, is_voice: bool = True):
    payload_text = text + "\n(Please reply in English)"
    try:
        # Use asyncio to run the blocking request
        def _get():
            return requests.post(N8N_WEBHOOK_URL, json={"message": payload_text}).json()
        
        resp = await asyncio.get_event_loop().run_in_executor(None, _get)
        bot_text = extract_response(resp)
    except Exception as e:
        print(f"N8N Error: {e}")
        bot_text = "I encountered an error connecting to my neural network."
        
    # Send WebSocket update
    try:
        if websocket.client_state.value == 1:
            await websocket.send_json({
                "type": "message",
                "role": "assistant",
                "text": bot_text
            })
    except: pass
    
    # Play TTS if requested via voice
    if is_voice and websocket.client_state.value == 1:
        # Signal frontend to stop any leftover audio before we begin
        try:
            await websocket.send_json({"type": "stop_audio"})
        except: pass
        await speak_text_ws(bot_text, websocket)

# --- CONTINUOUS LISTENING LOOP ---
# No longer using a background thread for listening. Logic moved to WebSocket.

# --- REST ENDPOINTS ---

@app.get("/api/profiles")
def get_profiles():
    profiles = db.get_all_profiles()
    return [{"slot_id": p["slot_id"], "user_name": p["user_name"]} for p in profiles]

@app.delete("/api/profiles/{slot_id}")
def delete_profile(slot_id: int):
    success = db.delete_profile(slot_id)
    if success:
        return {"status": "success"}
    raise HTTPException(status_code=400, detail="Failed to delete profile")

# --- WEBSOCKET ---

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    app_state.active_websockets.append(websocket)
    
    # Per-connection State
    audio_buffer = []
    pre_buffer_ws = []
    is_speaking_ws = False
    silence_count_ws = 0
    consecutive_speech_ws = 0
    auth_checked_ws = False
    last_auth_len = 0  # Track audio size when auth was last attempted
    is_enrolling = False
    enroll_data_ws = []
    enroll_step_info = {}
    current_assistant_task = None

    try:
        # Fetch full profiles (including embeddings) for voice verification
        all_profiles = db.get_all_profiles()
        # Keep a UI-friendly version for the sidebar
        ui_profiles = [{"slot_id": p["slot_id"], "user_name": p["user_name"]} for p in all_profiles]
        await websocket.send_json({"type": "status", "listening": False, "profiles": ui_profiles})
        
        while True:
            message = await websocket.receive()
            
            if "text" in message:
                payload = json.loads(message["text"])
                action = payload.get("action")
                
                if action == "start_listening":
                    # Refresh profiles when starting to listen
                    all_profiles = db.get_all_profiles()
                    app_state.continuous_listening = True
                    await websocket.send_json({"type": "status", "listening": True})
                elif action == "stop_listening":
                    app_state.continuous_listening = False
                    await websocket.send_json({"type": "status", "listening": False})
                elif action == "send_text":
                    text = payload.get("text", "")
                    if text:
                        await websocket.send_json({"type": "message", "role": "user", "text": text})
                        await process_n8n_request_async(text, websocket, is_voice=False)
                elif action == "enroll_step":
                    is_enrolling = True
                    enroll_data_ws = []
                    enroll_step_info = {
                        "step": payload.get("step"),
                        "target": payload.get("target"),
                        "slot": payload.get("slot", 1),
                        "name": payload.get("name", "Unknown")
                    }
                    if enroll_step_info["step"] == 1:
                        app_state.enrollment_data = [] # Reset global if starting new
                    await websocket.send_json({"type": "enroll_status", "status": "recording"})

            elif "bytes" in message:
                # Handle incoming binary audio (expected 16kHz 16-bit PCM)
                raw_bytes = message["bytes"]
                chunk = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                
                if not app_state.continuous_listening and not is_enrolling:
                    continue

                # VAD Logic
                import torch
                speech_prob = 0
                if vad_mod:
                    tensor_chunk = torch.tensor(chunk[:512]) # VAD usually works on 512
                    if len(tensor_chunk) == 512:
                        speech_prob = vad_mod(tensor_chunk, FS).item()
                else:
                    rms = np.sqrt(np.mean(chunk**2))
                    speech_prob = 1.0 if rms > 0.02 else 0.0

                if is_enrolling:
                    enroll_data_ws.append(chunk)
                    # Duration-based collection for enrollment
                    target_len = len(enroll_step_info.get("target", ""))
                    required_samples = max(FS * 3, int(target_len * 0.1 * FS))
                    
                    if sum(len(c) for c in enroll_data_ws) >= required_samples:
                        is_enrolling = False
                        final_enroll = np.concatenate(enroll_data_ws).flatten()
                        
                        step = enroll_step_info["step"]
                        name = enroll_step_info["name"]
                        slot = enroll_step_info["slot"]
                        target = enroll_step_info.get("target", "")
                        
                        # --- VALIDATION 1: Silence / No speech check ---
                        rms = np.sqrt(np.mean(final_enroll**2))
                        if rms < 0.005:
                            await websocket.send_json({
                                "type": "enroll_result",
                                "success": False,
                                "error": "No speech detected. Please speak clearly and try again."
                            })
                            enroll_data_ws = []
                            continue
                        
                        # --- VALIDATION 2: Phrase matching ---
                        target_words = re.sub(r'[^\w\s]', '', target.lower()).split()
                        target_word_count = len(target_words)
                        # Expected minimum audio duration: target words / 2.5 words-per-second
                        min_required_samples = int((target_word_count / 2.5) * FS * 0.75)

                        if stt_model:
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                                with wave.open(tf, 'wb') as wav_file:
                                    wav_file.setnchannels(1)
                                    wav_file.setsampwidth(2)
                                    wav_file.setframerate(FS)
                                    wav_file.writeframes((final_enroll * 32767).astype(np.int16).tobytes())
                                tf_path = tf.name
                            
                            segments, _ = stt_model.transcribe(tf_path, task="transcribe", language="en")
                            spoken_text = "".join([s.text for s in segments]).strip()
                            os.remove(tf_path)
                            
                            spoken_words = set(re.sub(r'[^\w\s]', '', spoken_text.lower()).split())
                            target_words_set = set(target_words)
                            
                            print(f"DEBUG ENROLL: Target words={target_words_set} | Spoken words={spoken_words}")
                            
                            # Word overlap: how many target words were spoken
                            if len(target_words_set) == 0:
                                overlap_ratio = 0.0
                            else:
                                overlap_ratio = len(spoken_words & target_words_set) / len(target_words_set)
                            
                            print(f"DEBUG ENROLL: Word overlap ratio={overlap_ratio:.2f}")
                            
                            # Require at least 70% of target words to be spoken
                            if len(spoken_words) == 0 or overlap_ratio < 0.70:
                                await websocket.send_json({
                                    "type": "enroll_result",
                                    "success": False,
                                    "error": f"Please say exactly: '{target}'. You said: '{spoken_text or 'nothing'}'"
                                })
                                enroll_data_ws = []
                                continue
                        else:
                            # Fallback when STT is not loaded: validate via audio duration
                            # If the user only said a short word, audio will be much shorter than required
                            speech_samples = len(final_enroll)
                            print(f"DEBUG ENROLL (no STT): speech_samples={speech_samples}, min_required={min_required_samples}")
                            if speech_samples < min_required_samples:
                                await websocket.send_json({
                                    "type": "enroll_result",
                                    "success": False,
                                    "error": f"Please say the full phrase: '{target}'. Your speech was too short."
                                })
                                enroll_data_ws = []
                                continue
                        
                        # --- ACCEPTED: Append enrollment data ---
                        app_state.enrollment_data.append(final_enroll)
                        
                        if step == 5:
                            unified = np.concatenate(app_state.enrollment_data)
                            emb = get_embedding(unified)
                            if emb is not None:
                                db.save_profile(slot, name, emb.numpy() if hasattr(emb, 'numpy') else emb)
                            else:
                                import torch
                                db.save_profile(slot, name, torch.randn(512).tolist())
                            
                            app_state.enrollment_data = []
                            all_profiles = db.get_all_profiles()
                            ui_profiles = [{"slot_id": p["slot_id"], "user_name": p["user_name"]} for p in all_profiles]
                            await websocket.send_json({
                                "type": "enroll_result", "success": True, "done": True,
                                "profiles": ui_profiles
                            })
                        else:
                            await websocket.send_json({"type": "enroll_result", "success": True, "step": step})

                elif app_state.continuous_listening:
                    if speech_prob > 0.50:
                        consecutive_speech_ws += 1
                        if consecutive_speech_ws >= 2: # Need 2 consecutive chunks
                            if not is_speaking_ws:
                                is_speaking_ws = True
                                audio_buffer.extend(pre_buffer_ws[-5:])
                            audio_buffer.append(chunk)
                            silence_count_ws = 0
                            
                            # Smart Interruption / Early Auth
                            total_len = sum(len(c) for c in audio_buffer)
                            # Interrupt faster if assistant is talking
                            interrupt_limit = int(FS * 0.5) if (current_assistant_task and not current_assistant_task.done()) else int(FS * 1.5)
                            
                            # Rate-limit auth check: only run if we have 0.5s MORE audio than last attempt
                            should_check_auth = (
                                not auth_checked_ws and
                                total_len >= interrupt_limit and
                                (total_len - last_auth_len) >= int(FS * 0.5)
                            )
                            
                            if should_check_auth:
                                last_auth_len = total_len
                                temp_audio = np.concatenate(audio_buffer)
                                recognized = verify_voice(temp_audio, all_profiles)
                                if recognized:
                                    auth_checked_ws = True
                                    last_auth_len = 0
                                    # Interrupt any current speech
                                    if current_assistant_task and not current_assistant_task.done():
                                        current_assistant_task.cancel()
                                        try:
                                            await asyncio.wait_for(asyncio.shield(current_assistant_task), timeout=0.1)
                                        except: pass
                                    await websocket.send_json({"type": "stop_audio"})
                    elif is_speaking_ws:
                        audio_buffer.append(chunk)
                        silence_count_ws += 1
                        if silence_count_ws > 8: # ~2s of silence before processing
                            final_audio = np.concatenate(audio_buffer).flatten()
                            audio_buffer = []
                            is_speaking_ws = False
                            silence_count_ws = 0
                            consecutive_speech_ws = 0
                            last_auth_len = 0  # Reset auth counter for next utterance
                            
                            if len(final_audio) > FS:
                                recognized_user = verify_voice(final_audio, all_profiles, threshold=AUTH_THRESHOLD)
                                
                                if recognized_user or (not all_profiles and stt_model is None):
                                    recognized_display = recognized_user if recognized_user else "Guest (Mock)"
                                    if websocket.client_state.value == 1:
                                        await websocket.send_json({"type": "listening_state", "state": "processing", "user": recognized_display})
                                    
                                    # Save to temp file for Whisper (needs a file for .transcribe)
                                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                                        with wave.open(tf, 'wb') as wav_file:
                                            wav_file.setnchannels(1)
                                            wav_file.setsampwidth(2)
                                            wav_file.setframerate(FS)
                                            wav_file.writeframes((final_audio * 32767).astype(np.int16).tobytes())
                                        tf_path = tf.name
                                    
                                    if stt_model:
                                        segments, _ = stt_model.transcribe(
                                            tf_path, task="transcribe", language="en",
                                            beam_size=5, temperature=0, initial_prompt="A conversation with JARVIS."
                                        )
                                        text = "".join([s.text for s in segments]).strip()
                                    else:
                                        text = "[MOCK MODE: Speech detected]"
                                    
                                    os.remove(tf_path)
                                    if text:
                                        await websocket.send_json({"type": "message", "role": "user", "text": text})
                                        # Cancel old task and wait for it to actually stop
                                        if current_assistant_task and not current_assistant_task.done():
                                            current_assistant_task.cancel()
                                            try:
                                                await asyncio.wait_for(asyncio.shield(current_assistant_task), timeout=0.2)
                                            except: pass
                                        # Ensure frontend stops old audio before we start new TTS
                                        await websocket.send_json({"type": "stop_audio"})
                                        current_assistant_task = asyncio.create_task(process_n8n_request_async(text, websocket, is_voice=True))
                                else:
                                    await websocket.send_json({"type": "listening_state", "state": "unauthorized"})
                                    await speak_text_ws("I'm sorry, you are not an authorized user.", websocket)
                            
                            auth_checked_ws = False
                    else:
                        pre_buffer_ws.append(chunk)
                        if len(pre_buffer_ws) > 10:
                            pre_buffer_ws.pop(0)

    except WebSocketDisconnect:
        print("WebSocket Disconnected")
    except Exception as e:
        print(f"WebSocket Error: {e}")
    finally:
        if websocket in app_state.active_websockets:
            app_state.active_websockets.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

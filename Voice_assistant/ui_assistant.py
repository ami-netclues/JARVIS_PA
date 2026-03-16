import os
import uuid
import torch
import sounddevice as sd
import numpy as np
import soundfile as sf
import pyttsx3
import requests
import warnings
import streamlit as st
import queue
import time
import pygame
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from faster_whisper import WhisperModel

# Disable unnecessary logs
warnings.filterwarnings("ignore")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# --- CONFIGURATION ---
N8N_WEBHOOK_URL = "https://v1netclues.app.n8n.cloud/webhook/2a842b2d-2345-4fc1-a399-1838ac2c1da8"
FS = 16000
AUTH_THRESHOLD = 0.88

# Ensure pygame mixer is initialized for audio playback
if not pygame.mixer.get_init():
    pygame.mixer.init()

# --- PAGE SETUP ---
st.set_page_config(page_title="JARVIS PA", page_icon="🤖", layout="centered")

# Professional Clean UI Styling
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #000000; }
    .user-bubble {
        background-color: #000000; color: #ffffff; padding: 14px 20px; 
        border-radius: 20px 20px 4px 20px; margin: 15px 0 15px auto; 
        max-width: 80%; width: fit-content; font-family: 'Inter', sans-serif;
    }
    .bot-bubble {
        background-color: #f8f9fa; color: #000000; padding: 14px 20px; 
        border-radius: 20px 20px 20px 4px; margin: 15px auto 15px 0; 
        max-width: 80%; width: fit-content; border: 1px solid #e9ecef;
        font-family: 'Inter', sans-serif;
    }
    h1 { font-weight: 800; color: #000000 !important; margin-bottom: 30px; }
    .stButton>button { border: 1px solid #000000; border-radius: 8px; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_all_models():
    f_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    v_model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
    stt = WhisperModel("base", device="cpu", compute_type="int8")
    return f_extractor, v_model, vad_model, utils[0], stt

f_ext, v_mod, vad_mod, get_ts, stt_model = load_all_models()

# --- STATE MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "master_embedding" not in st.session_state:
    st.session_state.master_embedding = None
if "last_audio_file" not in st.session_state:
    st.session_state.last_audio_file = None

# --- HELPER FUNCTIONS ---
def kill_speech():
    """Instantly stops audio playback and frees up the temporary file."""
    if pygame.mixer.get_init():
        pygame.mixer.music.stop()
        try:
            pygame.mixer.music.unload()
        except AttributeError:
            pass
            
    # Clean up the old audio file to prevent storage bloat
    if st.session_state.last_audio_file and os.path.exists(st.session_state.last_audio_file):
        try:
            os.remove(st.session_state.last_audio_file)
        except OSError:
            pass

def speak_text(text):
    """Generates audio dynamically and plays it."""
    kill_speech()
    
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 200)
        
        # UUID prevents file lock crashes if the system holds onto the old file
        filename = f"ui_reply_{uuid.uuid4().hex[:6]}.wav"
        engine.save_to_file(text, filename)
        engine.runAndWait()
        
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        st.session_state.last_audio_file = filename
    except Exception as e:
        print(f"TTS Error: {e}")

def get_embedding(audio_data):
    waveform = torch.tensor(audio_data).float().unsqueeze(0)
    inputs = f_ext(waveform.squeeze().numpy(), sampling_rate=FS, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = v_mod(**inputs).embeddings
    return torch.nn.functional.normalize(embeddings, dim=-1)

def record_until_silence(max_time=300, silence_limit=1.5, energy_threshold=0.01):
    q = queue.Queue()
    def callback(indata, frames, time_info, status):
        q.put(indata.copy())
    
    audio_data = []
    has_spoken = False
    silence_start = None

    with sd.InputStream(samplerate=FS, channels=1, callback=callback):
        start_time = time.time()
        while time.time() - start_time < max_time:
            chunk = q.get()
            audio_data.append(chunk)
            volume = np.sqrt(np.mean(chunk**2))
            
            if volume > energy_threshold:
                has_spoken = True
                silence_start = None 
            elif has_spoken:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > silence_limit:
                    break
    return np.concatenate(audio_data).flatten() if audio_data else np.array([])

# --- MAIN UI ---
st.markdown("<h1 style='text-align:center;'>🤖 JARVIS</h1>", unsafe_allow_html=True)

# Display Full Chat (No container scroll)
for msg in st.session_state.messages:
    div_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
    st.markdown(f"<div class='{div_class}'>{msg['content']}</div>", unsafe_allow_html=True)

st.divider()
col1, col2 = st.columns([6, 1])

with col1:
    user_input = st.chat_input("Message JARVIS...")

with col2:
    mic_btn = st.button("🎙️ Mic", use_container_width=True)

# Processing Logic
if mic_btn:
    kill_speech()  # Force the old voice to stop instantly
    if st.session_state.master_embedding is None:
        st.error("Please register your voice in the sidebar.")
    else:
        with st.spinner("Listening..."):
            audio = record_until_silence()
            if len(audio) > FS:
                current_emb = get_embedding(audio)
                score = torch.nn.CosineSimilarity(dim=-1)(st.session_state.master_embedding, current_emb).item()
                if score >= AUTH_THRESHOLD:
                    with st.spinner("Processing..."):
                        sf.write("temp.wav", audio, FS)
                        segments, _ = stt_model.transcribe("temp.wav")
                        text = "".join([s.text for s in segments]).strip()
                        if text:
                            st.session_state.messages.append({"role": "user", "content": text})
                            resp = requests.post(N8N_WEBHOOK_URL, json={"message": text}).json()
                            bot_text = resp[0].get("output") or resp[0].get("outpput") if isinstance(resp, list) else resp.get("output")
                            st.session_state.messages.append({"role": "assistant", "content": bot_text, "spoken": False})
                            st.rerun()

if user_input:
    kill_speech()  # Stop the old voice instantly
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get response from n8n
    resp = requests.post(N8N_WEBHOOK_URL, json={"message": user_input}).json()
    bot_text = resp[0].get("output") or resp[0].get("outpput") if isinstance(resp, list) else resp.get("output")
    
    # Add to history and mark as NOT YET SPOKEN
    st.session_state.messages.append({"role": "assistant", "content": bot_text, "spoken": False})
    st.rerun()

# Execute Audio for the new response
if st.session_state.messages:
    last_msg = st.session_state.messages[-1]
    if last_msg["role"] == "assistant" and not last_msg.get("spoken", False):
        st.session_state.messages[-1]["spoken"] = True
        speak_text(last_msg["content"])

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Settings")
    if st.button("🎤 Register Voice", use_container_width=True):
        rec = sd.rec(int(6 * FS), samplerate=FS, channels=1, dtype='float32')
        sd.wait()
        st.session_state.master_embedding = get_embedding(rec.flatten())
        st.rerun()
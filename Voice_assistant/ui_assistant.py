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
if "continuous_listening" not in st.session_state:
    st.session_state.continuous_listening = False

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
    """Generates audio dynamically and plays it asynchronously."""
    kill_speech()
    
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 200)
        
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

def wait_for_speech_and_record(master_embedding):
    """Waits for speech, intelligently interrupts if auth passes, and records until silence."""
    q = queue.Queue()
    def callback(indata, frames, time_info, status):
        q.put(indata.copy())

    audio_data = []
    is_speaking = False
    silence_count = 0
    consecutive_speech = 0
    auth_checked = False
    auth_passed_early = False

    with sd.InputStream(samplerate=FS, channels=1, blocksize=512, callback=callback):
        while True:
            chunk = q.get().flatten()
            
            tensor_chunk = torch.tensor(chunk)
            if len(tensor_chunk) == 512:
                speech_prob = vad_mod(tensor_chunk, FS).item()
            else:
                speech_prob = 1.0 if np.sqrt(np.mean(chunk**2)) > 0.01 else 0.0

            if speech_prob > 0.65:
                consecutive_speech += 1
                if consecutive_speech >= 2:
                    if not is_speaking:
                        is_speaking = True
                    audio_data.append(chunk)
                    silence_count = 0
                    
                    # Smart Interruption Logic: Buffer ~0.9s, then verify before killing speech
                    current_len = sum(len(c) for c in audio_data)
                    if not auth_checked and current_len >= int(FS * 0.9):
                        temp_audio = np.concatenate(audio_data)
                        emb = get_embedding(temp_audio)
                        if emb is not None and master_embedding is not None:
                            score = torch.nn.CosineSimilarity(dim=-1)(master_embedding, emb).item()
                            if score >= AUTH_THRESHOLD:
                                auth_passed_early = True
                                kill_speech() # Only interrupt for the correct user
                        auth_checked = True
            elif is_speaking:
                audio_data.append(chunk)
                silence_count += 1
                if silence_count > 35:  # Approx 1.5 seconds of silence stops recording
                    break

    final_audio = np.concatenate(audio_data).flatten() if audio_data else np.array([])
    return final_audio, auth_passed_early

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Settings")
    
    if st.button("🎤 Register Voice", use_container_width=True):
        status_placeholder = st.empty()
        
        # Step-by-step visual feedback for registration
        with status_placeholder.container():
            st.info("🎙️ Initializing microphone...")
            time.sleep(1)
            st.warning("🗣️ Recording started! Please speak naturally for 6 seconds...")
            
        rec = sd.rec(int(6 * FS), samplerate=FS, channels=1, dtype='float32')
        sd.wait()
        
        with status_placeholder.container():
            st.info("⚙️ Processing audio and building voice profile...")
            st.session_state.master_embedding = get_embedding(rec.flatten())
            st.success("✅ Voice successfully registered! JARVIS is now listening continuously.")
            
        time.sleep(2.5)
        status_placeholder.empty()
        st.session_state.continuous_listening = True # Automatically start continuous mode
        st.rerun()

    if st.session_state.master_embedding is not None:
        st.success("Authentication Profile: Active")
    else:
        st.warning("Authentication Profile: Not Set")

# --- MAIN UI ---
st.markdown("<h1 style='text-align:center;'>🤖 JARVIS</h1>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    div_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
    st.markdown(f"<div class='{div_class}'>{msg['content']}</div>", unsafe_allow_html=True)

st.divider()
col1, col2 = st.columns([6, 1])

with col1:
    user_input = st.chat_input("Message JARVIS...")

with col2:
    if st.session_state.continuous_listening:
        mic_btn = st.button("🛑 Stop", use_container_width=True)
    else:
        mic_btn = st.button("🎙️ Mic", use_container_width=True)

# Processing Text Input
if user_input:
    kill_speech()
    st.session_state.continuous_listening = False  # Typing disables continuous voice mode
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    payload_text = user_input + "\n(Please reply in English)"
    resp = requests.post(N8N_WEBHOOK_URL, json={"message": payload_text}).json()
    bot_text = resp[0].get("output") or resp[0].get("outpput") if isinstance(resp, list) else resp.get("output")
    
    st.session_state.messages.append({"role": "assistant", "content": bot_text, "spoken": False})
    st.rerun()

# Processing Mic Button Toggle
if mic_btn:
    kill_speech()
    if st.session_state.continuous_listening:
        st.session_state.continuous_listening = False
        st.rerun()
    else:
        if st.session_state.master_embedding is None:
            st.error("Please register your voice in the sidebar first.")
        else:
            st.session_state.continuous_listening = True
            st.rerun()

# Playback execution for the latest unspoken message
if st.session_state.messages:
    last_msg = st.session_state.messages[-1]
    if last_msg["role"] == "assistant" and not last_msg.get("spoken", False):
        st.session_state.messages[-1]["spoken"] = True
        speak_text(last_msg["content"])

# --- CONTINUOUS LISTENING LOOP ---
if st.session_state.continuous_listening:
    with st.spinner("Listening... (Speak naturally to interact or interrupt)"):
        audio, auth_passed = wait_for_speech_and_record(st.session_state.master_embedding)
        
        if len(audio) > FS:  # Ignore clips less than 1 second
            if not auth_passed:
                # Do a final check if it wasn't validated early during the interruption logic
                current_emb = get_embedding(audio)
                if current_emb is not None:
                    score = torch.nn.CosineSimilarity(dim=-1)(st.session_state.master_embedding, current_emb).item()
                    auth_passed = (score >= AUTH_THRESHOLD)
            
            if auth_passed:
                with st.spinner("Processing..."):
                    sf.write("temp.wav", audio, FS)
                    # task="translate" forces English transcription regardless of spoken language
                    segments, _ = stt_model.transcribe("temp.wav", task="translate")
                    text = "".join([s.text for s in segments]).strip()
                    if text:
                        st.session_state.messages.append({"role": "user", "content": text})
                        try:
                            # Force english context for the webhook
                            payload_text = text + "\n(Please reply in English)"
                            resp = requests.post(N8N_WEBHOOK_URL, json={"message": payload_text}).json()
                            bot_text = resp[0].get("output") or resp[0].get("outpput") if isinstance(resp, list) else resp.get("output")
                        except Exception as e:
                            bot_text = "I encountered an error connecting to my neural network."
                            
                        st.session_state.messages.append({"role": "assistant", "content": bot_text, "spoken": False})
            else:
                st.toast("Voice signature not recognized. Request ignored.", icon="🛡️")
                
    # Automatically loop without requiring another button click
    st.rerun()
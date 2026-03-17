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
import re
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from faster_whisper import WhisperModel
from database import VoiceDatabase

# Disable unnecessary logs
warnings.filterwarnings("ignore")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# --- CONFIGURATION ---
N8N_WEBHOOK_URL = "https://v1netclues.app.n8n.cloud/webhook/2a842b2d-2345-4fc1-a399-1838ac2c1da8"
FS = 16000
AUTH_THRESHOLD = 0.88
DB_PATH = r"c:\Users\darshil.patel\Documents\GitHub\JARVIS_PA\Voice_match\database\voice_profiles.db"
db = VoiceDatabase(DB_PATH)

# Ensure pygame mixer is initialized for audio playback
if not pygame.mixer.get_init():
    pygame.mixer.init()

# --- PAGE SETUP ---
st.set_page_config(page_title="JARVIS PA", page_icon="🤖", layout="centered")

# Professional Clean UI Styling
st.markdown("""
    <style>
    /* Inter font for a modern feel */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    .stApp { font-family: 'Inter', sans-serif; }

    /* Bubble Styling - Theme Aware */
    .user-container {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        margin-bottom: 20px;
    }
    .bot-container {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        margin-bottom: 20px;
    }
    
    .user-bubble {
        background-color: #000000; color: #ffffff; padding: 12px 18px; 
        border-radius: 20px 20px 4px 20px;
        max-width: 80%; width: fit-content;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .bot-bubble {
        background-color: #f0f2f6; color: #1f1f1f; padding: 12px 18px; 
        border-radius: 20px 20px 20px 4px; border: 1px solid #e0e0e0;
        max-width: 80%; width: fit-content;
    }
    
    /* Dark Mode Overrides for Bot Bubble */
    @media (prefers-color-scheme: dark) {
        .bot-bubble {
            background-color: #262730; color: #e0e0e0; border: 1px solid #464855;
        }
        .user-bubble {
            background-color: #ffffff; color: #000000;
        }
    }

    .bubble-label {
        font-size: 0.75rem; font-weight: 600; color: #888; margin-bottom: 4px;
        text-transform: uppercase; letter-spacing: 0.5px;
    }

    h1 { font-weight: 800; text-align: center; margin-bottom: 20px; }
    .stButton>button { border-radius: 10px; font-weight: 600; transition: all 0.2s; }
    .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
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
if "profiles" not in st.session_state:
    st.session_state.profiles = db.get_all_profiles()
if "last_audio_file" not in st.session_state:
    st.session_state.last_audio_file = None
if "continuous_listening" not in st.session_state:
    st.session_state.continuous_listening = False
if "enrollment_step" not in st.session_state:
    st.session_state.enrollment_step = 0
if "enrollment_data" not in st.session_state:
    st.session_state.enrollment_data = []
if "enroll_slot" not in st.session_state:
    st.session_state.enroll_slot = 1
if "enroll_name" not in st.session_state:
    st.session_state.enroll_name = ""
if "last_auth_time" not in st.session_state:
    st.session_state.last_auth_time = 0
if "last_recognized_user" not in st.session_state:
    st.session_state.last_recognized_user = None

# --- HELPER FUNCTIONS ---
def kill_speech():
    """Instantly stops audio playback and frees up the temporary file."""
    if pygame.mixer.get_init():
        pygame.mixer.music.stop()
        try:
            pygame.mixer.music.unload()
        except AttributeError:
            print("Pygame version does not support unload, skipping.")
            
    # Clean up the old audio file to prevent storage bloat
    if st.session_state.last_audio_file and os.path.exists(st.session_state.last_audio_file):
        try:
            os.remove(st.session_state.last_audio_file)
        except OSError:
            print(f"Failed to delete {st.session_state.last_audio_file}")
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
    """Generates a normalized voice embedding from audio data."""
    waveform = torch.tensor(audio_data).float().unsqueeze(0)
    inputs = f_ext(waveform.squeeze().numpy(), sampling_rate=FS, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = v_mod(**inputs).embeddings
    return torch.nn.functional.normalize(embeddings, dim=-1)

def verify_voice(audio_data, threshold=AUTH_THRESHOLD):
    if not st.session_state.profiles:
        return None
    
    current_emb = get_embedding(audio_data)
    if current_emb is None:
        return None
        
    best_score = -1
    best_user = None
    
    cosine_sim = torch.nn.CosineSimilarity(dim=-1)
    for profile in st.session_state.profiles:
        emb = torch.tensor(profile['embedding'])
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        score = cosine_sim(emb, current_emb).item()
        if score > best_score:
            best_score = score
            best_user = profile['user_name']
            
    if best_score >= threshold:
        st.session_state.last_auth_time = time.time()
        st.session_state.last_recognized_user = best_user
        return best_user
    return None

def wait_for_speech_and_record():
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
                        recognized_user = verify_voice(temp_audio)
                        if recognized_user:
                            auth_passed_early = True
                            kill_speech() # Only interrupt for correct user
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
    
    st.subheader("🎤 Voice Registration")
    
    # Registration Flow UI
    if st.session_state.enrollment_step == 0:
        st.session_state.enroll_slot = st.selectbox("Select Slot", [1, 2, 3], index=st.session_state.enroll_slot - 1)
        st.session_state.enroll_name = st.text_input("User Name", value=st.session_state.enroll_name)
        
        if st.button("Start Registration"):
            if not st.session_state.enroll_name.strip():
                st.error("Please enter a name.")
            else:
                st.session_state.enrollment_step = 1
                st.session_state.enrollment_data = []
                st.rerun()
    else:
        sentences = [
            "I use Jarvis every single day",
            "I am registering my voice for the JARVIS personal assistant.",
            "Voice authentication keeps me safe",
            "The integration of voice authentication enhances system security.",
            "My voice is my password, and it is unique to me."
        ]
        
        step = st.session_state.enrollment_step
        if step <= 5:
            target = sentences[step-1]
            st.write(f"**Step {step} of 5**")
            st.info(f"Please say: \"{target}\"")
            
            if st.button(f"Record Sentence {step}", key=f"rec_{step}"):
                with st.spinner("Recording..."):
                    duration = max(3, len(target.split()) * 0.6)
                    rec = sd.rec(int(duration * FS), samplerate=FS, channels=1, dtype='float32')
                    sd.wait()
                    
                    # Extract speech and verify with Whisper
                    audio = rec.flatten()
                    sf.write("enroll_temp.wav", audio, FS)
                    segments, _ = stt_model.transcribe("enroll_temp.wav", task="translate")
                    spoken_text = "".join([s.text for s in segments]).strip().lower()
                    
                    target_clean = re.sub(r'[^\w\s]', '', target.lower())
                    spoken_clean = re.sub(r'[^\w\s]', '', spoken_text)
                    
                    if len(spoken_clean) < len(target_clean) * 0.8:
                        st.error(f"Did not match closely enough. You said: \"{spoken_text}\"")
                    else:
                        st.session_state.enrollment_data.append(audio)
                        st.session_state.enrollment_step += 1
                        st.rerun()
        else:
            with st.spinner("Processing final voice profile..."):
                unified = np.concatenate(st.session_state.enrollment_data)
                emb = get_embedding(unified)
                if emb is not None:
                    db.save_profile(st.session_state.enroll_slot, st.session_state.enroll_name, emb)
                    st.session_state.profiles = db.get_all_profiles()
                    st.success(f"✅ Registered {st.session_state.enroll_name}!")
                    st.session_state.enrollment_step = 0
                    st.session_state.enrollment_data = []
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("Failed to generate embedding.")
                    st.session_state.enrollment_step = 0

    st.divider()
    st.subheader("👥 Profiles")
    for p in st.session_state.profiles:
        st.text(f"Slot {p['slot_id']}: {p['user_name']}")
        if st.button(f"Delete Slot {p['slot_id']}", key=f"del_{p['slot_id']}"):
            db.delete_profile(p['slot_id'])
            st.session_state.profiles = db.get_all_profiles()
            st.rerun()

    if st.session_state.profiles:
        st.success("Authentication: Active")
    else:
        st.warning("Authentication: No Profiles Set")

# --- MAIN UI ---
st.markdown("<h1 style='text-align:center;'>🤖 JARVIS</h1>", unsafe_allow_html=True)

# Keep chat history inside a fixed-height scrollable box
chat_container = st.container(height=450, border=True)
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-container'><div class='bubble-label'>You</div><div class='user-bubble'>{msg['content']}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-container'><div class='bubble-label'>Jarvis</div><div class='bot-bubble'>{msg['content']}</div></div>", unsafe_allow_html=True)

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
        if not st.session_state.profiles:
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
    with st.spinner("Listening..."):
        audio, auth_passed_early = wait_for_speech_and_record()
        
        recognized_user = None
        current_time = time.time()
        grace_period_active = (current_time - st.session_state.last_auth_time < 15)
        
        if len(audio) > int(FS * 0.4):  # Reduced from 1.0s to 0.4s to allow "ok", "yes"
            if auth_passed_early:
                recognized_user = st.session_state.last_recognized_user
            else:
                # Use a slightly more lenient threshold during conversation grace period
                temp_threshold = 0.82 if grace_period_active else AUTH_THRESHOLD
                recognized_user = verify_voice(audio, threshold=temp_threshold)
            
            # If still not recognized but in grace period, trust it's the same user for short clips
            if not recognized_user and grace_period_active and len(audio) < FS:
                recognized_user = st.session_state.last_recognized_user
            
            if recognized_user:
                with st.spinner(f"Processing ({recognized_user})..."):
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
                st.toast("⚠️ You are not an authenticated person.", icon="🛡️")
                speak_text("I'm sorry, you are not an authorized user.")
                
    # Automatically loop without requiring another button click
    st.rerun()
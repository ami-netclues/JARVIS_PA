import streamlit as st
import requests
import numpy as np
from pathlib import Path
import re
import time
from Voice_match.voice_auth import (
    cosine_similarity,
    load_voice_embedding,
    load_voice_sequences,
    load_voice_meta,
    dtw_distance,
    best_prompt_match,
)

# ── Global Constants ─────────────────────────────────────────────────────────
N8N_WEBHOOK_URL = "https://v1netclues.app.n8n.cloud/webhook/2a842b2d-2345-4fc1-a399-1838ac2c1da8"
VOICE_USERS_DIR = Path("Voice_match/database/users")
MAX_REGISTERED_USERS = 5

VOICE_SIMILARITY_MEAN_THRESHOLD = 0.72
VOICE_SIMILARITY_MAX_THRESHOLD = 0.77
REGISTRATION_MIN_SIMILARITY = 0.65
VOICE_DTW_DISTANCE_THRESHOLD = 0.65
VOICE_CHALLENGE_MIN_RATIO = 0.78
REGISTRATION_PHRASES = [
    "My voice secures my account",
    "I use Jarvis every single day",
    "Voice authentication keeps me safe",
    "This is my personal registration sample",
    "Jarvis please verify my identity",
]

# ── Session state initialization ─────────────────────────────────────────────
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "mic_text" not in st.session_state:
        st.session_state.mic_text = ""
    if "mic_error" not in st.session_state:
        st.session_state.mic_error = ""
    if "pending_user_text" not in st.session_state:
        st.session_state.pending_user_text = ""
    if "voice_reg_active" not in st.session_state:
        st.session_state.voice_reg_active = False
    if "voice_reg_index" not in st.session_state:
        st.session_state.voice_reg_index = 0
    if "voice_reg_embeddings" not in st.session_state:
        st.session_state.voice_reg_embeddings = []
    if "voice_reg_sequences" not in st.session_state:
        st.session_state.voice_reg_sequences = []
    if "voice_reg_status" not in st.session_state:
        st.session_state.voice_reg_status = ""
    if "voice_reg_error" not in st.session_state:
        st.session_state.voice_reg_error = ""
    if "voice_username" not in st.session_state:
        st.session_state.voice_username = ""
    if "voice_verify_status" not in st.session_state:
        st.session_state.voice_verify_status = ""
    if "voice_verify_error" not in st.session_state:
        st.session_state.voice_verify_error = ""
    if "voice_verified_until" not in st.session_state:
        st.session_state.voice_verified_until = 0.0
    if "handsfree_active" not in st.session_state:
        st.session_state.handsfree_active = False
    if "handsfree_status" not in st.session_state:
        st.session_state.handsfree_status = ""
    if "interaction_status" not in st.session_state:
        st.session_state.interaction_status = ""

# ── Helper functions ──────────────────────────────────────────────────────────
def normalize_username(username: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", username.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized

def get_user_profile_path(username: str) -> Path:
    return VOICE_USERS_DIR / normalize_username(username)

def list_registered_users() -> list[str]:
    if not VOICE_USERS_DIR.exists():
        return []
    profiles = []
    for npy_path in VOICE_USERS_DIR.glob("*.npy"):
        stem = npy_path.stem
        if not stem.endswith(".seq"):
            profiles.append(stem)
    return sorted(profiles)

def can_use_chat(username: str) -> tuple[bool, str]:
    normalized_username = normalize_username(username)
    if not normalized_username:
        return False, "Enter a user name first."
    if load_voice_embedding(get_user_profile_path(normalized_username)) is None:
        return False, f"User '{normalized_username}' is not registered. Register voice first."
    return True, ""

def evaluate_voice_match(stored_profile: np.ndarray, probe_embedding: np.ndarray) -> tuple[bool, float, float]:
    if stored_profile.ndim == 1:
        score = cosine_similarity(stored_profile, probe_embedding)
        passed = score >= VOICE_SIMILARITY_MEAN_THRESHOLD
        return passed, score, score

    scores = [cosine_similarity(ref_embedding, probe_embedding) for ref_embedding in stored_profile]
    mean_score = float(np.mean(scores))
    max_score = float(np.max(scores))
    passed = mean_score >= VOICE_SIMILARITY_MEAN_THRESHOLD and max_score >= VOICE_SIMILARITY_MAX_THRESHOLD
    return passed, mean_score, max_score

def evaluate_voice_match_with_thresholds(
    stored_profile: np.ndarray,
    probe_embedding: np.ndarray,
    required_mean: float,
    required_max: float,
) -> tuple[bool, float, float]:
    if stored_profile.ndim == 1:
        score = cosine_similarity(stored_profile, probe_embedding)
        passed = score >= required_mean
        return passed, score, score

    scores = [cosine_similarity(ref_embedding, probe_embedding) for ref_embedding in stored_profile]
    mean_score = float(np.mean(scores))
    max_score = float(np.max(scores))
    passed = mean_score >= required_mean and max_score >= required_max
    return passed, mean_score, max_score

def bot_reply(user_text: str) -> str:
    try:
        response = requests.post(
            N8N_WEBHOOK_URL,
            json={"message": user_text},
            timeout=200,
        )
        response.raise_for_status()
        raw = response.text.strip()
        if not raw:
            return "⚠️ No response from n8n. Make sure the webhook workflow is **active**."
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            item = data[0]
            return item.get("output") or item.get("outpput") or str(item)
        if isinstance(data, dict):
            return data.get("output") or data.get("outpput") or str(data)
        return str(data)
    except requests.exceptions.Timeout:
        return "⏱️ Request timed out."
    except requests.exceptions.ConnectionError:
        return "🌐 Could not reach the server."
    except requests.exceptions.HTTPError as e:
        return f"❌ Server error: {e.response.status_code}"
    except Exception as e:
        return f"❌ Error: {e}"

# ── Custom CSS ────────────────────────────────────────────────────────────────
def apply_custom_css():
    st.markdown(
        """
        <style>
            .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
            .user-bubble {
                background: #4f46e5;
                color: #fff;
                padding: 10px 16px;
                border-radius: 18px 18px 4px 18px;
                margin: 6px 0 6px auto;
                max-width: 75%;
                width: fit-content;
                word-wrap: break-word;
            }
            .bot-bubble {
                background: #1e293b;
                color: #e2e8f0;
                padding: 10px 16px;
                border-radius: 18px 18px 18px 4px;
                margin: 6px auto 6px 0;
                max-width: 75%;
                width: fit-content;
                word-wrap: break-word;
            }
            .role-label {
                font-size: 0.72rem;
                opacity: 0.6;
                margin-bottom: 2px;
            }
            .chat-wrapper { display: flex; flex-direction: column; gap: 4px; }
            div[data-testid="stTextInput"] input {
                background: #1e293b !important;
                color: #f1f5f9 !important;
                border: 1px solid #4f46e5 !important;
                border-radius: 10px !important;
            }
            .thinking-bubble {
                background: #1e293b;
                color: #94a3b8;
                padding: 10px 18px;
                border-radius: 18px 18px 18px 4px;
                margin: 6px auto 6px 0;
                width: fit-content;
                font-style: italic;
                animation: pulse 1.2s ease-in-out infinite;
            }
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50%       { opacity: 0.4; }
            }
            .stButton > button {
                border-radius: 10px !important;
                font-weight: 600 !important;
            }
            #MainMenu, header, footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )

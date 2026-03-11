import streamlit as st
import requests
import numpy as np
from pathlib import Path
import re
from simple_tts import text_to_voice
from Voice_match.voice_auth import (
    capture_speech_and_embedding,
    cosine_similarity,
    load_voice_embedding,
    load_voice_sequences,
    load_voice_meta,
    dtw_distance,
    best_prompt_match,
    prompt_matches,
    save_voice_profile,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JARVIS Chat",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Dark gradient background */
        .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

        /* Chat bubble wrappers */
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

        /* Input row */
        div[data-testid="stTextInput"] input {
            background: #1e293b !important;
            color: #f1f5f9 !important;
            border: 1px solid #4f46e5 !important;
            border-radius: 10px !important;
        }

        /* Thinking / typing indicator bubble */
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

        /* Buttons */
        .stButton > button {
            border-radius: 10px !important;
            font-weight: 600 !important;
        }

        /* Hide Streamlit default header */
        #MainMenu, header, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
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

# ── n8n Webhook ───────────────────────────────────────────────────────────────
N8N_WEBHOOK_URL = "https://v1netclues.app.n8n.cloud/webhook/2a842b2d-2345-4fc1-a399-1838ac2c1da8"
VOICE_USERS_DIR = Path("Voice_match/database/users")
MAX_REGISTERED_USERS = 3
# Speaker verification thresholds (cosine similarity) tuned for resemblyzer-style embeddings.
# If resemblyzer isn't installed, `voice_auth.py` falls back to weaker features; in that case
# you may need to raise thresholds and/or re-register users.
# Defaults (used when a profile has no adaptive thresholds yet).
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


def reset_registration_state():
    st.session_state.voice_reg_active = False
    st.session_state.voice_reg_index = 0
    st.session_state.voice_reg_embeddings = []
    st.session_state.voice_reg_sequences = []


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
        profiles.append(npy_path.stem)
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
    """Send message to n8n webhook and return the bot reply."""
    try:
        response = requests.post(
            N8N_WEBHOOK_URL,
            json={"message": user_text},
            timeout=200,
        )
        response.raise_for_status()

        # Empty body — webhook may not be active / test mode not triggered
        raw = response.text.strip()
        if not raw:
            return "⚠️ No response from n8n. Make sure the webhook workflow is **active** and listening in n8n."

        data = response.json()

        # Handle list response: [{"output": "..."}]
        if isinstance(data, list) and len(data) > 0:
            item = data[0]
            return item.get("output") or item.get("outpput") or str(item)

        # Handle dict response: {"output": "..."}
        if isinstance(data, dict):
            return data.get("output") or data.get("outpput") or str(data)

        return str(data)

    except requests.exceptions.Timeout:
        return "⏱️ Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        return "🌐 Could not reach the server. Check your connection."
    except requests.exceptions.HTTPError as e:
        return f"❌ Server error: {e.response.status_code} — {e.response.text[:200]}"
    except ValueError:
        # JSON decode failed — show raw text so we can debug
        return f"⚠️ Unexpected response from n8n: `{response.text[:300]}`"
    
    except Exception as e:
        return f"❌ Error: {e}"

# ── Send a message helper ──────────────────────────────────────────────────────
def send_message(text: str):
    text = text.strip()
    if not text:
        return
    st.session_state.messages.append({"role": "user", "content": text})
    st.session_state.mic_text = ""
    st.session_state.pending_user_text = text  # trigger thinking display on next render

# ── Resolve pending webhook call ───────────────────────────────────────────────
pending = st.session_state.get("pending_user_text", "")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;color:#818cf8;margin-bottom:0'>🤖 JARVIS Chat</h1>"
    "<p style='text-align:center;color:#64748b;margin-top:4px'>Type or speak your message</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── Voice Registration / Verification ────────────────────────────────────────
st.markdown("### 🔐 Voice Match")
with st.container(border=True):
    st.caption("Register with 5 voice samples. Each sample must match the shown text and your voice profile.")

    registered_users = list_registered_users()
    st.caption(f"Registered users: {len(registered_users)}/{MAX_REGISTERED_USERS}")
    if registered_users:
        st.caption("Users: " + ", ".join(registered_users))

    username = st.text_input(
        "User name",
        value=st.session_state.voice_username,
        placeholder="Enter user name",
        key="voice_username_input",
    )
    st.session_state.voice_username = username

    start_registration = st.button("Start Voice Registration", use_container_width=True)
    cancel_registration = st.button("Cancel Registration", use_container_width=True)

    # with info_col:
    #     st.markdown("**Ask-time verification:**")
    #     st.info("Voice is verified only when you ask JARVIS a question.")

    if start_registration:
        normalized_username = normalize_username(username)
        if not normalized_username:
            st.session_state.voice_reg_error = "Please enter a user name before registration."
        elif normalized_username not in registered_users and len(registered_users) >= MAX_REGISTERED_USERS:
            st.session_state.voice_reg_error = (
                f"Only {MAX_REGISTERED_USERS} users are allowed. Use one of existing users or remove one profile."
            )
        else:
            st.session_state.voice_username = normalized_username
            st.session_state.voice_reg_active = True
            st.session_state.voice_reg_index = 0
            st.session_state.voice_reg_embeddings = []
            st.session_state.voice_reg_status = f"Registration started for {normalized_username}."
            st.session_state.voice_reg_error = ""

    if cancel_registration:
        reset_registration_state()
        st.session_state.voice_reg_status = "Registration cancelled."
        st.session_state.voice_reg_error = ""

    if st.session_state.voice_reg_active:
        sample_number = st.session_state.voice_reg_index + 1
        expected_phrase = REGISTRATION_PHRASES[st.session_state.voice_reg_index]
        st.markdown(f"**Sample {sample_number}/5**")
        st.warning(expected_phrase)
        st.caption("Speak this exact sentence within 5 seconds.")

        capture_sample = st.button("🎤 Record Current Sample", use_container_width=True)

        if capture_sample:
            with st.spinner("Listening for up to 5 seconds..."):
                result = capture_speech_and_embedding(timeout_seconds=5, phrase_seconds=5)

            if result.error:
                st.session_state.voice_reg_error = result.error
                st.session_state.voice_reg_status = ""
            else:
                text_ok = prompt_matches(result.spoken_text, expected_phrase)
                if not text_ok:
                    st.session_state.voice_reg_error = (
                        f"Text mismatch. You said: '{result.spoken_text}'. Please repeat the shown sentence."
                    )
                    st.session_state.voice_reg_status = ""
                else:
                    speaker_ok = True
                    if st.session_state.voice_reg_embeddings:
                        scores = [
                            cosine_similarity(ref_embedding, result.embedding)
                            for ref_embedding in st.session_state.voice_reg_embeddings
                        ]
                        similarity = float(np.mean(scores))
                        min_similarity = float(np.min(scores))
                        speaker_ok = min_similarity >= REGISTRATION_MIN_SIMILARITY
                        print(
                            f"[REG] sample={sample_number} mean={similarity:.4f} min={min_similarity:.4f} "
                            f"threshold={REGISTRATION_MIN_SIMILARITY:.4f}"
                        )
                        if not speaker_ok:
                            st.session_state.voice_reg_error = (
                                "Voice mismatch "
                                f"(mean: {similarity:.4f}, min: {min_similarity:.4f}, "
                                f"required min: {REGISTRATION_MIN_SIMILARITY:.4f}). "
                                "Please speak again with the same voice."
                            )
                            st.session_state.voice_reg_status = ""

                    if speaker_ok:
                        if result.mfcc_sequence is None:
                            st.session_state.voice_reg_error = "Could not extract voice features from this sample. Please try again."
                            st.session_state.voice_reg_status = ""
                            st.rerun()
                        st.session_state.voice_reg_embeddings.append(result.embedding)
                        st.session_state.voice_reg_sequences.append(result.mfcc_sequence)
                        st.session_state.voice_reg_index += 1
                        st.session_state.voice_reg_error = ""

                        if st.session_state.voice_reg_index == 5:
                            final_profile = np.stack(st.session_state.voice_reg_embeddings)
                            save_voice_profile(
                                get_user_profile_path(st.session_state.voice_username),
                                final_profile,
                                sample_count=len(st.session_state.voice_reg_embeddings),
                                mfcc_sequences=st.session_state.voice_reg_sequences,
                            )
                            reset_registration_state()
                            st.session_state.voice_reg_status = (
                                f"✅ Registration complete for {st.session_state.voice_username}. Voice profile saved."
                            )
                        else:
                            st.session_state.voice_reg_status = (
                                f"✅ Sample {sample_number} accepted. Please continue with the next sentence."
                            )

    if st.session_state.voice_reg_status:
        st.success(st.session_state.voice_reg_status)
    if st.session_state.voice_reg_error:
        st.error(st.session_state.voice_reg_error)
    if st.session_state.voice_verify_status:
        st.success(st.session_state.voice_verify_status)
    if st.session_state.voice_verify_error:
        st.error(st.session_state.voice_verify_error)

st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────
chat_container = st.container(height=420)
with chat_container:
    if not st.session_state.messages and not pending:
        st.markdown(
            "<p style='text-align:center;color:#475569;margin-top:120px'>"
            "💬 Start a conversation below…</p>",
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f"<div class='chat-wrapper'>"
                    f"<div style='text-align:right'><span class='role-label'>You</span></div>"
                    f"<div class='user-bubble'>{msg['content']}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='chat-wrapper'>"
                    f"<div><span class='role-label'>JARVIS</span></div>"
                    f"<div class='bot-bubble'>{msg['content']}</div></div>",
                    unsafe_allow_html=True,
                )

        # ── Thinking indicator + webhook call ──────────────────────────────
        if pending:
            thinking_slot = st.empty()
            thinking_slot.markdown(
                "<div class='chat-wrapper'>"
                "<div><span class='role-label'>JARVIS</span></div>"
                "<div class='thinking-bubble'>💭 JARVIS is thinking.........</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            # Call webhook while "thinking" bubble is visible
            reply = bot_reply(pending)
            # Clear thinking bubble and save real reply
            thinking_slot.empty()
            
            # Display immediately before speaking
            st.markdown(
                f"<div class='chat-wrapper'>"
                f"<div><span class='role-label'>JARVIS</span></div>"
                f"<div class='bot-bubble'>{reply}</div></div>",
                unsafe_allow_html=True,
            )
            
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.session_state.pending_user_text = ""
            
            # Speak the text
            try:
                text_to_voice(reply)
            except Exception as e:
                print(f"TTS Error: {e}")
                
            st.rerun()

# ── Status messages ───────────────────────────────────────────────────────────
if st.session_state.mic_error:
    st.error(st.session_state.mic_error)
    st.session_state.mic_error = ""
# if st.session_state.mic_text:
#     st.success(f"🎤 Heard: **{st.session_state.mic_text}**")

# ── Input row (form so Enter key submits) ────────────────────────────────────
with st.form(key="chat_form", clear_on_submit=True):
    col_input, col_send, col_mic, col_clear = st.columns([5, 1, 1, 1])

    with col_input:
        user_input = st.text_input(
            "Message",
            value=st.session_state.mic_text,
            placeholder="Type a message and press Enter…",
            label_visibility="collapsed",
            key="chat_input",
        )

    with col_send:
        submitted = st.form_submit_button("Send", use_container_width=True, type="primary")

    with col_mic:
        mic_clicked = st.form_submit_button("🎤", use_container_width=True, help="Click to speak")

    with col_clear:
        clear_clicked = st.form_submit_button("🗑️", use_container_width=True, help="Clear chat")

# Handle form actions outside the form block
if submitted and user_input.strip():
    st.session_state.voice_verify_status = ""
    st.session_state.voice_verify_error = ""
    allowed, reason = can_use_chat(st.session_state.voice_username)
    if not allowed:
        st.session_state.voice_verify_error = f"🔒 {reason}"
    else:
        send_message(user_input)
    st.rerun()

if mic_clicked:
    st.session_state.voice_verify_status = ""
    st.session_state.voice_verify_error = ""
    allowed, reason = can_use_chat(st.session_state.voice_username)
    if not allowed:
        st.session_state.voice_verify_error = f"🔒 {reason}"
    else:
        stored_embedding = load_voice_embedding(get_user_profile_path(st.session_state.voice_username))
        stored_sequences = load_voice_sequences(get_user_profile_path(st.session_state.voice_username))
        stored_meta = load_voice_meta(get_user_profile_path(st.session_state.voice_username)) or {}
        with st.spinner("🎙️ Listening and verifying voice…"):
            import time

            # One-step flow: user speaks their actual message; we verify on that audio.
            verify_result = capture_speech_and_embedding(timeout_seconds=8, phrase_seconds=15)

        if verify_result.error:
            st.session_state.voice_verify_error = verify_result.error
        else:
            required_mean = float(stored_meta.get("verify_mean_threshold", VOICE_SIMILARITY_MEAN_THRESHOLD))
            required_max = float(stored_meta.get("verify_max_threshold", VOICE_SIMILARITY_MAX_THRESHOLD))
            passed, mean_score, max_score = evaluate_voice_match_with_thresholds(
                stored_embedding,
                verify_result.embedding,
                required_mean=required_mean,
                required_max=required_max,
            )
            dtw_ok = True
            dtw_score = None
            if stored_sequences is None:
                dtw_ok = False
                st.session_state.voice_verify_error = (
                    "Your voice profile is missing sequence data (older registration). "
                    "Please re-register your voice for stronger verification."
                )
                st.rerun()
            else:
                try:
                    # DTW is only reliable when the *spoken content* is close to one of the
                    # enrollment phrases. For arbitrary questions, we skip DTW and use cosine only.
                    phrase_idx, phrase_ratio = best_prompt_match(verify_result.spoken_text, REGISTRATION_PHRASES)
                    use_dtw = phrase_idx >= 0 and phrase_ratio >= VOICE_CHALLENGE_MIN_RATIO

                    if use_dtw:
                        if verify_result.mfcc_sequence is None:
                            raise ValueError("Could not extract MFCC sequence from captured audio")
                        ref_seq = stored_sequences[phrase_idx]
                        dtw_score = float(dtw_distance(ref_seq, verify_result.mfcc_sequence))
                        dtw_ok = dtw_score <= VOICE_DTW_DISTANCE_THRESHOLD
                    else:
                        dtw_ok = True
                except Exception as e:
                    dtw_ok = False
                    st.session_state.voice_verify_error = f"DTW verification error: {e}"
                    st.rerun()

            dtw_str = "n/a" if dtw_score is None else f"{dtw_score:.4f}"
            print(
                "[VERIFY:MIC] "
                f"mean={mean_score:.4f} max={max_score:.4f} dtw={dtw_str} "
                f"required_mean={required_mean:.4f} "
                f"required_max={required_max:.4f}"
            )
            if not passed or not dtw_ok:
                st.session_state.voice_verify_error = (
                    "🔒 Voice not matched. Message was NOT sent. "
                    f"(mean: {mean_score:.4f}, max: {max_score:.4f}, dtw: {dtw_str}, "
                    f"required mean/max/dtw: {required_mean:.4f}/{required_max:.4f}/{VOICE_DTW_DISTANCE_THRESHOLD:.4f})."
                )
            else:
                st.session_state.voice_verify_status = (
                    "✅ Voice verified "
                    f"(mean: {mean_score:.4f}, max: {max_score:.4f}, dtw: {dtw_str})."
                )
                st.session_state.voice_verified_until = time.time() + 60.0
                # Send the user's actual spoken question
                send_message(verify_result.spoken_text)
    st.rerun()

if clear_clicked:
    st.session_state.messages = []
    st.session_state.mic_text = ""
    st.session_state.mic_error = ""
    st.session_state.pending_user_text = ""
    st.rerun()

st.caption("💡 Supports up to 3 registered users · Select/enter registered user name first · Typed Send sends text directly · 🎤 verifies live speech every time · 🗑️ to clear chat")

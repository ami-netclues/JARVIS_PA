import streamlit as st
import speech_recognition as sr
import requests
from simple_tts import text_to_voice

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

# ── n8n Webhook ───────────────────────────────────────────────────────────────
N8N_WEBHOOK_URL = "https://v1netclues.app.n8n.cloud/webhook/2a842b2d-2345-4fc1-a399-1838ac2c1da8"

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

# ── Speech recognition (synchronous, runs in main thread via button) ───────────
def listen_from_mic() -> tuple[str, str]:
    """Returns (recognized_text, error_message)."""
    r = sr.Recognizer()
    r.pause_threshold = 1.0
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source, timeout=8, phrase_time_limit=15)
        text = r.recognize_google(audio)
        return text, ""
    except sr.WaitTimeoutError:
        return "", "⏱️ No speech detected. Please try again."
    except sr.UnknownValueError:
        return "", "🔇 Could not understand audio. Please speak clearly."
    except sr.RequestError as e:
        return "", f"🌐 Speech service error: {e}"
    except Exception as e:
        return "", f"❌ Error: {e}"

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
    send_message(user_input)
    st.rerun()

if mic_clicked:
    with st.spinner("🎙️ Listening…"):
        spoken, err = listen_from_mic()
    if err:
        st.session_state.mic_error = err
    elif spoken:
        send_message(spoken)
    st.rerun()

if clear_clicked:
    st.session_state.messages = []
    st.session_state.mic_text = ""
    st.session_state.mic_error = ""
    st.session_state.pending_user_text = ""
    st.rerun()

st.caption("💡 Press **Enter** or **Send** to submit · 🎤 to use microphone · 🗑️ to clear chat")

import streamlit as st
import time
import re
from simple_tts import text_to_voice, text_to_voice_async, stop_speaking, is_speaking
from Voice_match.voice_auth import (
    capture_speech_and_embedding,
    is_voice_detected,
    load_voice_embedding,
    load_voice_sequences,
    load_voice_meta,
    dtw_distance,
    best_prompt_match,
)
from shared_utils import (
    init_session_state,
    apply_custom_css,
    normalize_username,
    get_user_profile_path,
    can_use_chat,
    bot_reply,
    evaluate_voice_match_with_thresholds,
    VOICE_SIMILARITY_MEAN_THRESHOLD,
    VOICE_SIMILARITY_MAX_THRESHOLD,
    REGISTRATION_PHRASES,
    VOICE_CHALLENGE_MIN_RATIO,
    VOICE_DTW_DISTANCE_THRESHOLD,
)

st.set_page_config(page_title="JARVIS Chat", page_icon="🤖", layout="centered")
init_session_state()
apply_custom_css()

# ── Send a message helper ──────────────────────────────────────────────────────
def send_message(text: str):
    text = text.strip()
    if not text:
        return
    st.session_state.messages.append({"role": "user", "content": text})
    st.session_state.mic_text = ""
    st.session_state.pending_user_text = text
    st.session_state.interaction_status = "" # Clear status when message sent

def verify_and_send_from_result(verify_result) -> bool:
    username = st.session_state.voice_username
    stored_embedding = load_voice_embedding(get_user_profile_path(username))
    stored_sequences = load_voice_sequences(get_user_profile_path(username))
    stored_meta = load_voice_meta(get_user_profile_path(username)) or {}

    if verify_result.error:
        st.session_state.voice_verify_error = verify_result.error
        st.session_state.interaction_status = f"❌ {verify_result.error}"
        return False

    required_mean = float(stored_meta.get("verify_mean_threshold", VOICE_SIMILARITY_MEAN_THRESHOLD))
    required_max = float(stored_meta.get("verify_max_threshold", VOICE_SIMILARITY_MAX_THRESHOLD))

    word_count = len(verify_result.spoken_text.split()) if verify_result.spoken_text else 0
    if word_count == 3:
        discount = 0.90
        required_mean *= discount
        required_max *= discount
    elif word_count <= 2:
        required_mean = max(required_mean, 0.66)
        required_max = max(required_max, 0.70)

    passed, mean_score, max_score = evaluate_voice_match_with_thresholds(
        stored_embedding,
        verify_result.embedding,
        required_mean=required_mean,
        required_max=required_max,
    )
    dtw_ok = True
    dtw_score = None
    if stored_sequences is None:
        st.session_state.voice_verify_error = "Missing sequence data. Re-register."
        st.session_state.interaction_status = "❌ Voice profile error."
        return False
    try:
        phrase_idx, phrase_ratio = best_prompt_match(verify_result.spoken_text, REGISTRATION_PHRASES)
        use_dtw = phrase_idx >= 0 and phrase_ratio >= VOICE_CHALLENGE_MIN_RATIO
        if use_dtw:
            if verify_result.mfcc_sequence is not None:
                ref_seq = stored_sequences[phrase_idx]
                dtw_score = float(dtw_distance(ref_seq, verify_result.mfcc_sequence))
                dtw_ok = dtw_score <= VOICE_DTW_DISTANCE_THRESHOLD
    except Exception as e:
        st.session_state.interaction_status = f"❌ DTW error."
        return False

    if not passed or not dtw_ok:
        msg = "You are not an authenticated person."
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.session_state.interaction_status = "🔒 Authentication failed."
        text_to_voice(msg)
        return False

    st.session_state.voice_verify_status = "✅ Voice verified"
    st.session_state.interaction_status = "✅ Verified"
    st.session_state.voice_verified_until = time.time() + 60.0
    send_message(verify_result.spoken_text)
    return True

def verify_and_send_captured_message() -> bool:
    st.session_state.interaction_status = "🎙️ Listening and verifying..."
    with st.spinner("🎙️ Listening..."):
        verify_result = capture_speech_and_embedding(timeout_seconds=8, phrase_seconds=None)
    return verify_and_send_from_result(verify_result)

def listen_for_barge_in_while_speaking(current_reply: str) -> bool:
    deadline = time.time() + 1.0
    while not is_speaking() and time.time() < deadline:
        time.sleep(0.03)
    time.sleep(0.25)
    while is_speaking():
        if is_voice_detected(timeout_seconds=0.25):
            st.session_state.interaction_status = "🎤 Interrupt detected..."
            confirm_result = capture_speech_and_embedding(timeout_seconds=1, phrase_seconds=2)
            spoken = (confirm_result.spoken_text or "").strip()
            if not spoken or len(spoken.split()) < 2:
                continue
            stop_speaking()
            st.session_state.interaction_status = "🎤 Listening..."
            return verify_and_send_captured_message()
    return False

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;color:#818cf8;margin-bottom:0'>🤖 JARVIS Chat</h1>"
    "<p style='text-align:center;color:#64748b;margin-top:4px'>Speak to JARVIS</p>",
    unsafe_allow_html=True,
)

# ── User Selection / Settings ─────────────────────────────────────────────────
with st.sidebar:
    st.title("Settings")
    username = st.text_input("Username", value=st.session_state.voice_username)
    st.session_state.voice_username = username
    # Redundant status displays removed from sidebar

# ── Chat history ──────────────────────────────────────────────────────────────
chat_container = st.container(height=420)
with chat_container:
    for msg in st.session_state.messages:
        role = "You" if msg["role"] == "user" else "JARVIS"
        css_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
        align = "right" if msg["role"] == "user" else "left"
        st.markdown(
            f"<div class='chat-wrapper'>"
            f"<div style='text-align:{align}'><span class='role-label'>{role}</span></div>"
            f"<div class='{css_class}'>{msg['content']}</div></div>",
            unsafe_allow_html=True,
        )

    # Status Placeholder inside chat container
    status_placeholder = st.empty()
    if st.session_state.interaction_status:
        status_placeholder.markdown(
            f"<div style='text-align:center; color:#94a3b8; font-style:italic; margin:10px 0;'>"
            f"{st.session_state.interaction_status}</div>",
            unsafe_allow_html=True
        )

    pending = st.session_state.get("pending_user_text", "")
    if pending:
        thinking_slot = st.empty()
        thinking_slot.markdown("<div class='thinking-bubble'>💭 JARVIS is thinking...</div>", unsafe_allow_html=True)
        reply = bot_reply(pending)
        thinking_slot.empty()
        st.markdown(f"<div class='bot-bubble'>{reply}</div>", unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.session_state.pending_user_text = ""
        try:
            if st.session_state.handsfree_active:
                st.session_state.interaction_status = "🔊 JARVIS is speaking..."
                text_to_voice_async(reply)
                if not listen_for_barge_in_while_speaking(reply):
                    st.session_state.interaction_status = "👂 Waiting for you..."
            else:
                text_to_voice(reply)
                st.session_state.interaction_status = ""
        except Exception:
            pass
        st.rerun()

# ── Interaction ───────────────────────────────────────────────────────────────
with st.form(key="chat_form", clear_on_submit=True):
    col_input, col_send, col_mic, col_clear = st.columns([5, 1, 1, 1])
    with col_input:
        user_input = st.text_input("Message", value=st.session_state.mic_text, label_visibility="collapsed")
    with col_send:
        submitted = st.form_submit_button("Send", use_container_width=True, type="primary")
    with col_mic:
        mic_clicked = st.form_submit_button("🎤", use_container_width=True)
    with col_clear:
        clear_clicked = st.form_submit_button("🗑️", use_container_width=True)

if submitted and user_input.strip():
    allowed, reason = can_use_chat(st.session_state.voice_username)
    if not allowed:
        st.error(reason)
    else:
        send_message(user_input)
        st.rerun()

if mic_clicked:
    if st.session_state.handsfree_active:
        st.session_state.handsfree_active = False
        stop_speaking()
    else:
        allowed, reason = can_use_chat(st.session_state.voice_username)
        if not allowed:
            st.error(reason)
        else:
            if verify_and_send_captured_message():
                st.session_state.handsfree_active = True
    st.rerun()

if clear_clicked:
    st.session_state.messages = []
    st.session_state.pending_user_text = ""
    st.rerun()

if st.session_state.handsfree_active and not st.session_state.pending_user_text:
    verify_and_send_captured_message()
    st.rerun()

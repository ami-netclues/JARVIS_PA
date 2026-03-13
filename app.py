import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
from shared_utils import init_session_state, apply_custom_css, list_registered_users

st.set_page_config(
    page_title="JARVIS PA",
    page_icon="🤖",
    layout="wide",
)

init_session_state()
apply_custom_css()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='text-align:center; padding: 50px 0;'>
        <h1 style='color:#818cf8; font-size: 3rem; margin-bottom: 10px;'>🤖 Welcome to JARVIS</h1>
        <p style='color:#94a3b8; font-size: 1.2rem;'>Your Advanced Personal Assistant with Voice Authentication</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# col1, col2 = st.columns([1, 1])

# with col1:
st.markdown("### 📢 Getting Started")
st.write(
    """
    JARVIS is a next-generation personal assistant that uses voice biometrics to ensure 
    that only you can access your personal data and commands.

    **Follow these steps to begin:**
    1. **Register your voice:** Go to the **Voice Registration** page in the sidebar. 
        You will need to provide 5 voice samples using specific phrases.
    2. **Start Chatting:** Once registered, head over to the **Chat** page. 
        You can type your messages or use the microphone for a hands-free experience.
    """
)

if st.button("Go to Voice Registration", type="secondary", use_container_width=False):
    st.switch_page("pages/1_Voice_Registration.py")

st.markdown("### ⚡ Live Streaming Chat (Beta)")
st.write("Experience low-latency, real-time voice interaction powered by FastAPI and WebSockets.")
if st.button("Open Streaming Chat", type="secondary", use_container_width=False):
    st.markdown(f'<meta http-equiv="refresh" content="0; url=http://localhost:8000/static/streaming_client.html">', unsafe_allow_html=True)

st.divider()

# with col2:
st.markdown("### 🔐 Security & Features")
registered_users = list_registered_users()
if registered_users:
    st.success(f"Number of registered users: {len(registered_users)}")
    st.info("Users: " + ", ".join(registered_users))
else:
    st.warning("No users registered yet. Start by creating a voice profile!")

st.markdown(
    """
    - **Voice Matching:** Advanced embedding-based speaker verification.
    - **Barge-in Support:** Interrupt JARVIS while he's speaking.
    - **Hands-free Mode:** Continuous listening for a natural conversation flow.
    - **n8n Integration:** Connected to powerful automation workflows.
    """
)

st.divider()
st.caption("Powered by Streamlit and AI biometric security.")

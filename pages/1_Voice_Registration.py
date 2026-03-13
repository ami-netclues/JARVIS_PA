import streamlit as st
import numpy as np
from Voice_match.voice_auth import (
    capture_speech_and_embedding,
    cosine_similarity,
    save_voice_profile,
    prompt_matches,
)
from shared_utils import (
    init_session_state,
    apply_custom_css,
    normalize_username,
    get_user_profile_path,
    list_registered_users,
    MAX_REGISTERED_USERS,
    REGISTRATION_PHRASES,
    REGISTRATION_MIN_SIMILARITY,
)

st.set_page_config(page_title="Voice Registration", page_icon="🔐", layout="centered")
init_session_state()
apply_custom_css()

def reset_registration_state():
    st.session_state.voice_reg_active = False
    st.session_state.voice_reg_index = 0
    st.session_state.voice_reg_embeddings = []
    st.session_state.voice_reg_sequences = []

st.markdown(
    "<h1 style='text-align:center;color:#818cf8;margin-bottom:0'>🔐 Voice Registration</h1>"
    "<p style='text-align:center;color:#64748b;margin-top:4px'>Register your voice profile to use JARVIS</p>",
    unsafe_allow_html=True,
)
st.divider()

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

    col1, col2 = st.columns(2)
    with col1:
        start_registration = st.button("Start Voice Registration", use_container_width=True)
    with col2:
        cancel_registration = st.button("Cancel Registration", use_container_width=True)

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

st.info("💡 You can navigate to the Chat page once registered.")


if st.button("Go to Chat page", type="secondary", use_container_width=False):
    st.switch_page("pages/2_Chat.py")




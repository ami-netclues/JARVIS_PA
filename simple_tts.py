import threading
import pyttsx3


_state_lock = threading.Lock()
_engine = None
_speak_thread = None
_is_speaking = False


def _create_engine():
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 1.0)
    voices = engine.getProperty("voices")
    if voices:
        preferred = voices[1].id if len(voices) > 1 else voices[0].id
        engine.setProperty("voice", preferred)
    return engine


def _speak_worker(text: str):
    global _is_speaking, _engine
    engine = None
    try:
        engine = _create_engine()
        with _state_lock:
            _engine = engine
            _is_speaking = True
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")
    finally:
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass
        with _state_lock:
            _engine = None
            _is_speaking = False


def text_to_voice(text):
    thread = text_to_voice_async(text)
    if thread is not None:
        thread.join()


def text_to_voice_async(text: str):
    global _speak_thread
    text = (text or "").strip()
    if not text:
        return None
    stop_speaking()
    # Wait for the previous thread to fully exit before creating a new engine.
    # This avoids the race where the old finally-block sets _engine = None
    # right after the new thread already stored its engine reference.
    if _speak_thread is not None and _speak_thread.is_alive():
        _speak_thread.join(timeout=3.0)
    _speak_thread = threading.Thread(target=_speak_worker, args=(text,), daemon=True)
    _speak_thread.start()
    return _speak_thread


def stop_speaking():
    global _is_speaking
    engine = None
    with _state_lock:
        engine = _engine
    if engine is not None:
        try:
            engine.stop()
        except Exception:
            pass
    with _state_lock:
        _is_speaking = False


def is_speaking() -> bool:
    return bool(_is_speaking)

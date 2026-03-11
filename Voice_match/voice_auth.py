from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

# import librosa
import numpy as np
import speech_recognition as sr

try:
    # A proper speaker embedding model with VAD-based preprocessing.
    # This is significantly more robust than handcrafted MFCC features.
    from resemblyzer import VoiceEncoder, preprocess_wav  # type: ignore

    _RESEMBLYZER_AVAILABLE = True
    _VOICE_ENCODER: VoiceEncoder | None = None
except Exception:
    _RESEMBLYZER_AVAILABLE = False
    _VOICE_ENCODER = None

import librosa
import librosa.sequence


@dataclass
class CaptureResult:
    spoken_text: str
    embedding: np.ndarray | None
    mfcc_sequence: np.ndarray | None
    error: str


def normalize_text(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in value)
    return " ".join(cleaned.split())


def prompt_matches(spoken_text: str, expected_text: str, min_ratio: float = 0.75) -> bool:
    spoken = normalize_text(spoken_text)
    expected = normalize_text(expected_text)
    if not spoken or not expected:
        return False
    if spoken == expected:
        return True
    ratio = SequenceMatcher(None, spoken, expected).ratio()
    return ratio >= min_ratio


def best_prompt_match(spoken_text: str, expected_texts: list[str]) -> tuple[int, float]:
    """
    Return (best_index, best_ratio) using the same normalization used by prompt matching.
    If the spoken text is empty, returns (-1, 0.0).
    """
    spoken = normalize_text(spoken_text)
    if not spoken or not expected_texts:
        return -1, 0.0

    best_idx = -1
    best_ratio = 0.0
    for idx, expected in enumerate(expected_texts):
        expected_norm = normalize_text(expected)
        if not expected_norm:
            continue
        ratio = 1.0 if spoken == expected_norm else SequenceMatcher(None, spoken, expected_norm).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = idx
    return best_idx, float(best_ratio)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """
    Normalized DTW distance between two MFCC(+delta) sequences.
    Lower is better (more similar).
    """
    if seq_a.size == 0 or seq_b.size == 0:
        return float("inf")

    # librosa's DTW expects (features, frames). We'll compute a cosine cost on frame vectors.
    # Transpose to (frames, features) for a stable per-frame metric, then transpose back for DTW.
    x = seq_a.T.astype(np.float32, copy=False)
    y = seq_b.T.astype(np.float32, copy=False)

    # Compute DTW cost matrix with cosine metric.
    D, wp = librosa.sequence.dtw(X=x.T, Y=y.T, metric="cosine")
    if D.size == 0 or len(wp) == 0:
        return float("inf")

    final_cost = float(D[-1, -1])
    path_len = float(len(wp))
    return final_cost / max(path_len, 1.0)

def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32, copy=False)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def _load_and_trim_speech(wav_path: Path) -> tuple[np.ndarray, int]:
    signal, sample_rate = librosa.load(wav_path, sr=16000, mono=True)
    if signal.size == 0:
        raise ValueError("Empty audio signal")

    signal, _ = librosa.effects.trim(signal, top_db=30)
    if signal.size == 0:
        raise ValueError("Empty audio after trimming")

    intervals = librosa.effects.split(signal, top_db=30)
    if len(intervals) > 0:
        chunks: list[np.ndarray] = []
        total = 0
        max_samples = int(sample_rate * 6.0)
        for start, end in intervals:
            chunk = signal[start:end]
            if chunk.size == 0:
                continue
            remaining = max_samples - total
            if remaining <= 0:
                break
            if chunk.size > remaining:
                chunk = chunk[:remaining]
            chunks.append(chunk)
            total += chunk.size
        if chunks:
            signal = np.concatenate(chunks)

    if signal.size < int(sample_rate * 0.6):
        raise ValueError("Audio too short for reliable voice match (need ~0.6s+ of speech)")

    return signal.astype(np.float32, copy=False), sample_rate


def build_mfcc_sequence(wav_path: Path) -> np.ndarray:
    """
    Build a time-series feature matrix for DTW matching.
    Output shape: (n_features, n_frames).
    """
    signal, sample_rate = _load_and_trim_speech(wav_path)

    n_fft = 512
    hop_length = 160
    win_length = 400

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=20,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    ).astype(np.float32)
    d1 = librosa.feature.delta(mfcc).astype(np.float32)
    d2 = librosa.feature.delta(mfcc, order=2).astype(np.float32)
    feat = np.vstack([mfcc, d1, d2]).astype(np.float32)  # (60, T)

    feat -= feat.mean(axis=1, keepdims=True)
    feat /= feat.std(axis=1, keepdims=True) + 1e-6
    return feat


def _build_embedding_resemblyzer(wav_path: Path) -> np.ndarray:
    global _VOICE_ENCODER
    if _VOICE_ENCODER is None:
        _VOICE_ENCODER = VoiceEncoder()

    wav = preprocess_wav(wav_path)  # includes VAD + resample
    if wav is None or len(wav) == 0:
        raise ValueError("Empty audio after preprocessing")

    emb = _VOICE_ENCODER.embed_utterance(wav).astype(np.float32)
    return _l2_normalize(emb)


def _build_embedding_fallback(wav_path: Path) -> np.ndarray:
    """
    Fallback embedding (less reliable) when resemblyzer isn't installed.
    Kept only so the app can still run, but results will be weaker.
    """
    signal, sample_rate = _load_and_trim_speech(wav_path)

    # Frame settings ~25ms window, 10ms hop at 16kHz.
    n_fft = 512
    hop_length = 160
    win_length = 400

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=20,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    feat = np.vstack([mfcc, d1, d2]).astype(np.float32)  # (60, T)

    # Cepstral mean/variance normalization across time.
    feat -= feat.mean(axis=1, keepdims=True)
    feat_std = feat.std(axis=1, keepdims=True) + 1e-6
    feat /= feat_std

    # Stats pooling: mean/std + robust percentiles per coefficient.
    p25 = np.percentile(feat, 25, axis=1)
    p50 = np.percentile(feat, 50, axis=1)
    p75 = np.percentile(feat, 75, axis=1)
    pooled = np.concatenate(
        [
            feat.mean(axis=1),
            feat.std(axis=1),
            p25,
            p50,
            p75,
        ]
    )
    # Pitch features were removed because they were unstable across devices/content and
    # increased false rejects for "speak any question" verification.
    return _l2_normalize(pooled.astype(np.float32))


def _pairwise_cosine_stats(embeddings: np.ndarray) -> tuple[float, float]:
    """
    Compute mean/std of pairwise cosine similarities between enrollment samples.
    embeddings: (N, D) or (D,)
    """
    if embeddings.ndim == 1:
        return 1.0, 0.0
    n = int(embeddings.shape[0])
    if n < 2:
        return 1.0, 0.0
    sims: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(cosine_similarity(embeddings[i], embeddings[j]))
    if not sims:
        return 1.0, 0.0
    return float(np.mean(sims)), float(np.std(sims))


def recommend_thresholds(embeddings: np.ndarray) -> dict[str, float]:
    """
    Adaptive per-user thresholds derived from enrollment consistency.
    Returns dict with verify_max_threshold and verify_mean_threshold.
    """
    mean, std = _pairwise_cosine_stats(embeddings)
    # Be permissive enough for noisy mics but still tied to enrollment tightness.
    base = mean - 3.0 * std
    verify_max = float(np.clip(base, 0.55, 0.90))
    verify_mean = float(np.clip(base - 0.03, 0.52, verify_max))
    return {
        "enroll_pairwise_mean": float(mean),
        "enroll_pairwise_std": float(std),
        "verify_max_threshold": verify_max,
        "verify_mean_threshold": verify_mean,
    }


def build_speaker_embedding(wav_path: Path) -> np.ndarray:
    if _RESEMBLYZER_AVAILABLE:
        return _build_embedding_resemblyzer(wav_path)
    return _build_embedding_fallback(wav_path)


def capture_speech_and_embedding(timeout_seconds: int = 5, phrase_seconds: int = 5) -> CaptureResult:
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 0.8

    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.4)
            audio = recognizer.listen(source, timeout=timeout_seconds, phrase_time_limit=phrase_seconds)

        spoken_text = recognizer.recognize_google(audio)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio.get_wav_data(convert_rate=16000, convert_width=2))
            wav_path = Path(temp_file.name)

        embedding = build_speaker_embedding(wav_path)
        mfcc_sequence = build_mfcc_sequence(wav_path)
        wav_path.unlink(missing_ok=True)

        return CaptureResult(spoken_text=spoken_text, embedding=embedding, mfcc_sequence=mfcc_sequence, error="")
    except sr.WaitTimeoutError:
        return CaptureResult(spoken_text="", embedding=None, mfcc_sequence=None, error="⏱️ No speech detected in 5 seconds.")
    except sr.UnknownValueError:
        return CaptureResult(spoken_text="", embedding=None, mfcc_sequence=None, error="🔇 Could not understand speech. Try again clearly.")
    except sr.RequestError as err:
        return CaptureResult(spoken_text="", embedding=None, mfcc_sequence=None, error=f"🌐 Speech service error: {err}")
    except Exception as err:
        return CaptureResult(spoken_text="", embedding=None, mfcc_sequence=None, error=f"❌ Voice capture error: {err}")


def save_voice_profile(
    profile_path: Path,
    embedding: np.ndarray,
    sample_count: int,
    mfcc_sequences: list[np.ndarray] | None = None,
) -> None:
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(profile_path.with_suffix(".npy"), embedding)
    if mfcc_sequences:
        # IMPORTANT: MFCC sequences are variable-length (60, T). Force a 1D object array so
        # numpy doesn't try to broadcast them into a numeric 2D/3D array.
        seq_obj = np.empty((len(mfcc_sequences),), dtype=object)
        for i, seq in enumerate(mfcc_sequences):
            if seq is None:
                raise ValueError("Missing MFCC sequence while saving voice profile")
            seq_obj[i] = np.asarray(seq, dtype=np.float32)
        np.save(profile_path.with_suffix(".seq.npy"), seq_obj, allow_pickle=True)
    meta: dict[str, float | int] = {"sample_count": int(sample_count)}
    try:
        meta.update(recommend_thresholds(embedding))
    except Exception:
        pass
    profile_path.with_suffix(".json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )


def load_voice_embedding(profile_path: Path) -> np.ndarray | None:
    embedding_path = profile_path.with_suffix(".npy")
    if not embedding_path.exists():
        return None
    return np.load(embedding_path)


def load_voice_meta(profile_path: Path) -> dict | None:
    meta_path = profile_path.with_suffix(".json")
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_voice_sequences(profile_path: Path) -> list[np.ndarray] | None:
    seq_path = profile_path.with_suffix(".seq.npy")
    if not seq_path.exists():
        return None
    seq_obj = np.load(seq_path, allow_pickle=True)
    return [np.array(x, dtype=np.float32) for x in seq_obj.tolist()]

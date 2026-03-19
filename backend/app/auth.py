import os
import json
import shutil
import numpy as np
import librosa
import av
from typing import List, Dict, Tuple
from resemblyzer import VoiceEncoder
from scipy.spatial.distance import cosine

class VoiceAuthManager:
    def __init__(self, base_path: str):
        self.profiles_path = os.path.join(base_path, "profiles.json")
        self.signatures_dir = os.path.join(base_path, "signatures")
        
        if not os.path.exists(self.signatures_dir):
            os.makedirs(self.signatures_dir)
            
        self.profiles = self._load_profiles()
        
        # Load Resemblyzer VoiceEncoder (Singleton/Global)
        self.encoder = VoiceEncoder()

    def _load_profiles(self) -> Dict:
        if os.path.exists(self.profiles_path):
            with open(self.profiles_path, "r") as f:
                return json.load(f)
        return {"profiles": []}

    def _save_profiles(self):
        with open(self.profiles_path, "w") as f:
            json.dump(self.profiles, f, indent=4)

    def get_profiles(self) -> List[str]:
        return [p["name"] for p in self.profiles["profiles"]]

    def _load_audio_robust(self, path: str) -> np.ndarray:
        """Load audio using PyAV to support any container/codec without FFmpeg in PATH."""
        try:
            container = av.open(path)
            stream = container.streams.audio[0]
            
            # Resample and convert to mono using PyAV's internal resampler
            resampler = av.AudioResampler(
                format='fltp',
                layout='mono',
                rate=16000,
            )
            
            frames = []
            for frame in container.decode(stream):
                resampled_frames = resampler.resample(frame)
                for resampled_frame in resampled_frames:
                    frames.append(resampled_frame.to_ndarray())
            
            if not frames:
                raise ValueError("No audio frames decoded")
                
            audio = np.concatenate(frames, axis=1).reshape(-1)
            container.close()
            return audio.astype(np.float32)
            
        except Exception as e:
            # Fallback to librosa/soundfile if PyAV fails
            print(f"PyAV decoding failed, trying librosa fallback: {e}")
            wav, _ = librosa.load(path, sr=16000, mono=True)
            return wav

    def register_profile(self, name: str, audio_path: str) -> bool:
        if len(self.profiles["profiles"]) >= 10:
            return False
        
        for p in self.profiles["profiles"]:
            if p["name"] == name:
                return False

        sig_filename = f"{name.replace(' ', '_').lower()}.wav"
        sig_path = os.path.join(self.signatures_dir, sig_filename)
        
        try:
            # Use PyAV for robust decoding
            wav = self._load_audio_robust(audio_path)
            
            # Save as standard PCM WAV for reference
            import soundfile as sf
            sf.write(sig_path, wav, 16000)
        except Exception as e:
            import traceback
            print(f"Registration error saving audio: {e}\n{traceback.format_exc()}")
            return False
        
        self.profiles["profiles"].append({
            "name": name,
            "signature_file": sig_filename
        })
        self._save_profiles()
        return True

    def delete_profile(self, name: str) -> bool:
        profile_idx = next((i for i, p in enumerate(self.profiles["profiles"]) if p["name"] == name), -1)
        if profile_idx == -1:
            return False

        profile = self.profiles["profiles"].pop(profile_idx)
        sig_path = os.path.join(self.signatures_dir, profile["signature_file"])
        if os.path.exists(sig_path):
            try:
                os.remove(sig_path)
            except Exception as e:
                print(f"Error removing signature file: {e}")

        self._save_profiles()
        return True

    def verify_voice(self, name: str, new_audio_path: str, threshold: float = 0.8) -> Tuple[bool, float]:
        profile = next((p for p in self.profiles["profiles"] if p["name"] == name), None)
        if not profile:
            return False, 0.0

        stored_sig_path = os.path.join(self.signatures_dir, profile["signature_file"])
        if not os.path.exists(stored_sig_path):
            return False, 0.0

        try:
            # Robust loading for both verify sample and stored reference
            wav1 = self._load_audio_robust(new_audio_path)
            wav2 = self._load_audio_robust(stored_sig_path)

            # Generate embeddings
            emb1 = self.encoder.embed_utterance(wav1)
            emb2 = self.encoder.embed_utterance(wav2)

            # Compute similarity
            similarity = 1 - cosine(emb1, emb2)
            
            is_authenticated = bool(similarity > threshold)
            return is_authenticated, float(similarity)
        except Exception as e:
            print(f"Verification error: {e}")
            return False, 0.0

# Global instance
auth_manager = VoiceAuthManager(os.path.dirname(os.path.dirname(__file__)))

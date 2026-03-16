import os
import uuid
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import torch
import sounddevice as sd
import numpy as np
import threading
import queue
import warnings
import pygame
import soundfile as sf
import pyttsx3
import requests
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore")

# Your verified n8n Webhook URL
N8N_WEBHOOK_URL = "https://v1netclues.app.n8n.cloud/webhook/2a842b2d-2345-4fc1-a399-1838ac2c1da8"

class VoiceAuthenticator:
    def __init__(self, threshold=0.88):
        print("Loading WavLM Voice Authentication Model...")
        self.model_name = "microsoft/wavlm-base-plus-sv"
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        self.model = WavLMForXVector.from_pretrained(self.model_name)
        self.threshold = threshold
        self.master_embedding = None

    def get_embedding(self, audio_data, sample_rate=16000):
        if len(audio_data) < int(sample_rate * 0.8):
            return None
        waveform = torch.tensor(audio_data).float().unsqueeze(0)
        inputs = self.feature_extractor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            embeddings = self.model(**inputs).embeddings
        return torch.nn.functional.normalize(embeddings, dim=-1)

    def enroll(self, clean_audio_data, sample_rate=16000):
        print("Analyzing enrollment audio and building master profile...")
        chunk_size = sample_rate * 1 
        embeddings = []
        for i in range(0, len(clean_audio_data), chunk_size):
            chunk = clean_audio_data[i:i+chunk_size]
            emb = self.get_embedding(chunk, sample_rate)
            if emb is not None:
                embeddings.append(emb)
        if not embeddings:
            print("Failed to enroll: Not enough clean speech detected.")
            return False
        stacked_embeddings = torch.cat(embeddings, dim=0)
        self.master_embedding = torch.mean(stacked_embeddings, dim=0, keepdim=True)
        self.master_embedding = torch.nn.functional.normalize(self.master_embedding, dim=-1)
        print(f"Master voice successfully enrolled using {len(embeddings)} high-quality samples.")
        return True

    def verify(self, clean_audio_data):
        if self.master_embedding is None:
            return False
        current_embedding = self.get_embedding(clean_audio_data)
        if current_embedding is None:
            return False
        cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        similarity = cosine_sim(self.master_embedding, current_embedding).item()
        print(f"Auth Score: {similarity:.3f} (Required: {self.threshold})")
        return similarity >= self.threshold

class VoiceAssistant:
    def __init__(self):
        self.fs = 16000
        self.chunk_size = 512
        pygame.mixer.init() 
        self.auth = VoiceAuthenticator(threshold=0.88)
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        self.get_speech_timestamps = utils[0]
        self.stt_model = WhisperModel("base", device="cpu", compute_type="int8")
        self.audio_queue = queue.Queue()
        self.last_audio_file = None

    def extract_pure_speech(self, audio_data):
        tensor_audio = torch.tensor(audio_data).float()
        timestamps = self.get_speech_timestamps(tensor_audio, self.vad_model, sampling_rate=self.fs)
        if not timestamps:
            return np.array([])
        speech_chunks = [audio_data[ts['start']:ts['end']] for ts in timestamps]
        return np.concatenate(speech_chunks)

    def record_enrollment(self, duration=6):
        print("\n--- ENROLLMENT ---")
        print(f"Recording for {duration} seconds. Please speak naturally.")
        recording = sd.rec(int(duration * self.fs), samplerate=self.fs, channels=1, dtype='float32')
        sd.wait()
        clean_audio = self.extract_pure_speech(recording.flatten())
        if self.auth.enroll(clean_audio):
            print("ENROLLMENT SUCCESSFUL\n")
        else:
            print("ENROLLMENT FAILED\n")

    def kill_speech(self):
        """Instantly stops audio and deletes the temp file to prevent locks."""
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            try:
                pygame.mixer.music.unload()
            except AttributeError:
                pass
        
        if self.last_audio_file and os.path.exists(self.last_audio_file):
            try:
                os.remove(self.last_audio_file)
            except OSError:
                pass

    def tts_and_play(self, text):
        print(f"JARVIS: {text}")
        self.kill_speech() # Ensure absolute silence before speaking
        
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 200)
            
            # Use unique filenames to completely avoid file lock crashes
            self.last_audio_file = f"reply_{uuid.uuid4().hex[:6]}.wav"
            engine.save_to_file(text, self.last_audio_file)
            engine.runAndWait()
            
            pygame.mixer.music.load(self.last_audio_file)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"TTS Error: {e}")

    def process_speech(self, raw_audio_data):
        clean_audio = self.extract_pure_speech(raw_audio_data)
        if len(clean_audio) == 0 or not self.auth.verify(clean_audio):
            return

        sf.write("temp.wav", clean_audio, self.fs)
        segments, _ = self.stt_model.transcribe("temp.wav", beam_size=5)
        user_text = "".join([s.text for s in segments]).strip()
        
        if user_text:
            print(f"You said: {user_text}")
            try:
                print("Thinking...")
                response = requests.post(N8N_WEBHOOK_URL, json={"message": user_text}, timeout=15)
                response.raise_for_status()
                
                res_data = response.json()
                reply = ""
                
                if isinstance(res_data, list) and len(res_data) > 0:
                    item = res_data[0]
                    reply = item.get("output") or item.get("outpput") or item.get("text") or str(item)
                elif isinstance(res_data, dict):
                    reply = res_data.get("output") or res_data.get("outpput") or res_data.get("text") or str(res_data)
                else:
                    reply = str(res_data)

                threading.Thread(target=self.tts_and_play, args=(reply,), daemon=True).start()
                
            except Exception as e:
                print(f"Cloud Connection Error: {e}")

    def run_core_loop(self):
        print("\nJARVIS is listening... (Ctrl+C to stop)")
        audio_buffer = []
        is_speaking = False
        consecutive_speech = 0
        silence_count = 0
        
        def audio_callback(indata, frames, time, status):
            self.audio_queue.put(indata.copy())

        with sd.InputStream(samplerate=self.fs, channels=1, callback=audio_callback):
            while True:
                chunk = self.audio_queue.get().flatten()
                speech_prob = self.vad_model(torch.tensor(chunk), self.fs).item()
                
                if speech_prob > 0.65:
                    consecutive_speech += 1
                    if consecutive_speech >= 2:
                        # INSTANT INTERRUPTION: Kill audio if JARVIS is currently speaking
                        if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                            self.kill_speech()
                            
                        if not is_speaking:
                            is_speaking = True
                            audio_buffer = []
                        audio_buffer.append(chunk)
                        silence_count = 0
                elif is_speaking:
                    audio_buffer.append(chunk)
                    silence_count += 1
                    if silence_count > 75:
                        is_speaking = False
                        full_audio = np.concatenate(audio_buffer)
                        threading.Thread(target=self.process_speech, args=(full_audio,), daemon=True).start()
                        audio_buffer = []

if __name__ == "__main__":
    app = VoiceAssistant()
    app.record_enrollment()
    try:
        app.run_core_loop()
    except KeyboardInterrupt:
        print("\nShutting down...")
        app.kill_speech()
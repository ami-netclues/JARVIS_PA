import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import torch
import sounddevice as sd
import numpy as np
import threading
import queue
import io
import warnings
import pygame
import soundfile as sf
import pyttsx3
from gtts import gTTS
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore")

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
        
        inputs = self.feature_extractor(
            waveform.squeeze().numpy(), 
            sampling_rate=sample_rate, 
            return_tensors="pt", 
            padding=True
        )
        
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
            print("Error: No master voice enrolled.")
            return False
            
        current_embedding = self.get_embedding(clean_audio_data)
        if current_embedding is None:
            print("Audio too short for matching. Ignoring.")
            return False
            
        cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        similarity = cosine_sim(self.master_embedding, current_embedding).item()
        
        print(f"Auth Score: {similarity:.3f} (Required: {self.threshold})")
        return similarity >= self.threshold

class VoiceAssistant:
    def __init__(self):
        self.fs = 16000
        self.chunk_size = 512
        
        pygame.mixer.init(frequency=32000) 
        self.auth = VoiceAuthenticator(threshold=0.88)
        
        print("Loading Silero VAD...")
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        self.get_speech_timestamps = utils[0]
        
        print("Loading Faster-Whisper STT...")
        self.stt_model = WhisperModel("base", device="cpu", compute_type="int8")

        self.is_playing = False
        self.interrupted = False
        self.audio_queue = queue.Queue()

    def extract_pure_speech(self, audio_data):
        tensor_audio = torch.tensor(audio_data).float()
        timestamps = self.get_speech_timestamps(tensor_audio, self.vad_model, sampling_rate=self.fs)
        
        if not timestamps:
            return np.array([])
            
        speech_chunks = [audio_data[ts['start']:ts['end']] for ts in timestamps]
        return np.concatenate(speech_chunks)

    def record_enrollment(self, duration=6):
        print("\nENROLLMENT")
        print(f"Recording for {duration} seconds.")
        print("Please read a sentence continuously...")
        recording = sd.rec(int(duration * self.fs), samplerate=self.fs, channels=1, dtype='float32')
        sd.wait()
        
        raw_audio = recording.flatten()
        clean_audio = self.extract_pure_speech(raw_audio)
        
        success = self.auth.enroll(clean_audio)
        if success:
            print("ENROLLMENT COMPLETE\n")
        else:
            print("ENROLLMENT FAILED. PLEASE RESTART\n")

    def tts_and_play(self, text):
        print(f"\nSystem Echo: {text}")
        self.is_playing = True
        self.interrupted = False
        
        try:

            engine = pyttsx3.init()
            
            engine.setProperty('rate', 210) 
        
            def on_word(name, location, length):
                if self.interrupted:
                    engine.stop()
                    print("\n[!] Audio Interrupted by user!")
                    
            engine.connect('started-word', on_word)
            
            # Play the audio
            engine.say(text)
            engine.runAndWait()
            
        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
            self.is_playing = False

    def process_speech(self, raw_audio_data):
        clean_audio = self.extract_pure_speech(raw_audio_data)
        if len(clean_audio) == 0:
            return
        
        if not self.auth.verify(clean_audio):
            print("ACCESS DENIED: Voice does not match.")
            return

        print("ACCESS GRANTED: Master voice recognized.")
        
        sf.write("temp.wav", clean_audio, self.fs)
        segments, _ = self.stt_model.transcribe("temp.wav", beam_size=5, language="en")
        text = "".join([segment.text for segment in segments]).strip()
        
        if text:
            print(f"You said: {text}")
            threading.Thread(target=self.tts_and_play, args=(text,), daemon=True).start()

    def run_core_loop(self):
        self.running = True 
        print("\nSystem listening... (Press Ctrl+C to stop)")
        
        audio_buffer = []
        is_speaking = False
        consecutive_speech_chunks = 0
        silence_chunks = 0
        
        trigger_threshold = 2 
        silence_threshold = int(self.fs / self.chunk_size * 0.8) 
        def audio_callback(indata, frames, time, status):
            if status:
                pass 
            self.audio_queue.put(indata.copy())

        with sd.InputStream(samplerate=self.fs, channels=1, dtype='float32', blocksize=self.chunk_size, callback=audio_callback):
            while True:
                chunk = self.audio_queue.get()
                chunk_tensor = torch.tensor(chunk.flatten())
                
                speech_prob = self.vad_model(chunk_tensor, self.fs).item()
                
                if speech_prob > 0.65: 
                    consecutive_speech_chunks += 1
                else:
                    consecutive_speech_chunks = 0
                
                if consecutive_speech_chunks >= trigger_threshold:
                    if self.is_playing and not self.interrupted:
                        # Interrupts the TTS immediately
                        self.interrupted = True 
                    
                    if not is_speaking:
                        is_speaking = True
                        audio_buffer = []
                    
                    audio_buffer.append(chunk)
                    silence_chunks = 0
                    
                elif is_speaking:
                    audio_buffer.append(chunk)
                    silence_chunks += 1
                    
                    if silence_chunks > silence_threshold:
                        is_speaking = False
                        complete_audio = np.concatenate(audio_buffer).flatten()
                        
                        threading.Thread(target=self.process_speech, args=(complete_audio,), daemon=True).start()
                        
                        audio_buffer = []
                        silence_chunks = 0
                        consecutive_speech_chunks = 0

if __name__ == "__main__":
    app = VoiceAssistant()
    app.record_enrollment(duration=6) 
    
    try:
        app.run_core_loop()
    except KeyboardInterrupt:
        print("\nStopping system safely...")
        app.running = False 
        pygame.mixer.quit()
        print("Exiting application.")
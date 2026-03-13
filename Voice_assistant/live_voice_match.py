import torch
import torchaudio
import sounddevice as sd
from scipy.io.wavfile import write
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import warnings

warnings.filterwarnings("ignore")

torchaudio.set_audio_backend("soundfile")

def record_audio(prompt_message, filename, duration=5, fs=16000):
    print(f"\n{prompt_message}")
    input("Press Enter to start recording...")
    print(f"🎤 Recording for {duration} seconds... Speak now!")

    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    
    write(filename, fs, recording)
    print(f"Saved to {filename}")

def get_voice_embedding(file_path, model, feature_extractor):
    # Load and standardize audio
    waveform, sample_rate = torchaudio.load(file_path)
    
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    # Extract features and get embedding
    inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        embeddings = model(**inputs).embeddings
    
    return torch.nn.functional.normalize(embeddings, dim=-1)

def main():
    # Setup Models
    print("Loading AI Models... Please wait.")
    model_name = "microsoft/wavlm-base-plus-sv"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = WavLMForXVector.from_pretrained(model_name)

    # Record Person 1
    record_audio("RECORDING PERSON 1", "person1.wav")

    # Record Person 2
    record_audio("RECORDING PERSON 2", "person2.wav")

    # Compare
    print("\nComparing voices... Analyzing patterns...")
    emb1 = get_voice_embedding("person1.wav", model, feature_extractor)
    emb2 = get_voice_embedding("person2.wav", model, feature_extractor)

    cosine_sim = torch.nn.CosineSimilarity(dim=-1)
    similarity = cosine_sim(emb1, emb2).item()

    # Final Result
    print("-" * 30)
    print(f"RESULT: {similarity:.2f}")
    if similarity >= 0.86:
        print("MATCH: Same Person")
    else:
        print("NO MATCH: Different People")
    print("-" * 30)

if __name__ == "__main__":
    main()
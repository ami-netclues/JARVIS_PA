from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
from recorder import record_audio

encoder = VoiceEncoder()

# load stored embedding
stored_embedding = np.load("database/embeddings.npy")

# record test voice
record_audio("test_voice.wav")

wav = preprocess_wav(Path("test_voice.wav"))
test_embedding = encoder.embed_utterance(wav)

# cosine similarity
similarity = np.dot(stored_embedding, test_embedding)

print("Similarity Score:", similarity)

if similarity > 0.75:
    print("✅ Same Person")
else:
    print("❌ Different Person")
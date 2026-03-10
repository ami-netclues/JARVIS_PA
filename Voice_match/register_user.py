from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
from recorder import record_audio

encoder = VoiceEncoder()

# record user voice
record_audio("user_voice.wav")

# preprocess audio
wav = preprocess_wav(Path("user_voice.wav"))

# generate embedding
embedding = encoder.embed_utterance(wav)

# save embedding
np.save("database/embeddings.npy", embedding)

print("User voice registered successfully!")
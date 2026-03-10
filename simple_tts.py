import pyttsx3

def text_to_voice(text):
    """
    Convert text to speech using pyttsx3.
    Note: You may need to install it first using: pip install pyttsx3
    """
    # Initialize the TTS engine
    engine = pyttsx3.init()
    
    # --- Optional: Adjusting properties ---
    # Speed of speech (default is usually around 200)
    rate = engine.getProperty('rate')
    engine.setProperty('rate', 150)  
    
    # Volume (0.0 to 1.0)
    volume = engine.getProperty('volume')
    engine.setProperty('volume', 1.0) 
    
    # Voice (0 for male, 1 for female depending on your OS)
    voices = engine.getProperty('voices')
    if len(voices) > 1:
        engine.setProperty('voice', voices[1].id) # Try female voice
    else:
        engine.setProperty('voice', voices[0].id)

    # print(f"Speaking: '{text}'")
    
    # Queue the text to be spoken
    engine.say(text)
    
    # Process the voice command
    engine.runAndWait()

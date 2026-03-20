import React, { useEffect, useMemo, useRef, useState } from "react";

type MessageResponse = {
  replyText: string;
};

type ChatMessage = {
  id: number;
  role: "user" | "assistant";
  text: string;
};

const SILENCE_DELAY_MS = 1300;
const SILENCE_THRESHOLD = 20;

const REGISTRATION_TEXT = "The future of artificial intelligence lies in seamless integration between human intuition and machine precision. JARVIS is designed to be more than just a tool; it is a personalized assistant that understands your voice and adapts to your needs. By speaking this paragraph clearly, you are helping the system build a unique profile that ensures your interactions remain secure and truly yours.";

type AuthStep = "LANDING" | "VERIFYING" | "ADMIN_PWD" | "ADMIN_PANEL" | "REGISTERING" | "CHAT";

const App: React.FC = () => {
  const [authStep, setAuthStep] = useState<AuthStep>("LANDING");
  const [availableProfiles, setAvailableProfiles] = useState<string[]>([]);
  const [selectedProfile, setSelectedProfile] = useState("");
  const [adminPassword, setAdminPassword] = useState("");
  const [regName, setRegName] = useState("");
  const [authError, setAuthError] = useState("");
  const [isVerifying, setIsVerifying] = useState(false);

  const [isRecording, setIsRecording] = useState(false);
  const [recognizedText, setRecognizedText] = useState("");
  const [replyText, setReplyText] = useState("");
  const [status, setStatus] = useState("Idle");
  const [error, setError] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [autoListen, setAutoListen] = useState(true);
  const [textInput, setTextInput] = useState("");

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const isRecordingRef = useRef(false);
  const autoListenRef = useRef(true);
  const recognizedTextRef = useRef("");

  const ttsUnlockedRef = useRef(false);
  const keepAliveIntervalRef = useRef<number | null>(null);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  const unlockTTS = () => {
    if (ttsUnlockedRef.current) return;
    if (!("speechSynthesis" in window)) return;

    // Only cancel if NOT currently speaking something important
    if (!window.speechSynthesis.speaking) {
      const utter = new SpeechSynthesisUtterance("");
      window.speechSynthesis.speak(utter);
      window.speechSynthesis.cancel();
    }

    ttsUnlockedRef.current = true;
    keepAudioAlive();
  };

  const keepAudioAlive = () => {
    if (keepAliveIntervalRef.current) return;
    keepAliveIntervalRef.current = window.setInterval(() => {
      if (!("speechSynthesis" in window)) return;
      // CRITICAL: Do NOT cancel or speak if we are already speaking a real response
      if (window.speechSynthesis.speaking) return;

      const utter = new SpeechSynthesisUtterance("");
      window.speechSynthesis.speak(utter);
      window.speechSynthesis.cancel();
    }, 10000);
  };

  useEffect(() => {
    const handleInteraction = () => {
      unlockTTS();
      document.removeEventListener("click", handleInteraction);
      document.removeEventListener("touchstart", handleInteraction);
    };
    document.addEventListener("click", handleInteraction);
    document.addEventListener("touchstart", handleInteraction);
    return () => {
      document.removeEventListener("click", handleInteraction);
      document.removeEventListener("touchstart", handleInteraction);
      if (keepAliveIntervalRef.current) {
        clearInterval(keepAliveIntervalRef.current);
      }
    };
  }, []);

  const waitForVoices = (): Promise<SpeechSynthesisVoice[]> => {
    return new Promise(resolve => {
      let voices = window.speechSynthesis.getVoices();
      if (voices.length) return resolve(voices);

      window.speechSynthesis.onvoiceschanged = () => {
        resolve(window.speechSynthesis.getVoices());
      };
    });
  };

  // Audio silence refs
  const audioContextRef = useRef<AudioContext | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const silenceStartRef = useRef<number | null>(null);

  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const userScrolledUpRef = useRef(false);
  const [isSpeechSupported, setIsSpeechSupported] = useState(true);

  // Optional: browser speech recognition
  const recognitionRef = useRef<SpeechRecognition | null>(null);

  const canUseSpeechRecognition = useMemo(() => {
    return Boolean((window as any).SpeechRecognition || (window as any).webkitSpeechRecognition);
  }, []);

  useEffect(() => {
    const SpeechRecognitionImpl =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (SpeechRecognitionImpl) {
      const rec: SpeechRecognition = new SpeechRecognitionImpl();
      rec.lang = "en-US";
      rec.continuous = false;
      rec.interimResults = false;

      rec.onresult = (event: SpeechRecognitionEvent) => {
        const t = event.results[0]?.[0]?.transcript ?? "";
        setRecognizedText(t);
        recognizedTextRef.current = t;
      };

      rec.onerror = () => {
        // Just ignore; we still have the raw audio
      };

      recognitionRef.current = rec;
    }
  }, []);

  useEffect(() => {
    setIsSpeechSupported("speechSynthesis" in window);
  }, []);

  useEffect(() => {
    fetchProfiles();
  }, []);

  const fetchProfiles = async () => {
    try {
      const res = await fetch("/api/auth/profiles");
      const data = await res.json();
      setAvailableProfiles(data.profiles || []);
    } catch (e) {
      console.error("Failed to fetch profiles", e);
    }
  };

  useEffect(() => {
    // Auto-scroll when new messages come in, unless the user intentionally scrolled up.
    if (!bottomRef.current) return;
    if (userScrolledUpRef.current) return;
    bottomRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length]);

  const decodeToPCM = async (blob: Blob): Promise<number[]> => {
    const arrayBuffer = await blob.arrayBuffer();
    const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
    const audioContext = new AudioCtx();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    // Resample to 16kHz if needed, but for now just get channel data
    // To be simple, we grab the first channel
    const rawData = audioBuffer.getChannelData(0);
    // Convert Float32Array to regular array for JSON
    return Array.from(rawData);
  };

  const startRecording = async () => {
    unlockTTS();
    if (isRecordingRef.current) return;

    setError(null);
    // keep history visible, so don't clear messages here
    setReplyText("");
    setRecognizedText("");
    recognizedTextRef.current = "";
    setStatus("Requesting microphone...");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      // Safari/iOS fix: AudioContext must be resumed on user interaction
      if (audioContextRef.current && audioContextRef.current.state === "suspended") {
        await audioContextRef.current.resume();
      }

      mediaRecorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      setIsRecording(true);
      isRecordingRef.current = true;
      setStatus("Recording...");

      // --- Silence Detection Setup ---
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      if (AudioCtx) {
        const audioContext = new AudioCtx();
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        const checkAudioLevel = () => {
          if (!isRecordingRef.current) return;

          analyser.getByteFrequencyData(dataArray);
          let sum = 0;
          for (let i = 0; i < bufferLength; i++) {
            sum += dataArray[i];
          }
          const average = sum / bufferLength;

          if (average < SILENCE_THRESHOLD) {
            if (!silenceStartRef.current) {
              silenceStartRef.current = Date.now();
            } else if (Date.now() - silenceStartRef.current > SILENCE_DELAY_MS) {
              void stopRecording();
              return; // Stop the loop
            }
          } else {
            silenceStartRef.current = null;
          }

          animationFrameRef.current = requestAnimationFrame(checkAudioLevel);
        };

        animationFrameRef.current = requestAnimationFrame(checkAudioLevel);
        audioContextRef.current = audioContext;
      }
      // -------------------------------

      if (recognitionRef.current) {
        try {
          recognitionRef.current.start();
        } catch {
          // Ignore if start fails
        }
      }
    } catch (e: any) {
      setError("Could not access microphone: " + (e?.message ?? String(e)));
      setStatus("Error");
    }
  };

  const stopRecording = async () => {
    if (!mediaRecorderRef.current) return;

    // Clean up silence detection
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (audioContextRef.current) {
      void audioContextRef.current.close().catch(() => { });
      audioContextRef.current = null;
    }
    silenceStartRef.current = null;

    setStatus("Stopping...");
    mediaRecorderRef.current.stop();
    setIsRecording(false);
    isRecordingRef.current = false;

    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch {
        // ignore
      }
    }

    // Wait a tiny bit to ensure dataavailable fired
    setTimeout(() => {
      if (authStep === "CHAT") {
        processAudioAndSend();
      } else if (authStep === "VERIFYING") {
        handleVoiceVerify();
      } else if (authStep === "REGISTERING") {
        handleVoiceRegister();
      }
    }, 300);
  };

  const handleVoiceVerify = async () => {
    setIsVerifying(true);
    setStatus("Verifying voice...");
    try {
      const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
      const formData = new FormData();
      formData.append("file", audioBlob, "audio.webm");
      formData.append("user_id", selectedProfile);

      const res = await fetch("/api/auth/verify", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.success) {
        setAuthStep("CHAT");
        setAuthError("");
        setStatus(`Authenticated (Score: ${data.score?.toFixed(3)})`);
      } else {
        setAuthError(data.message || "Voice match failed.");
        setStatus(`Mismatch (Score: ${data.score?.toFixed(3)})`);
        setAuthStep("LANDING");
      }
    } catch (e: any) {
      setAuthError("Auth error: " + (e.message || String(e)));
      setAuthStep("LANDING");
    } finally {
      setIsVerifying(false);
    }
  };

  const handleVoiceRegister = async () => {
    setIsVerifying(true);
    setStatus("Processing registration...");
    try {
      const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
      const formData = new FormData();
      formData.append("file", audioBlob, "audio.webm");
      formData.append("user_id", regName);
      formData.append("password", adminPassword);

      const res = await fetch("/api/auth/register", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.success) {
        await fetchProfiles();
        setAuthStep("ADMIN_PANEL");
        setStatus("Registered successfully");
      } else {
        setAuthError(data.message || "Registration failed.");
      }
    } catch (e: any) {
      setAuthError("Registration error: " + (e.message || String(e)));
    } finally {
      setIsVerifying(false);
    }
  };

  const handleProfileDelete = async (name: string) => {
    if (!window.confirm(`Are you sure you want to delete profile "${name}"?`)) return;

    try {
      const res = await fetch(`/api/auth/profiles/${encodeURIComponent(name)}?password=${encodeURIComponent(adminPassword)}`, {
        method: "DELETE"
      });
      const data = await res.json();
      if (data.success) {
        await fetchProfiles();
      } else {
        alert("Delete failed: " + data.message);
      }
    } catch (e) {
      alert("Error deleting profile");
    }
  };

  const processAudioAndSend = async () => {
    let text = recognizedTextRef.current.trim();

    if (!text) {
      setError("No speech recognized. Please try again.");
      setStatus("Idle");
      return;
    }

    await sendMessage(text);
  };

  const handleTextSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!textInput.trim()) return;

    const text = textInput.trim();
    setTextInput("");
    await sendMessage(text);
  };

  const sendMessage = async (text: string) => {
    setStatus("Sending to server...");
    setMessages((prev) => [
      ...prev,
      { id: Date.now(), role: "user", text },
    ]);

    try {
      const res = await fetch("/api/message", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });

      if (!res.ok) {
        const body = await res.text();
        throw new Error(`Server error ${res.status}: ${body}`);
      }

      const data = (await res.json()) as MessageResponse;
      setReplyText(data.replyText);
      setStatus("Success");
      setMessages((prev) => [
        ...prev,
        { id: Date.now() + 1, role: "assistant", text: data.replyText },
      ]);

      speakText(data.replyText);
    } catch (e: any) {
      setError("Failed to call server: " + (e?.message ?? String(e)));
      setStatus("Error");
    }
  };

  const speakText = async (text: string) => {
    if (!("speechSynthesis" in window)) {
      return;
    }
    if (!text) return;

    const voices = await waitForVoices();
    const utterance = new SpeechSynthesisUtterance(text);
    // Keep reference to prevent GC
    utteranceRef.current = utterance;

    if (voices.length > 0) {
      const preferredVoice = voices.find(v => v.lang.startsWith('en') && v.name.includes('Samantha')) ||
        voices.find(v => v.lang.startsWith('en')) ||
        voices[0];
      utterance.voice = preferredVoice;
    }

    window.speechSynthesis.cancel(); // clear queue

    // Re-introduce small delay for Safari
    setTimeout(() => {
      if (!utteranceRef.current) return;

      utteranceRef.current.onend = () => {
        utteranceRef.current = null;
        if (autoListenRef.current && !isRecordingRef.current) {
          // Fire-and-forget; errors will be surfaced via startRecording handlers
          void startRecording();
        }
      };

      utteranceRef.current.onerror = (e) => {
        console.error("SpeechSynthesis error:", e);
        utteranceRef.current = null;
      };

      window.speechSynthesis.speak(utteranceRef.current);
    }, 100);
  };

  const renderAuthLanding = () => (
    <div style={{ textAlign: "center", padding: "1rem" }}>
      <h2 style={{ marginBottom: "1.5rem", color: "#60a5fa" }}>Select Profile</h2>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: "1rem", marginBottom: "2rem" }}>
        {availableProfiles.map(name => (
          <div
            key={name}
            onClick={() => { setSelectedProfile(name); setAuthStep("VERIFYING"); setAuthError(""); }}
            style={{
              padding: "1.5rem 1rem",
              borderRadius: "1rem",
              background: "rgba(30,41,59,0.5)",
              border: "1px solid rgba(148,163,184,0.2)",
              cursor: "pointer",
              transition: "all 0.2s"
            }}
          >
            <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>👤</div>
            <div style={{ fontWeight: 600 }}>{name}</div>
          </div>
        ))}
        {availableProfiles.length === 0 && <p style={{ gridColumn: "1/-1", opacity: 0.6 }}>No profiles yet.</p>}
      </div>

      <button
        onClick={() => setAuthStep("ADMIN_PWD")}
        style={{
          background: "none",
          border: "1px solid rgba(148,163,184,0.3)",
          color: "#94a3b8",
          padding: "0.5rem 1rem",
          borderRadius: "0.5rem",
          cursor: "pointer",
          fontSize: "0.85rem"
        }}
      >
        Admin / Add Profile
      </button>
      {authError && <p style={{ color: "#ef4444", marginTop: "1rem", fontSize: "0.9rem" }}>{authError}</p>}
    </div>
  );

  const renderVerifying = () => (
    <div style={{ textAlign: "center", padding: "2rem" }}>
      <h2 style={{ marginBottom: "1rem" }}>Welcome, {selectedProfile}</h2>
      <p style={{ color: "#94a3b8", marginBottom: "2rem" }}>Please verify your voice to continue.</p>

      <div style={{ marginBottom: "2rem" }}>
        <button
          onClick={isRecording ? stopRecording : startRecording}
          disabled={isVerifying}
          style={{
            width: 120, height: 120, borderRadius: "50%", border: "none",
            background: isRecording ? "#ef4444" : "#3b82f6",
            color: "white", cursor: "pointer", fontSize: "3rem",
            boxShadow: isRecording ? "0 0 20px rgba(239,68,68,0.4)" : "0 8px 16px rgba(59,130,246,0.3)"
          }}
        >
          {isRecording ? "⏹" : "🎤"}
        </button>
      </div>
      <p style={{ fontSize: "1.1rem", color: isRecording ? "#60a5fa" : "#e5e7eb" }}>
        {isRecording ? "Listening..." : isVerifying ? "Processing..." : "Tap to Verify Voice"}
      </p>
      <button onClick={() => setAuthStep("LANDING")} style={{ marginTop: "2rem", background: "none", border: "none", color: "#94a3b8", cursor: "pointer" }}>Back</button>
    </div>
  );

  const renderAdminPwd = () => (
    <div style={{ textAlign: "center", padding: "2rem" }}>
      <h2 style={{ marginBottom: "1.5rem" }}>Admin Access</h2>
      <input
        type="password"
        placeholder="Enter Admin Password"
        value={adminPassword}
        onChange={(e) => setAdminPassword(e.target.value)}
        style={{
          padding: "0.75rem 1rem", borderRadius: "0.5rem", border: "1px solid #334155",
          background: "#0f172a", color: "white", marginBottom: "1rem", width: "100%", maxWidth: 300
        }}
      />
      <br />
      <button
        onClick={() => {
          if (adminPassword === "pass123") {
            setAuthStep("ADMIN_PANEL");
            setAuthError("");
          } else {
            setAuthError("Wrong password");
          }
        }}
        style={{
          padding: "0.75rem 2rem", borderRadius: "0.5rem", border: "none",
          background: "#3b82f6", color: "white", cursor: "pointer", fontWeight: 600
        }}
      >
        Unlock
      </button>
      {authError && <p style={{ color: "#ef4444", marginTop: "1rem" }}>{authError}</p>}
      <br />
      <button onClick={() => setAuthStep("LANDING")} style={{ marginTop: "1rem", background: "none", border: "none", color: "#94a3b8", cursor: "pointer" }}>Cancel</button>
    </div>
  );

  const renderAdminPanel = () => (
    <div style={{ padding: "1rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "2rem" }}>
        <h2 style={{ margin: 0 }}>Manage Profiles ({availableProfiles.length}/10)</h2>
        <button onClick={() => setAuthStep("LANDING")} style={{ background: "none", border: "none", color: "#3b82f6", cursor: "pointer" }}>Done</button>
      </div>

      <div style={{ background: "rgba(30,41,59,0.3)", padding: "1.5rem", borderRadius: "1rem", border: "1px solid rgba(148,163,184,0.1)", marginBottom: "2rem" }}>
        <h3 style={{ marginTop: 0, fontSize: "1rem", color: "#60a5fa" }}>Add New Profile</h3>
        <input
          placeholder="Person Name"
          value={regName}
          onChange={(e) => setRegName(e.target.value)}
          style={{
            padding: "0.6rem 1rem", borderRadius: "0.5rem", border: "1px solid #334155",
            background: "#0f172a", color: "white", marginBottom: "1rem", width: "100%", maxWidth: 300
          }}
        />
        <br />
        <button
          onClick={() => { if (regName.trim()) setAuthStep("REGISTERING"); }}
          disabled={availableProfiles.length >= 10 || !regName.trim()}
          style={{
            padding: "0.6rem 1.5rem", borderRadius: "0.5rem", border: "none",
            background: "#22c55e", color: "white", cursor: "pointer", fontWeight: 600,
            opacity: (availableProfiles.length >= 10 || !regName.trim()) ? 0.5 : 1
          }}
        >
          Begin Registration
        </button>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
        {availableProfiles.map(name => (
          <div key={name} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0.75rem 1rem", background: "rgba(30,41,59,0.5)", borderRadius: "0.5rem" }}>
            <span>{name}</span>
            <div style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
              <span style={{ color: "#22c55e", fontSize: "0.85rem" }}>✓ Active</span>
              <button
                onClick={() => handleProfileDelete(name)}
                style={{ background: "none", border: "none", color: "#ef4444", cursor: "pointer", fontSize: "1.2rem", padding: "0 0.5rem" }}
                title="Delete Profile"
              >
                🗑
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderRegistering = () => (
    <div style={{ padding: "1rem" }}>
      <h2 style={{ marginBottom: "1rem" }}>Registering {regName}</h2>
      <p style={{ color: "#94a3b8", marginBottom: "1.5rem" }}>Please read the following paragraph clearly to capture your voice signature:</p>

      <div style={{
        padding: "1.5rem", borderRadius: "1rem", background: "#0f172a",
        border: "1px solid #334155", marginBottom: "2rem", lineHeight: 1.6, fontSize: "1rem",
        color: isRecording ? "#86efac" : "#e5e7eb"
      }}>
        "{REGISTRATION_TEXT}"
      </div>

      <div style={{ textAlign: "center" }}>
        <button
          onClick={isRecording ? stopRecording : startRecording}
          disabled={isVerifying}
          style={{
            padding: "1rem 2rem", borderRadius: "999px", border: "none",
            background: isRecording ? "#ef4444" : "#3b82f6",
            color: "white", cursor: "pointer", fontWeight: 600, fontSize: "1.1rem",
            boxShadow: isRecording ? "0 0 20px rgba(239,68,68,0.4)" : "0 8px 16px rgba(59,130,246,0.2)"
          }}
        >
          {isRecording ? "Stop Recording" : "Start Recording"}
        </button>
        <p style={{ marginTop: "1rem", color: "#94a3b8" }}>{status}</p>
        <button onClick={() => setAuthStep("ADMIN_PANEL")} style={{ marginTop: "1rem", background: "none", border: "none", color: "#94a3b8", cursor: "pointer" }}>Cancel</button>
      </div>
    </div>
  );

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background:
          "radial-gradient(1200px 700px at 20% 10%, rgba(56,189,248,0.18), transparent 60%), radial-gradient(1000px 700px at 80% 0%, rgba(34,197,94,0.14), transparent 55%), linear-gradient(180deg, #0b1220 0%, #0f172a 55%, #060913 100%)",
        color: "#e5e7eb",
        fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      }}
    >
      <div
        style={{
          maxWidth: 860,
          width: "100%",
          padding: "2rem",
          borderRadius: "1rem",
          background: "rgba(2,6,23,0.62)",
          border: "1px solid rgba(148,163,184,0.18)",
          boxShadow: "0 30px 80px rgba(0,0,0,0.55)",
          backdropFilter: "blur(10px)",
        }}
      >
        {authStep === "LANDING" && renderAuthLanding()}
        {authStep === "VERIFYING" && renderVerifying()}
        {authStep === "ADMIN_PWD" && renderAdminPwd()}
        {authStep === "ADMIN_PANEL" && renderAdminPanel()}
        {authStep === "REGISTERING" && renderRegistering()}

        {authStep === "CHAT" && (
          <>
            {/* STT and TTS status */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <button
                onClick={() => { setAuthStep("LANDING"); setMessages([]); }}
                style={{ background: "none", border: "none", color: "#60a5fa", cursor: "pointer", fontSize: "0.9rem" }}
              >
                ← Switch Profile
              </button>
              <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", flexWrap: "wrap" }}>
                <span
                  style={{
                    padding: "0.25rem 0.6rem",
                    borderRadius: "999px",
                    fontSize: "0.8rem",
                    border: "1px solid rgba(148,163,184,0.22)",
                    background: "rgba(15,23,42,0.55)",
                    color: "#cbd5e1",
                  }}
                >
                  STT: {canUseSpeechRecognition ? "Browser" : "Not supported"}
                </span>
                <span
                  style={{
                    padding: "0.25rem 0.6rem",
                    borderRadius: "999px",
                    fontSize: "0.8rem",
                    border: "1px solid rgba(148,163,184,0.22)",
                    background: "rgba(15,23,42,0.55)",
                    color: "#cbd5e1",
                  }}
                >
                  TTS: {isSpeechSupported ? "Browser" : "Not supported"}
                </span>
              </div>
            </div>

            {/* Jarvis Voice Assistant Title and Description*/}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: "1rem",
                marginBottom: "1rem",
                flexWrap: "wrap",
              }}
            >
              <div>
                <h1 style={{ fontSize: "1.75rem", marginBottom: "0.25rem", alignItems: "center", justifyContent: "center" }}>
                  Jarvis Voice Assistant
                </h1>
                <div style={{ color: "#9ca3af", fontSize: "0.95rem", textAlign: "center" }}>
                  Active Profile: <span style={{ color: "#60a5fa", fontWeight: "bold" }}>{selectedProfile}</span>
                </div>
              </div>
            </div>

            {/* Conversation history */}
            {messages.length > 0 && (
              <div
                ref={scrollContainerRef}
                onScroll={() => {
                  const el = scrollContainerRef.current;
                  if (!el) return;
                  const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
                  userScrolledUpRef.current = distanceFromBottom > 60;
                }}
                style={{
                  marginTop: "2.5rem",
                  marginBottom: "1rem",
                  maxHeight: "420px",
                  overflowY: "auto",
                  paddingRight: "0.25rem",
                  padding: "0.5rem",
                  borderRadius: "0.75rem",
                  border: "1px solid rgba(148,163,184,0.16)",
                  background:
                    "linear-gradient(180deg, rgba(2,6,23,0.55) 0%, rgba(2,6,23,0.25) 100%)",
                }}
              >
                {messages.map((m) => (
                  <div
                    key={m.id}
                    style={{
                      display: "flex",
                      justifyContent: m.role === "user" ? "flex-end" : "flex-start",
                      marginBottom: "0.5rem",
                    }}
                  >
                    <div
                      style={{
                        padding: "0.6rem 0.9rem",
                        borderRadius: "0.75rem",
                        maxWidth: "80%",
                        fontSize: "0.95rem",
                        background:
                          m.role === "user"
                            ? "linear-gradient(to right, #22c55e, #4ade80)"
                            : "#010409",
                        color: m.role === "user" ? "#022c22" : "#e5e7eb",
                        border:
                          m.role === "user"
                            ? "1px solid rgba(22,163,74,0.8)"
                            : "1px solid rgba(56,189,248,0.4)",
                      }}
                    >
                      <div style={{ fontSize: "0.75rem", opacity: 0.8, marginBottom: "0.15rem" }}>
                        {m.role === "user" ? "You" : "Jarvis"}
                      </div>
                      <div>{m.text}</div>
                    </div>
                  </div>
                ))}
                <div ref={bottomRef} />
              </div>
            )}

            {/* Text Chat Input */}
            <form onSubmit={handleTextSubmit} style={{ display: 'flex', gap: '0.8rem', marginBottom: '1.5rem' }}>
              <input
                type="text"
                placeholder="Type your message..."
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                style={{
                  flex: 1,
                  padding: "0.8rem 1.2rem",
                  borderRadius: "0.75rem",
                  border: "1px solid rgba(148,163,184,0.2)",
                  background: "rgba(15,23,42,0.6)",
                  color: "white",
                  fontSize: "1rem",
                  outline: "none",
                  transition: "border-color 0.2s",
                }}
                onFocus={(e) => e.target.style.borderColor = "rgba(96,165,250,0.5)"}
                onBlur={(e) => e.target.style.borderColor = "rgba(148,163,184,0.2)"}
              />
              <button
                type="submit"
                style={{
                  padding: "0.8rem 1.5rem",
                  borderRadius: "0.75rem",
                  border: "none",
                  background: "linear-gradient(to right, #3b82f6, #2563eb)",
                  color: "white",
                  cursor: "pointer",
                  fontWeight: 600,
                  fontSize: "1rem",
                  boxShadow: "0 4px 12px rgba(37,99,235,0.2)",
                  transition: "transform 0.1s",
                }}
                onMouseDown={(e) => e.currentTarget.style.transform = "scale(0.98)"}
                onMouseUp={(e) => e.currentTarget.style.transform = "scale(1)"}
              >
                Send
              </button>
            </form>

            {/* Record Button and Auto listen checkbox*/}
            <div
              style={{
                marginBottom: "1rem",
                display: "flex",
                alignItems: "center",
                gap: "1rem",
                justifyContent: "space-between",
                flexWrap: "wrap",
              }}
            >
              <button
                onClick={isRecording ? stopRecording : startRecording}
                style={{
                  padding: "0.75rem 1.5rem",
                  borderRadius: "999px",
                  border: "none",
                  cursor: "pointer",
                  fontWeight: 600,
                  fontSize: "1rem",
                  background: isRecording
                    ? "linear-gradient(to right, #ef4444, #dc2626)"
                    : "linear-gradient(to right, #22c55e, #4ade80)",
                  color: "#04130a",
                  boxShadow: isRecording
                    ? "0 12px 28px rgba(239,68,68,0.25)"
                    : "0 12px 28px rgba(34,197,94,0.25)",
                }}
              >
                {isRecording ? "Stop recording" : "Start recording"}
              </button>

              <label
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.5rem",
                  fontSize: "0.9rem",
                  color: "#e5e7eb",
                }}
              >
                <input
                  type="checkbox"
                  checked={autoListen}
                  onChange={(e) => {
                    setAutoListen(e.target.checked);
                    autoListenRef.current = e.target.checked;
                  }}
                />
                Auto listen after reply
              </label>
            </div>


            {/* Error message*/}
            {error && (
              <div
                style={{
                  marginTop: "0.75rem",
                  padding: "0.75rem",
                  borderRadius: "0.5rem",
                  background: "rgba(127,29,29,0.9)",
                  color: "#fee2e2",
                  fontSize: "0.9rem",
                }}
              >
                {error}
              </div>
            )}

            {/* Status and Listening indicator*/}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: "1rem",
                marginBottom: "0.75rem",
                flexWrap: "wrap",
              }}
            >
              <p style={{ margin: 0, fontSize: "0.9rem", color: "#a5b4fc" }}>Status: {status}</p>
              <div style={{ fontSize: "0.85rem", color: "#94a3b8" }}>
                {isRecording ? "Listening…" : "Ready"}
              </div>
            </div>
          </>
        )}

      </div>
    </div>
  );
};

export default App;
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
const SILENCE_THRESHOLD = 15;

const App: React.FC = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [recognizedText, setRecognizedText] = useState("");
  const [replyText, setReplyText] = useState("");
  const [status, setStatus] = useState("Idle");
  const [error, setError] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [autoListen, setAutoListen] = useState(true);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const isRecordingRef = useRef(false);
  const autoListenRef = useRef(true);
  const recognizedTextRef = useRef("");

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
    // Auto-scroll when new messages come in, unless the user intentionally scrolled up.
    if (!bottomRef.current) return;
    if (userScrolledUpRef.current) return;
    bottomRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length]);

  const startRecording = async () => {
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
      processAudioAndSend();
    }, 300);
  };

  const processAudioAndSend = async () => {
    const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });

    let text = recognizedTextRef.current.trim();

    if (!text) {
      // Optional: send audio to backend /api/transcribe if you enable it.
      // For now we'll just show a message and not call that endpoint by default.
      setError("No speech recognized. Please try again.");
      setStatus("Idle");
      return;
    }

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

  const speakText = (text: string) => {
    if (!("speechSynthesis" in window)) {
      return;
    }
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.onend = () => {
      if (autoListenRef.current && !isRecordingRef.current) {
        // Fire-and-forget; errors will be surfaced via startRecording handlers
        void startRecording();
      }
    };
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
  };

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
        {/* STT and TTS status */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", flexWrap: "wrap", marginLeft: "auto" }}>
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
            <div style={{ color: "#9ca3af", fontSize: "0.95rem" }}>
              Talk → text → n8n → reply + voice
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
                        : "#020617",
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

      </div>
    </div>
  );
};

export default App;
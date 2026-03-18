import { useState, useEffect, useRef } from 'react';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import './index.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [isListening, setIsListening] = useState(false);
  const [profiles, setProfiles] = useState([]);
  const [socket, setSocket] = useState(null);
  const [listeningState, setListeningState] = useState('idle'); // idle, detecting, processing, unauthorized
  const [activeUser, setActiveUser] = useState(null);
  const [enrollStatus, setEnrollStatus] = useState({ step: 0, status: 'idle', error: null });

  const ttsContextRef = useRef(null);   // Dedicated context for TTS playback (44.1kHz)
  const micContextRef = useRef(null);   // Dedicated context for mic capture (16kHz)
  const processorRef = useRef(null);
  const streamRef = useRef(null);
  const activeSourcesRef = useRef([]); // Track multiple sources for streaming
  const nextStartTimeRef = useRef(0);  // Scheduling: when the next audio chunk should play

  // WebSocket Connection
  useEffect(() => {
    let ws = null;
    let reconnectTimer = null;
    let isCleanedUp = false; // Guard against React StrictMode double-mount

    const connectWs = () => {
      if (isCleanedUp) return; // Don't reconnect if component was unmounted
      ws = new WebSocket('ws://127.0.0.1:8000/ws');
      ws.binaryType = 'arraybuffer';
      
      ws.onopen = () => {
        if (isCleanedUp) { ws.close(); return; }
        setSocket(ws);
      };

      ws.onmessage = (event) => {
        if (isCleanedUp) return;
        if (event.data instanceof ArrayBuffer) {
          playAudioBuffer(event.data);
          return;
        }

        try {
          const data = JSON.parse(event.data);
          switch (data.type) {
            case 'status':
              setIsListening(data.listening);
              if (data.profiles) setProfiles(data.profiles);
              if (!data.listening) stopLocalRecording();
              break;
            case 'listening_state':
              setListeningState(data.state);
              if (data.user) setActiveUser(data.user);
              if (data.state === 'idle' || data.state === 'unauthorized') {
                if (!isListening) stopLocalRecording();
                setTimeout(() => setActiveUser(null), 3000);
              }
              break;
            case 'message':
              setMessages(prev => [...prev, { role: data.role, content: data.text }]);
              if (data.role === 'user') {
                setListeningState('idle');
                stopAudioPlayback();
              }
              break;
            case 'stop_audio':
              stopAudioPlayback();
              break;
            case 'enroll_status':
              setEnrollStatus(prev => ({ ...prev, status: data.status, error: null }));
              break;
            case 'enroll_result':
              if (data.success) {
                if (data.done) {
                  setEnrollStatus({ step: 0, status: 'success', error: null });
                  if (data.profiles) setProfiles(data.profiles);
                  setTimeout(() => setEnrollStatus({ step: 0, status: 'idle', error: null }), 3000);
                } else {
                  setEnrollStatus(prev => ({ ...prev, step: data.step + 1, status: 'idle' }));
                }
              } else {
                setEnrollStatus(prev => ({ ...prev, status: 'error', error: data.error }));
              }
              stopLocalRecording();
              break;
          }
        } catch (err) {
          console.error("DEBUG: Failed to parse WS message:", err);
        }
      };

      ws.onclose = () => {
        if (isCleanedUp) return; // Don't reconnect after intentional cleanup
        setSocket(null);
        setIsListening(false);
        stopLocalRecording();
        stopAudioPlayback();
        reconnectTimer = setTimeout(connectWs, 3000);
      };
    };
    
    connectWs();
    return () => {
      isCleanedUp = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (ws) { ws.onclose = null; ws.close(); }
      stopLocalRecording();
      stopAudioPlayback();
    };
  }, []);

  const stopAudioPlayback = () => {
    activeSourcesRef.current.forEach(source => {
      try { source.stop(); } catch (e) {}
    });
    activeSourcesRef.current = [];
    nextStartTimeRef.current = 0; // Reset schedule so next TTS starts immediately
  };

  const playAudioBuffer = (arrayBuffer) => {
    // Use a dedicated TTS context, never the 16kHz mic context
    if (!ttsContextRef.current || ttsContextRef.current.state === 'closed') {
      ttsContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      nextStartTimeRef.current = 0;
    }
    const context = ttsContextRef.current;

    context.decodeAudioData(arrayBuffer.slice(0), (buffer) => {
      const source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(context.destination);

      // Schedule chunk to start AFTER the previous chunk ends (sequential playback)
      const startAt = Math.max(context.currentTime, nextStartTimeRef.current);
      source.start(startAt);
      nextStartTimeRef.current = startAt + buffer.duration;
      
      activeSourcesRef.current.push(source);
      source.onended = () => {
        activeSourcesRef.current = activeSourcesRef.current.filter(s => s !== source);
      };
    }, (err) => {
      console.warn('TTS decodeAudioData error:', err);
    });
  };

  const startLocalRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      // Always use a dedicated 16kHz context for the mic, separate from TTS
      const micContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      micContextRef.current = micContext;
      const source = micContext.createMediaStreamSource(stream);
      const processor = micContext.createScriptProcessor(1024, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        // Convert Float32 to Int16 PCM
        const pcmData = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          pcmData[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7FFF;
        }
        if (socket && socket.readyState === WebSocket.OPEN) {
          socket.send(pcmData.buffer);
        }
      };

      source.connect(processor);
      processor.connect(micContext.destination);
    } catch (err) {
      console.error("Mic access error:", err);
      alert("Microphone access is required for voice features.");
    }
  };

  const stopLocalRecording = () => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (micContextRef.current) {
      micContextRef.current.close();
      micContextRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
  };

  // API Calls
  const fetchProfiles = async () => {
    try {
      const res = await fetch('http://127.0.0.1:8000/api/profiles');
      const data = await res.json();
      setProfiles(data);
    } catch (e) {
      console.error("Failed to fetch profiles:", e);
    }
  };

  const deleteProfile = async (slotId) => {
    try {
      await fetch(`http://127.0.0.1:8000/api/profiles/${slotId}`, { method: 'DELETE' });
      fetchProfiles();
    } catch (e) {
      console.error("Failed to delete profile:", e);
    }
  };

  // Chat Actions
  const sendMessage = (text) => {
    if (socket && text.trim()) {
      setMessages(prev => [...prev, { role: 'user', content: text }]);
      socket.send(JSON.stringify({ action: 'send_text', text }));
    }
  };

  const toggleListening = () => {
    if (socket) {
      if (isListening) {
        socket.send(JSON.stringify({ action: 'stop_listening' }));
        stopLocalRecording();
      } else {
        if (profiles.length === 0) {
          alert('Please register your voice in the sidebar first.');
          return;
        }
        socket.send(JSON.stringify({ action: 'start_listening' }));
        startLocalRecording();
      }
    }
  };

  // Enrollment Actions
  const startEnrollment = () => {
    setEnrollStatus({ step: 1, status: 'idle', error: null });
  };
  
  const cancelEnrollment = () => {
    setEnrollStatus({ step: 0, status: 'idle', error: null });
    stopLocalRecording();
  };

  const recordEnrollmentStep = (step, target, slot, name) => {
    if (socket) {
      socket.send(JSON.stringify({
        action: 'enroll_step',
        step,
        target,
        slot,
        name
      }));
      startLocalRecording(); // Backend will timeout/stop based on target length
    }
  };

  return (
    <div className="app-container">
      <Sidebar 
        profiles={profiles} 
        onDeleteProfile={deleteProfile}
        enrollStatus={enrollStatus}
        onStartEnrollment={startEnrollment}
        onCancelEnrollment={cancelEnrollment}
        onRecordEnrollmentStep={recordEnrollmentStep}
      />
      <main className="main-content">
        <header className="app-header">
          <h1> JARVIS</h1>
          <div className={`status-indicator ${socket ? 'connected' : 'disconnected'}`}>
            <span className="dot"></span>
            {socket ? 'Backend Connected' : 'Connecting to Server...'}
          </div>
          {profiles.length === 0 && (
            <div className="status-indicator mock-mode">
              <span className="dot yellow"></span>
              DEMO/MOCK MODE
            </div>
          )}
        </header>

        {profiles.length === 0 && (
          <div className="registration-warning glass-panel">
            <h3>🛡️ Voice Registration Required</h3>
            <p>Please use the sidebar on the left to register your voice before using the microphone. Recognition is currently disabled.</p>
          </div>
        )}
        <ChatInterface 
          messages={messages} 
          onSendMessage={sendMessage}
          isListening={isListening}
          onToggleListening={toggleListening}
          listeningState={listeningState}
          activeUser={activeUser}
        />
      </main>
    </div>
  );
}

export default App;

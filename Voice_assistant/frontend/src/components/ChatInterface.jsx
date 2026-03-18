import React, { useState, useEffect, useRef } from 'react';
import { Mic, Square, Send, User, Bot } from 'lucide-react';

export default function ChatInterface({ 
  messages, 
  onSendMessage, 
  isListening, 
  onToggleListening,
  listeningState,
  activeUser
}) {
  const [inputText, setInputText] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, listeningState]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputText.trim()) {
      onSendMessage(inputText);
      setInputText('');
    }
  };

  const renderListeningState = () => {
    if (!isListening) return null;

    let content = null;
    let className = "listening-banner ";

    switch (listeningState) {
      case 'idle':
        content = <><span className="pulse-ring"></span> 🟢 LISTENING ACTIVE</>;
        className += 'state-idle';
        break;
      case 'processing':
        content = <><span className="spinner"></span> Processing {activeUser ? `(${activeUser})` : ''}...</>;
        className += 'state-processing';
        break;
      case 'unauthorized':
        content = <>⚠️ Unauthorized person detected.</>;
        className += 'state-unauthorized';
        break;
      default:
        content = <><span className="pulse-ring"></span> Listening... (Active Loop)</>;
        className += 'state-idle';
    }

    return (
      <div className={className}>
        {content}
      </div>
    );
  };

  return (
    <div className="chat-container glass-panel">
      <div className="messages-area">
        {messages.length === 0 ? (
          <div className="empty-chat">
            <Bot size={48} className="empty-icon" />
            <p>Hello! I am JARVIS. How can I assist you today?</p>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div key={idx} className={`message-wrapper ${msg.role === 'user' ? 'user' : 'bot'}`}>
              <div className="message-avatar">
                {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
              </div>
              <div className="message-content">
                <span className="bubble-label">{msg.role === 'user' ? 'You' : 'Jarvis'}</span>
                <div className={`chat-bubble ${msg.role}-bubble`}>
                  {msg.content}
                </div>
              </div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {renderListeningState()}

      <div className="input-area">
        <form onSubmit={handleSubmit} className="input-form">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Message JARVIS..."
            disabled={isListening}
            className="chat-input"
          />
          <button 
            type="submit" 
            className="btn btn-icon send-btn" 
            disabled={!inputText.trim() || isListening}
            title="Send Message"
          >
            <Send size={20} />
          </button>
        </form>

        <button 
          className={`btn mic-btn ${isListening ? 'active stop' : 'start'}`}
          onClick={onToggleListening}
          title={isListening ? "Stop Listening" : "Start Voice Assistant"}
        >
          {isListening ? (
            <><Square size={20} fill="currentColor" /> Stop</>
          ) : (
            <><Mic size={20} /> Mic</>
          )}
        </button>
      </div>
    </div>
  );
}

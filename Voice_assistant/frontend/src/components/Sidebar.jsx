import React, { useState } from 'react';
import { Mic, UserPlus, Settings, Trash2, X } from 'lucide-react';

const SENTENCES = [
  "I use Jarvis every single day",
  "I am registering my voice for the JARVIS personal assistant.",
  "Voice authentication keeps me safe",
  "The integration of voice authentication enhances system security.",
  "My voice is my password, and it is unique to me."
];

export default function Sidebar({ 
  profiles, 
  onDeleteProfile, 
  enrollStatus, 
  onStartEnrollment, 
  onCancelEnrollment, 
  onRecordEnrollmentStep 
}) {
  const [enrollName, setEnrollName] = useState('');
  const [enrollSlot, setEnrollSlot] = useState(1);

  const renderRegistrationFlow = () => {
    const { step, status, error } = enrollStatus;

    if (step === 0) {
      return (
        <div className="registration-flow">
          <div className="input-group">
            <label>Select Slot</label>
            <select value={enrollSlot} onChange={e => setEnrollSlot(Number(e.target.value))}>
              <option value={1}>Slot 1</option>
              <option value={2}>Slot 2</option>
              <option value={3}>Slot 3</option>
            </select>
          </div>
          
          <div className="input-group">
            <label>User Name</label>
            <input 
              type="text" 
              placeholder="Enter your name" 
              value={enrollName}
              onChange={e => setEnrollName(e.target.value)}
            />
          </div>

          <button 
            className="btn btn-primary primary-action-btn"
            onClick={() => {
              if (!enrollName.trim()) {
                alert("Please enter a name");
                return;
              }
              onStartEnrollment();
            }}
          >
            <UserPlus size={18} /> Start Registration
          </button>
        </div>
      );
    }

    // Active Registration
    const targetSentence = SENTENCES[step - 1];

    return (
      <div className="active-registration">
        <div className="registration-header">
          <h4>Step {step} of 5</h4>
          <button className="btn-icon cancel-btn" onClick={onCancelEnrollment} title="Cancel">
            <X size={16} />
          </button>
        </div>

        <div className="sentence-card">
          <p className="instruction-text">Please say:</p>
          <p className="target-sentence">"{targetSentence}"</p>
        </div>

        {error && <div className="alert error-alert">{error}</div>}
        
        {status === 'success' ? (
          <div className="alert success-alert">Profile generated successfully! ✅</div>
        ) : (
          <button 
            className={`btn btn-primary recording-btn ${status === 'recording' || status === 'processing' ? 'active' : ''}`}
            disabled={status !== 'idle'}
            onClick={() => onRecordEnrollmentStep(step, targetSentence, enrollSlot, enrollName)}
          >
            {status === 'recording' ? (
              <><span className="pulsing-dot"></span> Recording...</>
            ) : status === 'processing' ? (
              <><span className="spinner"></span> Processing...</>
            ) : (
              <><Mic size={18} /> Record Sentence {step}</>
            )}
          </button>
        )}
      </div>
    );
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <Settings size={22} className="header-icon" />
        <h2>Settings</h2>
      </div>

      <div className="sidebar-section">
        <h3><Mic size={16} /> Voice Registration</h3>
        {renderRegistrationFlow()}
      </div>

      <hr className="divider" />

      <div className="sidebar-section">
        <h3><UserPlus size={16} /> Profiles</h3>
        
        {profiles.length === 0 ? (
          <div className="empty-state">No profiles registered.</div>
        ) : (
          <div className="profiles-list">
            {profiles.map(p => (
              <div key={p.slot_id} className="profile-item">
                <div className="profile-info">
                  <span className="slot-badge">Slot {p.slot_id}</span>
                  <span className="profile-name">{p.user_name}</span>
                </div>
                <button 
                  className="btn-icon delete-btn" 
                  onClick={() => onDeleteProfile(p.slot_id)}
                  title="Delete Profile"
                >
                  <Trash2 size={16} />
                </button>
              </div>
            ))}
          </div>
        )}

        <div className={`auth-status-badge ${profiles.length > 0 ? 'active' : 'inactive'}`}>
          {profiles.length > 0 ? '🛡️ Auth Active' : '⚠️ No Auth Set'}
        </div>
      </div>
    </aside>
  );
}

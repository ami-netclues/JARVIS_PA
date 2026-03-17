import sqlite3
import numpy as np
import io
import os

class VoiceDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS voice_profiles (
                slot_id INTEGER PRIMARY KEY,
                user_name TEXT,
                embedding BLOB,
                enrolled_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Ensure 3 slots exist
        for i in range(1, 4):
            cursor.execute('INSERT OR IGNORE INTO voice_profiles (slot_id) VALUES (?)', (i,))
        conn.commit()
        conn.close()

    def save_profile(self, slot_id, user_name, embedding):
        """Saves a voice profile to a specific slot (1, 2, or 3)."""
        if slot_id not in [1, 2, 3]:
            print(f"Invalid slot ID: {slot_id}. Must be 1, 2, or 3.")
            return False
            
        # Convert torch tensor to numpy then to bytes
        if hasattr(embedding, 'numpy'):
            embedding = embedding.detach().cpu().numpy()
        
        buffer = io.BytesIO()
        np.save(buffer, embedding)
        embedding_bytes = buffer.getvalue()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE voice_profiles 
            SET user_name = ?, embedding = ?, enrolled_at = CURRENT_TIMESTAMP
            WHERE slot_id = ?
        ''', (user_name, embedding_bytes, slot_id))
        conn.commit()
        conn.close()
        return True

    def get_all_profiles(self):
        """Returns a list of all enrolled profiles."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT slot_id, user_name, embedding FROM voice_profiles WHERE embedding IS NOT NULL')
        rows = cursor.fetchall()
        
        profiles = []
        for row in rows:
            buffer = io.BytesIO(row[2])
            embedding = np.load(buffer)
            profiles.append({
                'slot_id': row[0],
                'user_name': row[1],
                'embedding': embedding
            })
        conn.close()
        return profiles

    def delete_profile(self, slot_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE voice_profiles SET user_name = NULL, embedding = NULL WHERE slot_id = ?', (slot_id,))
        conn.commit()
        conn.close()

# Jarvis Voice Web App (React + Python) + n8n Webhook

Production-oriented starter that:
- Records audio in the browser (no server audio hardware needed)
- Converts speech → text (browser Web Speech API first; Python fallback transcription endpoint)
- Calls an **n8n webhook** and returns the reply
- Speaks the reply back using **browser TTS** (`speechSynthesis`)

## Repo layout
- `frontend/`: React (Vite + TypeScript)
- `backend/`: FastAPI (Python)

## Features
- **Client audio capture** via `MediaRecorder`
- **STT (speech-to-text)**:
  - Primary: Browser `SpeechRecognition` (Chrome/Edge)
  - Fallback: Upload audio to backend `/api/transcribe` (Whisper via `faster-whisper` if you enable it)
- **n8n webhook call** from backend (`/api/message`) so the webhook URL can stay server-side
- **TTS (text-to-speech)** in browser (`window.speechSynthesis`)

## Prerequisites
- Node.js 18+ (for frontend)
- Python 3.10+ (for backend)

## Local development

### 1) Backend
Create env file:

```bash
copy backend\.env.example backend\.env
```

Edit `backend\.env` and set `N8N_WEBHOOK_URL`.

Install + run:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2) Frontend

```bash
cd frontend
npm install
npm run dev
```

Open Vite URL (usually `http://localhost:5173`).

## API
- `POST /api/message`
  - body: `{ "text": "hello" }` OR `{ "text": "", "sessionId": "...", "meta": {...} }`
  - response: `{ "replyText": "..." }`
- `POST /api/transcribe` (optional fallback)
  - multipart form-data: `file` (audio blob)
  - response: `{ "text": "..." }`

## n8n webhook format
Backend sends:

```json
{
  "text": "user said ...",
  "sessionId": "uuid",
  "meta": {
    "source": "web"
  }
}
```

Expected response (recommended):

```json
{
  "replyText": "your reply"
}
```

If your n8n returns a plain string, backend will also handle that.

## Deployment

### Option A (recommended): Any VPS / Docker
Add your own reverse proxy (nginx) and run:
- Backend: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
- Frontend: `npm run build` and serve `frontend/dist/` as static files

### Option B: cPanel Shared Hosting (no Docker)
Most cPanel setups support Python apps via **Passenger**.

1) **Build frontend**

```bash
cd frontend
npm install
npm run build
```

Upload the contents of `frontend/dist/` to your domain’s `public_html/` (or a subfolder).

2) **Deploy backend as a Python app**
- In cPanel → “Setup Python App”
  - Python version: 3.10+
  - App root: point to your uploaded `backend/` folder
  - App startup file: `passenger_wsgi.py`

3) **Set environment variables**
In the Python app config:
- `N8N_WEBHOOK_URL`: your n8n webhook URL
- `ALLOWED_ORIGINS`: your website origin(s), comma separated (e.g. `https://yourdomain.com`)

4) **Install dependencies**
From cPanel’s Python app “pip install” (or terminal):
- Install from `backend/requirements.txt`

5) **Wire frontend → backend**
Set `VITE_API_BASE_URL` in `frontend/.env.production` to your backend URL (e.g. `https://api.yourdomain.com` or same domain path).

## Notes / limitations
- Browser speech recognition support is best in Chromium browsers.
- Server-side transcription is optional and requires extra dependencies; you can disable it by not installing the Whisper extras.


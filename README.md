# SpeechCoach — Local AI Speech Analyser

Analyse your speech from any video or audio recording. Everything runs entirely on your machine — no internet connection required after setup, no API keys, no data leaves your device.

---

## What it does

You upload a video or audio recording of yourself speaking. SpeechCoach transcribes it using OpenAI's Whisper model running locally, then analyses the transcript with a local large language model (Llama 3.2 via Ollama) and gives you a detailed communication coaching report.

### Speech analysis

Your speech is scored across 8 communication dimensions, each rated 0–100 with a Good / Fair / Needs Work label:

| Metric | What it measures |
|---|---|
| **Clarity** | How easy it is to follow what you're saying |
| **Confidence** | Assertiveness, use of filler words, hedging language |
| **Tone** | Appropriate warmth, professionalism, emotional register |
| **Pace** | Speaking speed, pausing, rhythm |
| **Structure** | Logical flow — opening, middle, conclusion |
| **Vocabulary** | Word choice, range, precision |
| **Conciseness** | Getting to the point, avoiding repetition |
| **Engagement** | How compelling and interesting you sound |

### Full report includes

- **Overall score** (0–100) with a 3-sentence summary
- **Per-metric feedback** — 2 sentences of specific, actionable feedback for each dimension
- **Per-metric practice exercise** — one concrete drill to improve that specific area
- **Strengths** — 3 things you did well
- **Improvements** — 3 priority areas to work on
- **Rewritten opening** — your first 3 sentences rewritten to model best practice
- **5-day practice plan** — a structured set of daily 10-minute exercises tailored to your weakest areas

### Scenario contexts

The LLM analyses your speech in the context of what you were actually doing. Built-in scenarios:

- Job interview
- Team presentation
- Salary negotiation
- Startup pitch
- Difficult conversation
- Custom — describe any scenario yourself

---

## Requirements

| Requirement | Notes |
|---|---|
| Python 3.9 or later | Check with `python --version` |
| ffmpeg | For decoding video and audio files |
| ~4 GB free disk space | For Whisper model (~150 MB) and Llama 3.2 (~2 GB) |
| macOS, Linux, or Windows | Ollama auto-installs on macOS/Linux |

### Install ffmpeg

**macOS**
```bash
brew install ffmpeg
```

**Ubuntu / Debian**
```bash
sudo apt install ffmpeg
```

**Windows**
Download from https://ffmpeg.org/download.html and add to your system PATH.

---

## How to start

```bash
python start.py
```

That is the only command you need. The script handles everything automatically on first run:

1. Installs Python packages (`faster-whisper`, `fastapi`, `uvicorn`, `httpx`) via pip
2. Checks that ffmpeg is available
3. Installs Ollama if not present (macOS and Linux only — Windows requires manual install)
4. Downloads `llama3.2` (~2 GB, one-time)
5. Starts the backend API on `http://localhost:8765`
6. Starts the frontend on `http://localhost:8766`
7. Opens your browser automatically

On subsequent runs, startup takes about 5–10 seconds.

To stop the app, press `Ctrl+C` in the terminal.

---

## How to use it

1. **Choose a scenario** — select the context that matches your recording, or write a custom one
2. **Upload your file** — drag and drop, or click to browse. Supported formats: MP4, MOV, WebM, AVI, MP3, WAV, M4A, FLAC
3. **Review the transcript** — Whisper generates it automatically. Edit anything it got wrong before continuing
4. **Click Analyse** — the local LLM processes the transcript and generates your full report (takes 20–60 seconds depending on your machine)
5. **Read your report** — scores, feedback, practice exercises, and your 5-day plan are all shown on the same page

---

## Supported file formats

| Type | Formats |
|---|---|
| Video | MP4, MOV, WebM, AVI, MKV |
| Audio | MP3, WAV, M4A, FLAC, OGG, OPUS |

Files are processed locally. Nothing is uploaded anywhere.

---

## Configuration

Open `start.py` in any text editor. The two settings at the top of the file control model selection:

```python
OLLAMA_MODEL  = "llama3.2"   # LLM used for analysis
WHISPER_MODEL = "base"       # Whisper model for transcription
```

### Whisper model options

Larger models are more accurate but slower. The model is downloaded once and cached.

| Model | Size | Speed | Best for |
|---|---|---|---|
| `tiny` | ~40 MB | Very fast | Quick tests, good audio quality |
| `base` | ~75 MB | Fast | Default — good balance |
| `small` | ~240 MB | Moderate | Better accuracy, accents |
| `medium` | ~770 MB | Slow | High accuracy, noisy audio |
| `large-v2` | ~1.5 GB | Very slow | Best possible accuracy |

### Ollama model options

These are downloaded via `ollama pull <model>` before changing the setting.

| Model | Size | Notes |
|---|---|---|
| `llama3.2` | ~2 GB | Default — fast, good quality |
| `llama3.1` | ~4.7 GB | More thorough analysis |
| `mistral` | ~4.1 GB | Good alternative |
| `gemma2` | ~5.5 GB | Google's model, strong reasoning |

---

## Ports

| Service | Default port |
|---|---|
| Frontend (browser UI) | 8766 |
| Backend API | 8765 |
| Ollama | 11434 (managed by Ollama) |

If either port is already in use, change `PORT_BACKEND` or `PORT_FRONTEND` near the top of `start.py`.

---

## Troubleshooting

**"Backend not connected" shown in the browser**
The backend may still be starting up. Wait 10 seconds and refresh. If it persists, check the terminal for error messages.

**Transcription fails with a format error**
Make sure ffmpeg is installed and accessible on your PATH. Run `ffmpeg -version` in your terminal to verify.

**Ollama not found on Windows**
Download and install from https://ollama.com/download, then re-run `start.py`.

**Analysis takes a very long time**
This is normal on machines without a GPU. `llama3.2` on CPU takes 30–90 seconds. Switching to `OLLAMA_MODEL = "llama3.2:1b"` (a smaller variant) will be faster at the cost of some quality.

**Port already in use**
Change `PORT_BACKEND` and/or `PORT_FRONTEND` at the top of `start.py` to any unused port numbers.

**Whisper transcription is inaccurate**
Try upgrading to a larger model by changing `WHISPER_MODEL = "small"` or `"medium"` in `start.py`. Also check that your audio is reasonably clear and not heavily compressed.

---

## Privacy

- All processing happens on your machine
- No audio, video, or transcript data is sent anywhere
- No analytics, telemetry, or network calls are made during analysis
- The only internet access is during first-time setup (downloading models)

---

## Project structure

```
start.py        ← the only file you need to run
README.md       ← this file
```

The entire application — backend API, frontend UI, and all configuration — is contained in `start.py`.

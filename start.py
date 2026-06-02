#!/usr/bin/env python3
"""
SpeechCoach - Local AI Speech Analysis
=======================================
Single-file launcher. Run with:  python start.py
Requires: Python 3.9+  |  ffmpeg  |  Ollama (auto-installed if missing)
"""

import sys, os, subprocess, threading, time, json, re, shutil, platform, textwrap, socket
from pathlib import Path

ROOT          = Path(__file__).parent
PORT_BACKEND  = 8765
PORT_FRONTEND = 8766
OLLAMA_MODEL  = "llama3.2"   # ~2 GB, fast, good quality
WHISPER_MODEL = "base"       # tiny/base/small/medium

# ─── check port availability ──────────────────────────────────────────────────
def port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False

# ─── Step 1: pip packages ─────────────────────────────────────────────────────
REQUIRED = [
    ("faster_whisper", "faster-whisper"),
    ("fastapi",        "fastapi"),
    ("uvicorn",        "uvicorn[standard]"),
    ("httpx",          "httpx"),
    ("multipart",      "python-multipart"),
]

def ensure_packages():
    missing = []
    for import_name, pip_name in REQUIRED:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"📦  Installing: {', '.join(missing)}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", *missing],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print("❌  pip install failed:\n" + result.stderr)
            raise SystemExit(1)
        print("✅  Packages installed")
        # re-import so they're available in this process
        import importlib
        for import_name, _ in REQUIRED:
            importlib.import_module(import_name)
    else:
        print("✅  Python packages OK")

# ─── Step 2: ffmpeg ───────────────────────────────────────────────────────────
def ensure_ffmpeg():
    if shutil.which("ffmpeg"):
        print("✅  ffmpeg OK")
        return
    system = platform.system()
    print("⚠️   ffmpeg not found — attempting install…")
    if system == "Darwin":
        subprocess.run(["brew", "install", "ffmpeg"], check=True)
    elif system == "Linux":
        subprocess.run(["sudo", "apt-get", "install", "-y", "ffmpeg"], check=True)
    else:
        print("   Install ffmpeg from https://ffmpeg.org/download.html then re-run.")
        raise SystemExit(1)
    if shutil.which("ffmpeg"):
        print("✅  ffmpeg installed")
    else:
        print("⚠️   ffmpeg not on PATH — some formats may fail")

# ─── Step 3: Ollama ───────────────────────────────────────────────────────────
def ensure_ollama():
    # install binary if missing
    if not shutil.which("ollama"):
        print("📥  Ollama not found — installing…")
        if platform.system() in ("Darwin", "Linux"):
            subprocess.run("curl -fsSL https://ollama.com/install.sh | sh",
                           shell=True, check=True)
        else:
            print("   Install Ollama from https://ollama.com/download then re-run.")
            raise SystemExit(1)

    # make sure ollama serve is running
    import httpx
    running = False
    for _ in range(3):
        try:
            httpx.get("http://localhost:11434", timeout=2)
            running = True
            break
        except Exception:
            pass
    if not running:
        print("🚀  Starting Ollama…")
        subprocess.Popen(["ollama", "serve"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # wait up to 10 s for it to come up
        for _ in range(20):
            time.sleep(0.5)
            try:
                httpx.get("http://localhost:11434", timeout=1)
                break
            except Exception:
                pass
    print("✅  Ollama running")

    # pull model if not present
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if OLLAMA_MODEL not in result.stdout:
        print(f"📥  Pulling {OLLAMA_MODEL} (one-time, ~2 GB)…")
        subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)
    print(f"✅  Ollama model '{OLLAMA_MODEL}' ready")

# ─────────────────────────────────────────────────────────────────────────────
# BACKEND  (execd in-process so we don't need a separate file)
# ─────────────────────────────────────────────────────────────────────────────
BACKEND_CODE = r'''
import os, json, re, tempfile, traceback, time
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import httpx

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
OLLAMA_MODEL  = os.environ.get("OLLAMA_MODEL",  "llama3.2")
OLLAMA_URL    = "http://localhost:11434/api/generate"

app = FastAPI(title="SpeechCoach API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_whisper = None
def get_whisper():
    global _whisper
    if _whisper is None:
        print(f"[whisper] loading '{WHISPER_MODEL}'…", flush=True)
        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        print("[whisper] ready", flush=True)
    return _whisper

# pre-load whisper at startup so /health is only green when it's truly ready
@app.on_event("startup")
async def startup():
    get_whisper()

ANALYSIS_PROMPT = """You are an expert communication coach with 20 years of experience.
Analyse the speech transcript below given the scenario context.
Return ONLY a valid JSON object — no markdown fences, no extra text.

SCENARIO: {scenario}

TRANSCRIPT:
{transcript}

Return exactly this JSON structure (all fields required):
{{
  "overall_score": <0-100>,
  "summary": "<3 sentence overall assessment>",
  "metrics": {{
    "clarity":        {{"score":<0-100>, "label":"<Good|Fair|Needs work>", "feedback":"<2 sentences>", "tip":"<1 concrete practice exercise>"}},
    "confidence":     {{"score":<0-100>, "label":"<Good|Fair|Needs work>", "feedback":"<2 sentences>", "tip":"<1 concrete practice exercise>"}},
    "tone":           {{"score":<0-100>, "label":"<Good|Fair|Needs work>", "feedback":"<2 sentences>", "tip":"<1 concrete practice exercise>"}},
    "pace":           {{"score":<0-100>, "label":"<Good|Fair|Needs work>", "feedback":"<2 sentences>", "tip":"<1 concrete practice exercise>"}},
    "structure":      {{"score":<0-100>, "label":"<Good|Fair|Needs work>", "feedback":"<2 sentences>", "tip":"<1 concrete practice exercise>"}},
    "vocabulary":     {{"score":<0-100>, "label":"<Good|Fair|Needs work>", "feedback":"<2 sentences>", "tip":"<1 concrete practice exercise>"}},
    "conciseness":    {{"score":<0-100>, "label":"<Good|Fair|Needs work>", "feedback":"<2 sentences>", "tip":"<1 concrete practice exercise>"}},
    "engagement":     {{"score":<0-100>, "label":"<Good|Fair|Needs work>", "feedback":"<2 sentences>", "tip":"<1 concrete practice exercise>"}}
  }},
  "strengths":    ["<strength 1>", "<strength 2>", "<strength 3>"],
  "improvements": ["<improvement 1>", "<improvement 2>", "<improvement 3>"],
  "rewrite": "<rewrite the opening 3 sentences to model best practice>",
  "practice_plan": [
    {{"day": 1, "exercise": "<specific 10-min exercise>", "goal": "<what it improves>"}},
    {{"day": 2, "exercise": "<specific 10-min exercise>", "goal": "<what it improves>"}},
    {{"day": 3, "exercise": "<specific 10-min exercise>", "goal": "<what it improves>"}},
    {{"day": 4, "exercise": "<specific 10-min exercise>", "goal": "<what it improves>"}},
    {{"day": 5, "exercise": "<specific 10-min exercise>", "goal": "<what it improves>"}}
  ]
}}"""

@app.get("/health")
def health():
    # only returns 200 once whisper is loaded (startup already called get_whisper)
    return {"status": "ok", "whisper": WHISPER_MODEL, "ollama": OLLAMA_MODEL}

@app.post("/transcribe")
async def transcribe(request: Request):
    # Read filename from header, fall back to .mp4
    filename = request.headers.get("X-Filename", "upload.mp4")
    suffix = Path(filename).suffix or ".mp4"
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file body")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        model = get_whisper()
        t0 = time.time()
        segments, info = model.transcribe(tmp_path, beam_size=5,
                                          language=None, task="transcribe")
        text = " ".join(seg.text.strip() for seg in segments)
        elapsed = round(time.time() - t0, 1)
        return {"transcript": text, "language": info.language,
                "duration": round(info.duration, 1), "elapsed": elapsed}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

@app.post("/analyse")
async def analyse(request: Request):
    body = await request.json()
    transcript = body.get("transcript", "")
    scenario   = body.get("scenario", "General presentation")
    if not transcript:
        raise HTTPException(status_code=400, detail="transcript is required")
    prompt = ANALYSIS_PROMPT.format(scenario=scenario, transcript=transcript)
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 2048}
            })
        raw = resp.json().get("response", "")
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON in LLM response:\n" + raw[:500])
        return json.loads(match.group())
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
'''

# ─────────────────────────────────────────────────────────────────────────────
# FRONTEND  — light theme
# ─────────────────────────────────────────────────────────────────────────────
FRONTEND_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SpeechCoach</title>
<style>
  :root {
    --bg:      #f5f5f0;
    --bg2:     #ffffff;
    --bg3:     #f0eeea;
    --border:  #e0ddd8;
    --text:    #1a1a18;
    --muted:   #6b6b66;
    --amber:   #c47d0e;
    --amber-l: #fdf3e0;
    --pink:    #c2185b;
    --teal:    #0077a8;
    --green:   #1a7a4a;
    --green-l: #e8f7ef;
    --purple:  #5e35b1;
    --purple-l:#f0ecfc;
    --red:     #c62828;
    --red-l:   #fdecea;
    --blue:    #1565c0;
    --orange:  #c84b00;
    --cyan:    #006b7a;
    --rose:    #ad1457;
  }
  * { box-sizing:border-box; margin:0; padding:0 }
  body { background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; min-height:100vh; font-size:15px }

  /* ── Layout ── */
  header { background:var(--bg2); border-bottom:1px solid var(--border); padding:14px 28px; display:flex; align-items:center; justify-content:space-between; position:sticky; top:0; z-index:10 }
  .logo { font-size:20px; font-weight:800; letter-spacing:-0.5px; color:var(--text) }
  .logo span { color:var(--amber) }
  .badge { font-size:11px; color:var(--muted); margin-top:2px }
  .local-tag { font-size:11px; color:var(--green); background:var(--green-l); border:1px solid #b2dfcc; border-radius:20px; padding:3px 10px; font-weight:600 }
  main { max-width:880px; margin:0 auto; padding:28px 20px 80px }

  /* ── Cards ── */
  .card { background:var(--bg2); border:1px solid var(--border); border-radius:14px; padding:22px 24px; margin-bottom:18px }
  h2 { font-size:16px; font-weight:700; margin-bottom:14px; color:var(--text) }
  h3 { font-size:14px; font-weight:600; color:var(--text) }

  /* ── Status banner ── */
  #status-banner { display:flex; align-items:center; gap:10px; padding:10px 16px; border-radius:10px; font-size:13px; margin-bottom:18px; background:#fff8e1; border:1px solid #ffe082; color:#7a5c00 }
  #status-banner.ok { background:var(--green-l); border-color:#b2dfcc; color:#1a5c38 }
  #status-banner.err { background:var(--red-l); border-color:#ffcdd2; color:var(--red) }
  .dot { width:8px; height:8px; border-radius:50%; flex-shrink:0 }
  .dot.spin { animation:pulse 1.2s ease-in-out infinite }
  .dot.yellow { background:#f9a825 }
  .dot.green  { background:var(--green) }
  .dot.red    { background:var(--red) }

  /* ── Scenarios ── */
  .scenarios { display:grid; grid-template-columns:repeat(auto-fill,minmax(148px,1fr)); gap:10px }
  .scenario-btn { background:var(--bg3); border:1.5px solid var(--border); border-radius:10px; padding:14px 12px; cursor:pointer; text-align:left; transition:all .15s }
  .scenario-btn:hover { border-color:var(--amber); background:var(--amber-l) }
  .scenario-btn.active { border-color:var(--amber); background:var(--amber-l) }
  .scenario-btn .icon { font-size:20px; margin-bottom:8px }
  .scenario-btn .name { font-size:13px; font-weight:600; color:var(--text) }
  .scenario-btn .desc { font-size:11px; color:var(--muted); margin-top:3px; line-height:1.4 }
  #custom-prompt { display:none; width:100%; margin-top:12px; background:var(--bg3); border:1.5px solid var(--border); border-radius:8px; color:var(--text); font-size:13px; padding:10px 12px; resize:none; font-family:inherit }
  #custom-prompt:focus { outline:none; border-color:var(--amber) }

  /* ── Upload ── */
  .upload-zone { border:2px dashed #c8c4bc; border-radius:12px; padding:44px 20px; text-align:center; cursor:pointer; transition:all .2s; background:var(--bg3) }
  .upload-zone:hover, .upload-zone.drag { border-color:var(--amber); background:var(--amber-l) }
  .upload-zone .uicon { font-size:42px; margin-bottom:12px }
  .upload-zone .utitle { font-size:15px; font-weight:600; color:var(--amber) }
  .upload-zone .usub { font-size:12px; color:var(--muted); margin-top:6px; line-height:1.5 }
  #file-input { display:none }

  /* ── Progress ── */
  .progress-wrap { display:none; margin-top:16px }
  .progress-label { font-size:13px; color:var(--muted); margin-bottom:8px }
  .progress-bar { background:var(--bg3); border-radius:4px; height:6px; overflow:hidden; border:1px solid var(--border) }
  .progress-fill { height:100%; background:linear-gradient(90deg,var(--amber),var(--pink)); border-radius:4px; transition:width .4s }
  .status-msg { font-size:13px; color:var(--purple); margin-top:8px; min-height:20px }

  /* ── Media players ── */
  #video-preview { width:100%; border-radius:10px; margin-top:14px; display:none; max-height:280px; border:1px solid var(--border) }
  #audio-preview { width:100%; margin-top:14px; display:none }

  /* ── Transcript ── */
  #transcript-section { display:none }
  .section-label { font-size:11px; font-weight:600; letter-spacing:.7px; text-transform:uppercase; color:var(--muted); margin-bottom:8px }
  .transcript-box { width:100%; min-height:100px; max-height:220px; overflow-y:auto; background:var(--bg3); border:1.5px solid var(--border); border-radius:8px; color:var(--text); font-size:13px; padding:12px 14px; resize:vertical; line-height:1.7; font-family:inherit }
  .transcript-box:focus { outline:none; border-color:var(--amber) }
  .tx-meta { font-size:11px; color:var(--muted); margin-top:6px }

  /* ── Analyse button ── */
  #analyse-btn { display:none; width:100%; margin-top:14px; padding:14px; background:var(--amber); border:none; border-radius:10px; color:#fff; font-size:15px; font-weight:700; cursor:pointer; transition:opacity .15s; font-family:inherit }
  #analyse-btn:hover { opacity:.88 }
  #analyse-btn:disabled { opacity:.45; cursor:default }

  /* ── Error ── */
  .error-box { background:var(--red-l); border:1px solid #ffcdd2; border-radius:8px; padding:12px 14px; font-size:13px; color:var(--red); margin-top:10px; display:none }

  /* ── Results ── */
  #results-section { display:none }
  .overall-card { background:linear-gradient(135deg,#fff8e1,#fff0f6); border:1.5px solid #ffe082; border-radius:16px; padding:28px; text-align:center; margin-bottom:20px }
  .overall-score { font-size:76px; font-weight:800; color:var(--amber); line-height:1 }
  .overall-verdict { color:var(--muted); font-size:14px; line-height:1.7; margin-top:10px; max-width:560px; margin-left:auto; margin-right:auto }

  /* ── Metrics grid ── */
  .metrics-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:20px }
  @media(max-width:540px){ .metrics-grid { grid-template-columns:1fr } }
  .metric-card { background:var(--bg2); border:1.5px solid var(--border); border-radius:12px; padding:16px }
  .metric-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:7px }
  .metric-name { font-size:13px; font-weight:600; display:flex; align-items:center; gap:7px; color:var(--text) }
  .metric-score { font-size:22px; font-weight:800; font-family:monospace }
  .bar-bg { background:var(--bg3); border-radius:3px; height:6px; overflow:hidden; margin-bottom:9px; border:1px solid var(--border) }
  .bar-fill { height:100%; border-radius:3px; transition:width 1.1s cubic-bezier(.16,1,.3,1) }
  .metric-feedback { font-size:12px; color:var(--muted); line-height:1.55; margin-bottom:9px }
  .metric-tip { font-size:12px; color:var(--purple); background:var(--purple-l); border:1px solid #d1c4f0; border-radius:7px; padding:8px 10px; line-height:1.5 }
  .tip-label { font-weight:700; margin-bottom:3px }

  .label-badge { font-size:10px; font-weight:700; padding:2px 7px; border-radius:4px; text-transform:uppercase; letter-spacing:.4px }
  .label-good  { background:#e8f5e9; color:#2e7d32; border:1px solid #a5d6a7 }
  .label-fair  { background:#fff8e1; color:#f57f17; border:1px solid #ffe082 }
  .label-needs { background:var(--red-l); color:var(--red); border:1px solid #ffcdd2 }

  /* ── Strengths / Improvements ── */
  .two-col { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:20px }
  @media(max-width:540px){ .two-col { grid-template-columns:1fr } }
  .str-card { border-radius:12px; padding:16px }
  .str-card.good { background:var(--green-l); border:1.5px solid #b2dfcc }
  .str-card.imp  { background:#fce4ec; border:1.5px solid #f48fb1 }
  .str-title { font-size:11px; font-weight:700; letter-spacing:.7px; text-transform:uppercase; margin-bottom:10px }
  .str-card.good .str-title { color:var(--green) }
  .str-card.imp  .str-title { color:var(--pink) }
  .str-list { list-style:none }
  .str-list li { font-size:13px; color:var(--text); line-height:1.55; margin-bottom:7px; padding-left:16px; position:relative }
  .str-list li::before { content:"▸"; position:absolute; left:0; opacity:.5 }

  /* ── Rewrite ── */
  .rewrite-card { background:var(--purple-l); border:1.5px solid #d1c4f0; border-radius:12px; padding:18px; margin-bottom:20px }
  .rewrite-label { font-size:11px; font-weight:700; letter-spacing:.7px; text-transform:uppercase; color:var(--purple); margin-bottom:8px }
  .rewrite-text { font-size:13px; color:#37286b; line-height:1.7; font-style:italic }

  /* ── Practice plan ── */
  .plan-grid { display:flex; flex-direction:column; gap:9px; margin-top:10px }
  .plan-day { background:var(--bg3); border:1px solid var(--border); border-radius:9px; padding:13px 15px; display:flex; gap:12px; align-items:flex-start }
  .plan-day-num { background:var(--amber); color:#fff; font-size:11px; font-weight:800; border-radius:6px; padding:3px 9px; white-space:nowrap; margin-top:2px }
  .plan-day-ex { font-size:13px; color:var(--text); line-height:1.5 }
  .plan-day-goal { font-size:11px; color:var(--muted); margin-top:4px }

  /* ── Reset ── */
  #reset-btn { display:none; width:100%; margin-top:14px; padding:12px; background:var(--bg2); border:1.5px solid var(--border); border-radius:10px; color:var(--muted); font-size:14px; cursor:pointer; font-family:inherit; transition:all .15s }
  #reset-btn:hover { border-color:var(--amber); color:var(--amber) }

  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
</style>
</head>
<body>

<header>
  <div>
    <div class="logo"><span>Speech</span>Coach</div>
    <div class="badge" id="header-badge">Connecting…</div>
  </div>
  <div class="local-tag">100 % local · no internet</div>
</header>

<main>

  <!-- Status banner -->
  <div id="status-banner">
    <div class="dot spin yellow" id="status-dot"></div>
    <span id="status-text">Connecting to backend — loading Whisper model, please wait…</span>
  </div>

  <!-- Scenario -->
  <div class="card">
    <h2>Choose a scenario</h2>
    <div class="scenarios" id="scenario-list"></div>
    <textarea id="custom-prompt" rows="3" placeholder="Describe your scenario…"></textarea>
  </div>

  <!-- Upload -->
  <div class="card">
    <h2>Upload your recording</h2>
    <div class="upload-zone" id="upload-zone">
      <div class="uicon">🎬</div>
      <div class="utitle">Drop video or audio here, or click to browse</div>
      <div class="usub">MP4 · MOV · WebM · AVI · MP3 · WAV · M4A · FLAC</div>
      <div class="usub">Processed entirely on your machine</div>
    </div>
    <input type="file" id="file-input" accept="video/*,audio/*,.mp3,.wav,.m4a,.flac">

    <video id="video-preview" controls></video>
    <audio id="audio-preview" controls></audio>

    <div class="progress-wrap" id="progress-wrap">
      <div class="progress-label" id="progress-label">Transcribing with Whisper…</div>
      <div class="progress-bar"><div class="progress-fill" id="progress-fill" style="width:0%"></div></div>
      <div class="status-msg" id="status-msg"></div>
    </div>
    <div class="error-box" id="error-box"></div>
  </div>

  <!-- Transcript review -->
  <div class="card" id="transcript-section">
    <h2>Transcript</h2>
    <div class="section-label">Whisper-generated — edit anything that's wrong</div>
    <textarea class="transcript-box" id="transcript-box" rows="6"></textarea>
    <div class="tx-meta" id="tx-meta"></div>
    <button id="analyse-btn" onclick="runAnalysis()">✨ Analyse my speech</button>
  </div>

  <!-- Results -->
  <div id="results-section">

    <div class="overall-card">
      <div class="section-label" style="color:var(--muted)">Overall score</div>
      <div class="overall-score" id="res-score">--</div>
      <div class="overall-verdict" id="res-verdict"></div>
    </div>

    <div class="card" style="margin-bottom:18px">
      <h2>Communication metrics</h2>
      <div class="metrics-grid" id="metrics-grid"></div>
    </div>

    <div class="two-col">
      <div class="str-card good">
        <div class="str-title">✅ Strengths</div>
        <ul class="str-list" id="res-strengths"></ul>
      </div>
      <div class="str-card imp">
        <div class="str-title">🎯 To improve</div>
        <ul class="str-list" id="res-improvements"></ul>
      </div>
    </div>

    <div class="rewrite-card">
      <div class="rewrite-label">✍️ Stronger opening — modelled for you</div>
      <div class="rewrite-text" id="res-rewrite"></div>
    </div>

    <div class="card">
      <h2>5-day practice plan</h2>
      <div class="plan-grid" id="plan-grid"></div>
    </div>

    <button id="reset-btn" onclick="resetAll()">↩ Analyse another recording</button>
  </div>

</main>

<script>
const API = 'http://localhost:''' + str(PORT_BACKEND) + r'''';

// ── Scenarios ──────────────────────────────────────────────────────────────
const SCENARIOS = [
  { id:'interview',    icon:'💼', name:'Job interview',         desc:'Introduce yourself & sell your skills', prompt:'You are in a job interview for a senior software engineering role.' },
  { id:'presentation', icon:'📊', name:'Team presentation',     desc:'Present a project to your team',        prompt:'You are presenting your latest project to your team.' },
  { id:'negotiation',  icon:'💰', name:'Salary negotiation',    desc:'Make the case for a raise',             prompt:'You are negotiating a 20% salary increase with your manager.' },
  { id:'pitch',        icon:'🚀', name:'Startup pitch',         desc:'Pitch to investors',                    prompt:'You are pitching your startup idea to a panel of venture capitalists.' },
  { id:'conflict',     icon:'🤝', name:'Difficult conversation',desc:'Address a workplace conflict',          prompt:'You are addressing a conflict with a colleague about missed deadlines.' },
  { id:'custom',       icon:'✏️', name:'Custom',                desc:'Describe your own scenario',            prompt:'' },
];

let selectedScenario = SCENARIOS[0];

const list = document.getElementById('scenario-list');
SCENARIOS.forEach(s => {
  const btn = document.createElement('button');
  btn.className = 'scenario-btn' + (s.id === 'interview' ? ' active' : '');
  btn.dataset.id = s.id;
  btn.innerHTML = `<div class="icon">${s.icon}</div><div class="name">${s.name}</div><div class="desc">${s.desc}</div>`;
  btn.onclick = () => {
    document.querySelectorAll('.scenario-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    selectedScenario = s;
    document.getElementById('custom-prompt').style.display = s.id === 'custom' ? 'block' : 'none';
  };
  list.appendChild(btn);
});

// ── Health check with retry ────────────────────────────────────────────────
// Backend takes time to load Whisper — poll until it's actually ready
let backendReady = false;
async function pollHealth() {
  const banner  = document.getElementById('status-banner');
  const dot     = document.getElementById('status-dot');
  const txt     = document.getElementById('status-text');
  const hbadge  = document.getElementById('header-badge');

  for (let attempt = 0; attempt < 120; attempt++) {   // up to 2 min
    try {
      const r = await fetch(API + '/health', { signal: AbortSignal.timeout(3000) });
      if (r.ok) {
        const d = await r.json();
        backendReady = true;
        banner.className = 'ok';
        dot.className = 'dot green';
        txt.textContent = `Ready  ·  Whisper ${d.whisper}  ·  ${d.ollama}`;
        hbadge.textContent = `✅ Backend ready`;
        hbadge.style.color = '#1a7a4a';
        return;
      }
    } catch {}
    // still waiting
    const secs = (attempt + 1) * 2;
    txt.textContent = `Loading Whisper model… (${secs}s) — this is a one-time wait`;
    await new Promise(r => setTimeout(r, 2000));
  }
  // timed out
  banner.className = 'err';
  dot.className = 'dot red';
  txt.textContent = 'Backend not responding — check your terminal for errors';
  hbadge.textContent = '⚠️ Backend error';
}
pollHealth();

// ── Upload & drag-drop ─────────────────────────────────────────────────────
const zone = document.getElementById('upload-zone');
zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('drag'); });
zone.addEventListener('dragleave', () => zone.classList.remove('drag'));
zone.addEventListener('drop',      e => { e.preventDefault(); zone.classList.remove('drag'); handleFile(e.dataTransfer.files[0]); });
zone.addEventListener('click',     () => document.getElementById('file-input').click());
document.getElementById('file-input').addEventListener('change', e => handleFile(e.target.files[0]));

function showError(msg) {
  const b = document.getElementById('error-box');
  b.textContent = '⚠️ ' + msg; b.style.display = 'block';
}
function hideError() { document.getElementById('error-box').style.display = 'none'; }

async function handleFile(file) {
  if (!file) return;
  if (!backendReady) { showError('Backend is still loading — please wait for the green status banner.'); return; }
  hideError();

  const url = URL.createObjectURL(file);
  if (file.type.startsWith('audio/')) {
    document.getElementById('audio-preview').src = url;
    document.getElementById('audio-preview').style.display = 'block';
    document.getElementById('video-preview').style.display = 'none';
  } else {
    document.getElementById('video-preview').src = url;
    document.getElementById('video-preview').style.display = 'block';
    document.getElementById('audio-preview').style.display = 'none';
  }

  const pw = document.getElementById('progress-wrap');
  pw.style.display = 'block';
  document.getElementById('transcript-section').style.display = 'none';
  document.getElementById('analyse-btn').style.display = 'none';
  document.getElementById('results-section').style.display = 'none';
  document.getElementById('reset-btn').style.display = 'none';
  setProgress(10, 'Sending to local backend…');

  try {
    setProgress(30, 'Whisper is transcribing your audio…');
    const r = await fetch(API + '/transcribe', {
      method: 'POST',
      headers: { 'X-Filename': file.name, 'Content-Type': 'application/octet-stream' },
      body: file
    });
    if (!r.ok) { const e = await r.json(); throw new Error(e.detail || 'Transcription failed'); }
    const data = await r.json();
    setProgress(100, '✅ Done');
    setTimeout(() => pw.style.display = 'none', 700);

    document.getElementById('transcript-box').value = data.transcript;
    document.getElementById('tx-meta').textContent =
      `Language: ${data.language}  ·  Duration: ${data.duration}s  ·  Transcribed in ${data.elapsed}s`;
    document.getElementById('transcript-section').style.display = 'block';
    document.getElementById('analyse-btn').style.display = 'block';
  } catch(e) {
    pw.style.display = 'none';
    showError(e.message);
  }
}

function setProgress(pct, msg) {
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('status-msg').textContent = msg;
}

// ── Analysis ───────────────────────────────────────────────────────────────
async function runAnalysis() {
  const transcript = document.getElementById('transcript-box').value.trim();
  if (!transcript) { showError('Transcript is empty.'); return; }
  const btn = document.getElementById('analyse-btn');
  btn.disabled = true; btn.textContent = '⏳ Analysing with local LLM…';

  const prompt = selectedScenario.id === 'custom'
    ? (document.getElementById('custom-prompt').value.trim() || 'General presentation')
    : selectedScenario.prompt;

  try {
    const r = await fetch(API + '/analyse', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ transcript, scenario: prompt })
    });
    if (!r.ok) { const e = await r.json(); throw new Error(e.detail); }
    renderResults(await r.json());
  } catch(e) {
    showError('Analysis failed: ' + e.message);
  } finally {
    btn.disabled = false; btn.textContent = '✨ Analyse my speech';
  }
}

// ── Render results ─────────────────────────────────────────────────────────
const METRIC_META = {
  clarity:     { icon:'🔍', color:'#0077a8' },
  confidence:  { icon:'💪', color:'#c2185b' },
  tone:        { icon:'🎭', color:'#c47d0e' },
  pace:        { icon:'⚡', color:'#5e35b1' },
  structure:   { icon:'🏗️', color:'#1a7a4a' },
  vocabulary:  { icon:'📚', color:'#c84b00' },
  conciseness: { icon:'✂️', color:'#006b7a' },
  engagement:  { icon:'🔥', color:'#ad1457' },
};

function labelClass(l) {
  if (!l) return '';
  const s = l.toLowerCase();
  if (s.includes('good') || s.includes('excellent')) return 'label-good';
  if (s.includes('fair') || s.includes('average'))   return 'label-fair';
  return 'label-needs';
}

function renderResults(data) {
  document.getElementById('results-section').style.display = 'block';
  document.getElementById('reset-btn').style.display = 'block';
  document.getElementById('res-score').textContent = data.overall_score ?? '--';
  document.getElementById('res-verdict').textContent = data.summary ?? '';

  const grid = document.getElementById('metrics-grid');
  grid.innerHTML = '';
  Object.entries(data.metrics || {}).forEach(([key, m]) => {
    const meta = METRIC_META[key] || { icon:'📊', color:'#555' };
    const card = document.createElement('div');
    card.className = 'metric-card';
    card.innerHTML = `
      <div class="metric-header">
        <div class="metric-name">${meta.icon} ${key.charAt(0).toUpperCase()+key.slice(1)}
          <span class="label-badge ${labelClass(m.label)}">${m.label||''}</span>
        </div>
        <div class="metric-score" style="color:${meta.color}">${m.score}</div>
      </div>
      <div class="bar-bg"><div class="bar-fill" style="width:0%;background:${meta.color}" data-score="${m.score}"></div></div>
      <div class="metric-feedback">${m.feedback||''}</div>
      <div class="metric-tip"><div class="tip-label">💡 Practice</div>${m.tip||''}</div>`;
    grid.appendChild(card);
  });
  requestAnimationFrame(() =>
    document.querySelectorAll('.bar-fill').forEach(b => b.style.width = b.dataset.score + '%')
  );

  document.getElementById('res-strengths').innerHTML   = (data.strengths||[]).map(s=>`<li>${s}</li>`).join('');
  document.getElementById('res-improvements').innerHTML = (data.improvements||[]).map(s=>`<li>${s}</li>`).join('');
  document.getElementById('res-rewrite').textContent = data.rewrite || '';

  document.getElementById('plan-grid').innerHTML = (data.practice_plan||[]).map(p=>`
    <div class="plan-day">
      <div class="plan-day-num">Day ${p.day}</div>
      <div>
        <div class="plan-day-ex">${p.exercise}</div>
        <div class="plan-day-goal">Goal: ${p.goal}</div>
      </div>
    </div>`).join('');

  document.getElementById('results-section').scrollIntoView({ behavior:'smooth' });
}

function resetAll() {
  ['video-preview','audio-preview','transcript-section','results-section','reset-btn']
    .forEach(id => document.getElementById(id).style.display = 'none');
  document.getElementById('analyse-btn').style.display = 'none';
  document.getElementById('transcript-box').value = '';
  document.getElementById('file-input').value = '';
  hideError();
  window.scrollTo({ top:0, behavior:'smooth' });
}
</script>
</body>
</html>
'''

# ─────────────────────────────────────────────────────────────────────────────
# RUNNERS
# ─────────────────────────────────────────────────────────────────────────────
def run_backend():
    import uvicorn, types
    if not port_free(PORT_BACKEND):
        print(f"❌  Port {PORT_BACKEND} is already in use.")
        print(f"   Change PORT_BACKEND in start.py or free the port and retry.")
        raise SystemExit(1)

    mod = types.ModuleType("speechcoach_backend")
    exec(compile(BACKEND_CODE, "<backend>", "exec"), mod.__dict__)
    sys.modules["speechcoach_backend"] = mod

    os.environ["WHISPER_MODEL"] = WHISPER_MODEL
    os.environ["OLLAMA_MODEL"]  = OLLAMA_MODEL

    uvicorn.run(mod.app, host="127.0.0.1", port=PORT_BACKEND, log_level="info")

def run_frontend():
    import http.server, socketserver
    if not port_free(PORT_FRONTEND):
        print(f"❌  Port {PORT_FRONTEND} is already in use.")
        print(f"   Change PORT_FRONTEND in start.py or free the port and retry.")
        raise SystemExit(1)

    html_bytes = FRONTEND_HTML.encode()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type",   "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.end_headers()
            self.wfile.write(html_bytes)
        def log_message(self, *a): pass

    with socketserver.TCPServer(("127.0.0.1", PORT_FRONTEND), Handler) as srv:
        srv.serve_forever()

def open_browser():
    time.sleep(2)
    import webbrowser
    webbrowser.open(f"http://localhost:{PORT_FRONTEND}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(textwrap.dedent("""
    ╔══════════════════════════════════════╗
    ║        SpeechCoach — Local AI        ║
    ╚══════════════════════════════════════╝
    """))

    print("── Step 1/3  Python packages")
    ensure_packages()

    print("\n── Step 2/3  ffmpeg")
    ensure_ffmpeg()

    print("\n── Step 3/3  Ollama + model")
    ensure_ollama()

    print(f"""
✅  All dependencies satisfied
──────────────────────────────
   App      →  http://localhost:{PORT_FRONTEND}
   API      →  http://localhost:{PORT_BACKEND}
──────────────────────────────
🚀  Starting…  (Ctrl+C to quit)
⏳  Whisper model loads in the background — the banner in the app will turn green when ready.
""")

    threading.Thread(target=run_frontend, daemon=True).start()
    threading.Thread(target=open_browser,  daemon=True).start()

    try:
        run_backend()           # blocks until Ctrl+C
    except KeyboardInterrupt:
        print("\n\n👋  SpeechCoach stopped.")

if __name__ == "__main__":
    main()

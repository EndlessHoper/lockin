import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

LOG_LEVEL = "INFO"
logger = logging.getLogger("whatyousee")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(LOG_LEVEL)
logger.propagate = False

ROOT = Path(__file__).resolve().parent


def _load_dotenv():
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_dotenv()
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")
LMSTUDIO_CHAT_URL = f"{LMSTUDIO_BASE_URL}/chat/completions"
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL") or "google/gemma-3n-e4b"
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "")
LMSTUDIO_TIMEOUT = float(os.getenv("LMSTUDIO_TIMEOUT", "60"))

MAX_TOKENS = 64
TEMPERATURE = 1.0
TOP_K = 64
TOP_P = 0.95
SYSTEM_PROMPT = "You are a concise vision assistant. Reply with one short sentence."
PROMPT = (
    "Describe what you see in the image in a single short sentence. "
    "If the image is unclear, say so briefly."
)

_lock = threading.Lock()
_last_result = None


class ImageRequest(BaseModel):
    image: str


def _init_lmstudio():
    logger.info("Using LM Studio model %s @ %s", LMSTUDIO_MODEL, LMSTUDIO_BASE_URL)


def _normalize_image_url(image_url: str) -> str:
    if image_url.startswith("data:"):
        return image_url
    return f"data:image/jpeg;base64,{image_url}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_lmstudio()
    yield


app = FastAPI(title="WhatYouSee", lifespan=lifespan)


def _clean_description(text: str) -> str:
    if not text:
        return "No response."
    line = text.strip().splitlines()[0].strip()
    if not line:
        return "No response."
    if len(line) > 200:
        return line[:197] + "..."
    return line


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if LMSTUDIO_API_KEY:
        headers["Authorization"] = f"Bearer {LMSTUDIO_API_KEY}"
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(f"LM Studio HTTP {exc.code}: {body}") from exc


def _analyze_lmstudio(image_url: str) -> dict:
    start = time.time()
    image_url = _normalize_image_url(image_url)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]
    payload = {
        "model": LMSTUDIO_MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }
    try:
        response = _post_json(LMSTUDIO_CHAT_URL, payload, timeout=LMSTUDIO_TIMEOUT)
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        raw = (message.get("content") or "").strip()
        detail = _clean_description(raw)
        return {
            "label": "SEEING",
            "distracted": False,
            "reason": "scene",
            "detail": detail,
            "elapsed": round(time.time() - start, 2),
            "raw": raw,
        }
    except Exception as exc:
        logger.error("LM Studio request failed: %s", exc)
        return {
            "label": "ERROR",
            "distracted": False,
            "reason": "error",
            "detail": str(exc),
            "elapsed": 0,
            "raw": "",
        }


@app.get("/")
def index():
    return HTMLResponse(INDEX_HTML)


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/config")
def config():
    return {
        "backend": "lmstudio",
        "model": LMSTUDIO_MODEL,
        "base_url": LMSTUDIO_BASE_URL,
    }


@app.post("/analyze")
def analyze(req: ImageRequest):
    global _last_result
    if not _lock.acquire(blocking=False):
        if _last_result:
            logger.info("Analyze skipped (busy); returning stale result")
            return {**_last_result, "stale": True}
        logger.info("Analyze skipped (busy); no cached result")
        return {"label": "BUSY", "reason": "busy", "detail": "model busy", "stale": True}
    try:
        result = _analyze_lmstudio(req.image)
        _last_result = {**result, "stale": False}
        logger.info(
            "Result %s elapsed=%.2fs",
            result.get("label"),
            result.get("elapsed", 0),
        )
        logger.debug("Raw output: %s", result.get("raw", ""))
        return _last_result
    finally:
        _lock.release()


INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>What You See - Local VLM</title>
    <style>
      @import url("https://fonts.googleapis.com/css2?family=Fraunces:wght@400;600&family=JetBrains+Mono:wght@400;600&family=Sora:wght@400;500;600&display=swap");
      :root {
        --bg: #f2eee8;
        --ink: #1b1d21;
        --muted: #5a6468;
        --accent: #1a5580;
        --accent-2: #c0692a;
        --card: rgba(255, 255, 255, 0.88);
        --border: rgba(27, 29, 33, 0.12);
        --shadow: 0 24px 70px rgba(26, 35, 46, 0.18);
        --grid: rgba(27, 29, 33, 0.06);
        --danger: #b42318;
        --success: #1f7a5a;
        --display: "Fraunces", "Georgia", serif;
        --sans: "Sora", "Avenir Next", "Helvetica Neue", sans-serif;
        --mono: "JetBrains Mono", "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: var(--sans);
        color: var(--ink);
        background: var(--bg);
        min-height: 100vh;
        position: relative;
        overflow-x: hidden;
      }
      body::before {
        content: "";
        position: fixed;
        inset: 0;
        background:
          radial-gradient(circle at 12% 18%, rgba(26, 85, 128, 0.14), transparent 55%),
          radial-gradient(circle at 85% 14%, rgba(192, 105, 42, 0.18), transparent 50%),
          radial-gradient(circle at 70% 78%, rgba(26, 85, 128, 0.08), transparent 55%);
        z-index: -2;
      }
      body::after {
        content: "";
        position: fixed;
        inset: 0;
        background-image:
          linear-gradient(to right, var(--grid) 1px, transparent 1px),
          linear-gradient(to bottom, var(--grid) 1px, transparent 1px);
        background-size: 88px 88px;
        opacity: 0.35;
        z-index: -1;
        pointer-events: none;
      }
      .shell {
        max-width: 1240px;
        margin: 0 auto;
        padding: 32px 20px 60px;
        display: grid;
        gap: 28px;
      }
      .topbar {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 24px;
        flex-wrap: wrap;
      }
      .kicker {
        text-transform: uppercase;
        letter-spacing: 0.22em;
        font-size: 11px;
        color: var(--muted);
      }
      h1 {
        margin: 8px 0 10px;
        font-family: var(--display);
        font-size: clamp(34px, 4.6vw, 58px);
        letter-spacing: -0.02em;
      }
      .subtitle {
        font-size: clamp(16px, 2vw, 20px);
        color: var(--muted);
        max-width: 520px;
        line-height: 1.6;
        margin: 0;
      }
      .top-actions {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 10px;
      }
      button {
        border: none;
        border-radius: 999px;
        padding: 12px 20px;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
        transition: transform 150ms ease, box-shadow 150ms ease;
        font-family: var(--sans);
      }
      button:active { transform: translateY(1px); }
      button:disabled { opacity: 0.5; cursor: not-allowed; }
      #start {
        background: linear-gradient(120deg, #1a5580, #2474a6);
        color: #fff;
        box-shadow: 0 14px 32px rgba(26, 85, 128, 0.28);
      }
      #stop {
        background: transparent;
        color: var(--ink);
        border: 1px solid var(--border);
      }
      .chip {
        border: 1px solid var(--border);
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 12px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--accent);
        background: rgba(255, 255, 255, 0.7);
        font-family: var(--mono);
      }
      .main-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.6fr) minmax(0, 1fr);
        gap: 24px;
      }
      .panel {
        background: var(--card);
        border-radius: 24px;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        padding: 20px;
        display: grid;
        gap: 16px;
        backdrop-filter: blur(8px);
      }
      .output-panel {
        min-height: 460px;
        align-content: start;
      }
      .panel-title {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        color: var(--muted);
      }
      .output-text {
        font-family: var(--display);
        font-size: clamp(13px, 1.5vw, 20px);
        line-height: 1.5;
        color: var(--ink);
        min-height: 240px;
      }
      .stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 12px;
      }
      .stat {
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 12px;
        background: rgba(255, 255, 255, 0.7);
      }
      .stat label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--muted);
        display: block;
        margin-bottom: 6px;
      }
      .value {
        font-size: 16px;
        font-weight: 600;
        font-family: var(--mono);
        color: var(--ink);
      }
      .value.stale { color: var(--danger); }
      .value.live { color: var(--success); }
      .video-shell {
        position: relative;
        border-radius: 20px;
        overflow: hidden;
        background: #0b0b0b;
        min-height: 320px;
        border: 1px solid rgba(255, 255, 255, 0.2);
      }
      #video {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transform: scaleX(-1);
        display: block;
      }
      .scan {
        position: absolute;
        left: 0;
        right: 0;
        height: 90px;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0) 0%, rgba(26, 85, 128, 0.35) 50%, rgba(255, 255, 255, 0) 100%);
        animation: scan 5.2s linear infinite;
        mix-blend-mode: screen;
        pointer-events: none;
      }
      @keyframes scan {
        0% { top: -90px; opacity: 0; }
        10% { opacity: 0.65; }
        50% { opacity: 0.35; }
        100% { top: 100%; opacity: 0; }
      }
      .status-pill {
        position: absolute;
        top: 16px;
        left: 16px;
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid var(--border);
      }
      .status-pill.SEEING { color: var(--accent); }
      .status-pill.ERROR { color: var(--danger); }
      .status-pill.UNKNOWN { color: var(--muted); }
      .controls {
        display: grid;
        gap: 16px;
      }
      .slider label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--muted);
        display: flex;
        justify-content: space-between;
      }
      input[type="range"] {
        width: 100%;
        accent-color: var(--accent);
      }
      .pipeline {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        padding: 12px;
        border-radius: 14px;
        border: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.65);
        font-family: var(--mono);
        font-size: 12px;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--muted);
      }
      .pipeline span:last-child {
        color: var(--ink);
        text-transform: none;
        letter-spacing: 0.02em;
      }
      @media (prefers-reduced-motion: reduce) {
        .scan { animation: none; }
      }
      @media (max-width: 980px) {
        .main-grid {
          grid-template-columns: 1fr;
        }
        .output-panel {
          min-height: 360px;
        }
      }
      @media (max-width: 720px) {
        .top-actions { width: 100%; justify-content: flex-start; }
        button { width: 100%; }
      }
    </style>
  </head>
  <body>
    <main class="shell">
      <header class="topbar">
        <div class="title-block">
          <div class="kicker">Local VLM Demo</div>
          <h1>What You See</h1>
          <p class="subtitle">A local vision model narrates the live webcam feed in one sentence.</p>
        </div>
        <div class="top-actions">
          <div class="chip" id="backend">Loading...</div>
          <button id="start">Start Lens</button>
          <button id="stop" disabled>Stop</button>
        </div>
      </header>

      <section class="main-grid">
        <div class="panel output-panel">
          <div class="panel-title">Live Narration</div>
          <div class="output-text" id="detail">Waiting for camera...</div>
          <div class="stats">
            <div class="stat">
              <label>Status</label>
              <span id="status" class="value">--</span>
            </div>
            <div class="stat">
              <label>Scene</label>
              <span id="reason" class="value">--</span>
            </div>
            <div class="stat">
              <label>Freshness</label>
              <span id="freshness" class="value">--</span>
            </div>
            <div class="stat">
              <label>Latency</label>
              <span id="latency" class="value">--</span>
            </div>
          </div>
        </div>

        <div class="panel side-panel">
          <div class="panel-title">Live Feed</div>
          <div class="video-shell">
            <video id="video" autoplay playsinline></video>
            <div id="badge" class="status-pill UNKNOWN">Ready</div>
            <div class="scan"></div>
          </div>
          <div class="controls">
            <div class="slider">
              <label for="interval"><span>Sampling Interval</span><span id="interval-val">2.5s</span></label>
              <input type="range" id="interval" min="500" max="10000" step="500" value="2500" />
            </div>
            <div class="pipeline">
              <span class="pipeline-label">Pipeline</span>
              <span id="pipeline">webcam -> lmstudio</span>
            </div>
          </div>
        </div>
      </section>
    </main>

    <canvas id="canvas" style="display:none"></canvas>
    <script>
      const video = document.getElementById("video");
      const canvas = document.getElementById("canvas");
      const badge = document.getElementById("badge");
      const statusEl = document.getElementById("status");
      const reasonEl = document.getElementById("reason");
      const detailEl = document.getElementById("detail");
      const freshnessEl = document.getElementById("freshness");
      const intervalInput = document.getElementById("interval");
      const intervalVal = document.getElementById("interval-val");
      const latencyEl = document.getElementById("latency");
      const backendEl = document.getElementById("backend");
      const pipelineEl = document.getElementById("pipeline");
      const startBtn = document.getElementById("start");
      const stopBtn = document.getElementById("stop");

      let stream;
      let isRunning = false;
      let currentInterval = 2500;
      let timeoutId;

      intervalInput.addEventListener("input", (e) => {
        currentInterval = parseInt(e.target.value, 10);
        intervalVal.textContent = (currentInterval / 1000).toFixed(1) + "s";
      });

      function setStatus(data) {
        const label = data.label || "UNKNOWN";
        const reason = data.reason || "--";
        const isStale = Boolean(data.stale);
        badge.textContent = label;
        badge.className = "status-pill " + label;
        statusEl.textContent = label;
        reasonEl.textContent = reason.replace("_", " ");
        const detail = data.detail || data.result || "Waiting...";
        detailEl.textContent = isStale ? `${detail} (stale)` : detail;
        freshnessEl.textContent = isStale ? "Stale" : "Live";
        freshnessEl.className = "value " + (isStale ? "stale" : "live");
        if (data.elapsed) {
          latencyEl.textContent = Math.round(data.elapsed * 1000) + "ms";
        } else {
          latencyEl.textContent = "--";
        }
      }

      async function loop() {
        if (!isRunning) return;
        if (!video.videoWidth) {
          timeoutId = setTimeout(loop, 100);
          return;
        }
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        try {
          const res = await fetch("/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: canvas.toDataURL("image/jpeg", 1.0) }),
          });
          const data = await res.json();
          setStatus(data);
        } catch (err) {
          detailEl.textContent = "Error calling server.";
        }

        if (isRunning) {
          timeoutId = setTimeout(loop, currentInterval);
        }
      }

      startBtn.onclick = async () => {
        try {
          stream = await navigator.mediaDevices.getUserMedia({ video: true });
          video.srcObject = stream;
          isRunning = true;
          startBtn.disabled = true;
          stopBtn.disabled = false;
          loop();
        } catch (err) {
          alert(err.message || "Unable to access webcam.");
        }
      };

      stopBtn.onclick = () => {
        isRunning = false;
        clearTimeout(timeoutId);
        if (stream) {
          stream.getTracks().forEach((t) => t.stop());
        }
        startBtn.disabled = false;
        stopBtn.disabled = true;
        badge.className = "status-pill UNKNOWN";
        badge.textContent = "Stopped";
      };

      (async () => {
        const res = await fetch("/config");
        const data = await res.json();
        backendEl.textContent = data.model.split("/").pop();
        pipelineEl.textContent = `webcam -> ${data.backend}`;
      })();
    </script>
  </body>
</html>
"""


if __name__ == "__main__":
    import argparse
    import uvicorn
    import webbrowser

    parser = argparse.ArgumentParser(description="WhatYouSee demo")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8083)
    args = parser.parse_args()

    _init_lmstudio()
    logger.info("Starting server on %s:%s", args.host, args.port)
    if args.host in {"0.0.0.0", "127.0.0.1", "localhost"}:
        try:
            webbrowser.open(f"http://localhost:{args.port}")
        except Exception:
            pass
    uvicorn.run(app, host=args.host, port=args.port)

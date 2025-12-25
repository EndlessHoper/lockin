"""
Focus Detection Demo - Browser-based webcam monitoring

A FastAPI server that uses a local vision model to detect focus/distraction.
The browser captures webcam frames and sends them for analysis.

Requirements:
    pip install mlx-vlm pillow fastapi uvicorn

Usage:
    python server.py
    # Opens browser to http://localhost:8000
"""

import base64
import io
import logging
import re
import threading
import time
import webbrowser

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PIL import Image

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("focus")

# Model options - smaller = faster, larger = better instruction following
MODEL = "mlx-community/Qwen2-VL-2B-Instruct-4bit"  # 2B, 4-bit - good balance
# MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"  # 3B, 4-bit - newer
# MODEL = "mlx-community/Phi-3.5-vision-instruct-4bit"  # alternative
# MODEL = "mlx-community/SmolVLM2-500M-Video-Instruct-mlx-8bit-skip-vision"  # smallest

# Settings
MAX_TOKENS = 32
TEMPERATURE = 0.0
MAX_IMAGE_SIZE = 384

# Prompt asking for structured output
PROMPT = """Look at this webcam image. Is the person focused on their computer or distracted?

Reply with exactly: STATUS: FOCUSED or DISTRACTED, REASON: <reason>

Where reason is one of: attentive, phone, not_looking, no_person"""

# Global state
model = None
processor = None
config = None
_lock = threading.Lock()
_last_result = None


class ImageRequest(BaseModel):
    image: str


def init_model():
    """Load the vision model."""
    global model, processor, config
    logger.info(f"Loading model: {MODEL}")
    start = time.time()
    config = load_config(MODEL)
    model, processor = load(MODEL)
    logger.info(f"Model loaded in {time.time() - start:.1f}s")


def resize_image(image: Image.Image, max_side: int) -> Image.Image:
    """Resize image to fit within max_side."""
    w, h = image.size
    scale = max_side / max(w, h)
    if scale >= 1.0:
        return image
    return image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)


def decode_image(data_url: str) -> Image.Image:
    """Decode base64 data URL to PIL Image."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data_url))).convert("RGB")


VALID_STATUSES = {"FOCUSED", "DISTRACTED"}
VALID_REASONS = {"attentive", "phone", "not_looking", "no_person"}


def parse_response(text: str) -> tuple[str, str]:
    """Parse model response to extract STATUS and REASON."""
    text_upper = text.upper()
    text_lower = text.lower()

    # Try to extract STATUS
    status = "FOCUSED"  # default
    if "DISTRACTED" in text_upper:
        status = "DISTRACTED"
    elif "FOCUSED" in text_upper:
        status = "FOCUSED"

    # Try to extract REASON
    reason = "attentive"  # default
    for valid_reason in VALID_REASONS:
        if valid_reason in text_lower:
            reason = valid_reason
            break

    # Fallback heuristics if model didn't use exact keywords
    if "phone" in text_lower:
        reason = "phone"
        status = "DISTRACTED"
    elif "no person" in text_lower or "nobody" in text_lower or "empty" in text_lower:
        reason = "no_person"
        status = "DISTRACTED"
    elif "not looking" in text_lower or "looking away" in text_lower:
        reason = "not_looking"
        status = "DISTRACTED"

    return status, reason


def analyze_image(image: Image.Image) -> dict:
    """Run inference and return focus status."""
    image = resize_image(image, MAX_IMAGE_SIZE)
    prompt = apply_chat_template(processor, config, PROMPT, num_images=1)

    start = time.time()
    response = generate(
        model, processor, prompt, [image],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    elapsed = time.time() - start

    # Extract text from GenerationResult object
    if hasattr(response, 'text'):
        raw = response.text.strip()
    else:
        raw = str(response).strip()

    status, reason = parse_response(raw)

    logger.info(f"[{elapsed:.2f}s] {status} ({reason}) | {raw[:80]}")

    return {
        "status": status,
        "reason": reason,
        "elapsed": round(elapsed, 2),
        "raw": raw,
    }


app = FastAPI(title="Focus Detection Demo")


@app.on_event("startup")
def startup():
    init_model()


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


@app.get("/config")
def get_config():
    return {"model": MODEL.split("/")[-1]}


@app.post("/analyze")
def analyze(req: ImageRequest):
    global _last_result

    if not _lock.acquire(blocking=False):
        if _last_result:
            return {**_last_result, "stale": True}
        return {"status": "BUSY", "reason": "processing", "stale": True}

    try:
        image = decode_image(req.image)
        result = analyze_image(image)
        _last_result = result
        return result
    finally:
        _lock.release()


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Focus Detection Demo</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 2rem;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        h1 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            color: #888;
            margin-bottom: 2rem;
        }
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }
        @media (max-width: 700px) {
            .main-grid { grid-template-columns: 1fr; }
        }
        .panel {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .video-container {
            position: relative;
            border-radius: 12px;
            overflow: hidden;
            background: #000;
            aspect-ratio: 4/3;
        }
        video {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transform: scaleX(-1);
        }
        .status-badge {
            position: absolute;
            top: 1rem;
            left: 1rem;
            padding: 0.5rem 1rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .status-badge.FOCUSED { background: #22c55e; color: #000; }
        .status-badge.DISTRACTED { background: #ef4444; color: #fff; }
        .status-badge.BUSY { background: #f59e0b; color: #000; }
        .status-badge.READY { background: #6b7280; color: #fff; }
        .controls {
            margin-top: 1rem;
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
        }
        button {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: transform 0.1s;
        }
        button:active { transform: scale(0.98); }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn-start { background: #22c55e; color: #000; }
        .btn-stop { background: #ef4444; color: #fff; }
        .panel-title {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #888;
            margin-bottom: 1rem;
        }
        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }
        .stat {
            background: rgba(255,255,255,0.05);
            padding: 1rem;
            border-radius: 8px;
        }
        .stat-label {
            font-size: 0.75rem;
            color: #888;
            margin-bottom: 0.25rem;
        }
        .stat-value {
            font-size: 1.25rem;
            font-weight: 600;
        }
        .raw-output {
            margin-top: 1rem;
            padding: 1rem;
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            font-family: monospace;
            font-size: 0.875rem;
            color: #aaa;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 150px;
            overflow-y: auto;
        }
        .slider-container {
            margin-top: 1rem;
        }
        .slider-container label {
            display: flex;
            justify-content: space-between;
            font-size: 0.875rem;
            color: #888;
            margin-bottom: 0.5rem;
        }
        input[type="range"] {
            width: 100%;
            accent-color: #22c55e;
        }
        .model-chip {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            background: rgba(255,255,255,0.1);
            border-radius: 999px;
            font-size: 0.75rem;
            color: #888;
            margin-bottom: 1rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Focus Detection</h1>
        <p class="subtitle">Local vision model monitoring your focus in real-time</p>
        <div class="model-chip" id="model">Loading...</div>

        <div class="main-grid">
            <div class="panel">
                <div class="video-container">
                    <video id="video" autoplay playsinline></video>
                    <div class="status-badge READY" id="badge">Ready</div>
                </div>
                <div class="controls">
                    <button class="btn-start" id="startBtn">Start</button>
                    <button class="btn-stop" id="stopBtn" disabled>Stop</button>
                </div>
                <div class="slider-container">
                    <label>
                        <span>Polling Interval</span>
                        <span id="intervalValue">1.0s</span>
                    </label>
                    <input type="range" id="interval" min="500" max="5000" step="250" value="1000">
                </div>
            </div>

            <div class="panel">
                <div class="panel-title">Detection Status</div>
                <div class="stat-grid">
                    <div class="stat">
                        <div class="stat-label">Status</div>
                        <div class="stat-value" id="status">--</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Reason</div>
                        <div class="stat-value" id="reason">--</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Latency</div>
                        <div class="stat-value" id="latency">--</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Freshness</div>
                        <div class="stat-value" id="freshness">--</div>
                    </div>
                </div>
                <div class="raw-output" id="raw">Model output will appear here...</div>
            </div>
        </div>
    </div>

    <canvas id="canvas" style="display:none"></canvas>
    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const badge = document.getElementById('badge');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const intervalSlider = document.getElementById('interval');
        const intervalValue = document.getElementById('intervalValue');
        const statusEl = document.getElementById('status');
        const reasonEl = document.getElementById('reason');
        const latencyEl = document.getElementById('latency');
        const freshnessEl = document.getElementById('freshness');
        const rawEl = document.getElementById('raw');
        const modelEl = document.getElementById('model');

        let stream = null;
        let running = false;
        let timeoutId = null;

        // Load config
        fetch('/config').then(r => r.json()).then(data => {
            modelEl.textContent = data.model;
        });

        intervalSlider.oninput = () => {
            intervalValue.textContent = (intervalSlider.value / 1000).toFixed(1) + 's';
        };

        function updateUI(data) {
            const status = data.status || 'UNKNOWN';
            badge.textContent = status;
            badge.className = 'status-badge ' + status;
            statusEl.textContent = status;
            reasonEl.textContent = (data.reason || '--').replace('_', ' ');
            latencyEl.textContent = data.elapsed ? (data.elapsed * 1000).toFixed(0) + 'ms' : '--';
            freshnessEl.textContent = data.stale ? 'Stale' : 'Live';
            freshnessEl.style.color = data.stale ? '#ef4444' : '#22c55e';
            rawEl.textContent = data.raw || 'No output';
        }

        async function captureAndAnalyze() {
            if (!running || !video.videoWidth) {
                if (running) timeoutId = setTimeout(captureAndAnalyze, 100);
                return;
            }

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            // Don't flip - send original orientation to model so text is readable
            ctx.drawImage(video, 0, 0);

            try {
                const res = await fetch('/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({image: canvas.toDataURL('image/jpeg', 0.8)})
                });
                const data = await res.json();
                updateUI(data);
            } catch (e) {
                rawEl.textContent = 'Error: ' + e.message;
            }

            if (running) {
                timeoutId = setTimeout(captureAndAnalyze, parseInt(intervalSlider.value));
            }
        }

        startBtn.onclick = async () => {
            try {
                stream = await navigator.mediaDevices.getUserMedia({video: true});
                video.srcObject = stream;
                running = true;
                startBtn.disabled = true;
                stopBtn.disabled = false;
                captureAndAnalyze();
            } catch (e) {
                alert('Camera error: ' + e.message);
            }
        };

        stopBtn.onclick = () => {
            running = false;
            clearTimeout(timeoutId);
            if (stream) stream.getTracks().forEach(t => t.stop());
            startBtn.disabled = false;
            stopBtn.disabled = true;
            badge.textContent = 'Stopped';
            badge.className = 'status-badge READY';
        };
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    print("Starting Focus Detection Demo...")
    print("Opening browser to http://localhost:8000")
    webbrowser.open("http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

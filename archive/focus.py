import json
import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from together import Together

LOG_LEVEL = "INFO"
logger = logging.getLogger("focus")
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
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
TOGETHER_MODEL = "google/gemma-3n-E4B-it"
MAX_TOKENS = 48
TOGETHER_SYSTEM_PROMPT = "Only respond in JSON."
TOGETHER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "focus_signals",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "person_present": {"type": "boolean"},
                "looking_at_camera": {"type": "boolean"},
                "phone_visible": {"type": "boolean"},
            },
            "required": ["person_present", "looking_at_camera", "phone_visible"],
            "additionalProperties": False,
        },
    },
}

PROMPT = (
    "Classify attention from this webcam image.\n"
    "Rules (be conservative):\n"
    "- person_present=true ONLY if a person is clearly visible.\n"
    "- looking_at_camera=true ONLY if eyes are clearly visible and directed at the camera lens.\n"
    "- phone_visible=true ONLY if a phone/smartphone is clearly visible.\n"
    "- Do NOT count monitors, laptops, remotes, reflections, or hands as phones.\n"
    "- If person_present=false, set looking_at_camera=false and phone_visible=false.\n"
    "- If unsure about any field, set it to false.\n"
    "DISTRACTED if phone_visible=true OR person_present=false OR looking_at_camera=false.\n"
    "Otherwise FOCUSED.\n"
    "Return JSON only in this schema:\n"
    "{\"person_present\":true|false,\"looking_at_camera\":true|false,\"phone_visible\":true|false}\n"
    "No extra text."
)

client = None
_lock = threading.Lock()
_last_result = None


class ImageRequest(BaseModel):
    image: str


def _init_client():
    global client
    if client is not None:
        return
    if not TOGETHER_API_KEY:
        raise RuntimeError("Set TOGETHER_API_KEY in .env or the environment.")
    client = Together(api_key=TOGETHER_API_KEY)
    logger.info("Using Together model %s", TOGETHER_MODEL)


def _normalize_image_url(image_url: str) -> str:
    if image_url.startswith("data:"):
        return image_url
    return f"data:image/jpeg;base64,{image_url}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_client()
    yield


app = FastAPI(title="CodexVision Focus", lifespan=lifespan)


def _parse_json_payload(raw: str) -> dict | None:
    if not raw:
        return None
    candidate = raw.strip()
    if not candidate.startswith("{"):
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _parse_response(text: str) -> dict:
    raw = (text or "").strip()
    data = _parse_json_payload(raw)
    if data:
        person_present = _to_bool(data.get("person_present"))
        looking_at_camera = _to_bool(data.get("looking_at_camera"))
        phone_visible = _to_bool(data.get("phone_visible"))

        if person_present is None or looking_at_camera is None or phone_visible is None:
            line = ""
        else:
            if person_present is False:
                looking_at_camera = False
                phone_visible = False
            distracted = (not person_present) or (not looking_at_camera) or phone_visible
            status = "DISTRACTED" if distracted else "FOCUSED"
            if phone_visible:
                reason = "phone"
            elif not person_present:
                reason = "no_person"
            elif not looking_at_camera:
                reason = "not_looking"
            else:
                reason = "focused"
            line = f"{status}: {reason}"
            detail_map = {
                "phone": "phone visible",
                "not_looking": "not looking at camera",
                "no_person": "no person in frame",
                "focused": "focused",
            }
            return {
                "label": status,
                "distracted": distracted,
                "reason": reason,
                "detail": detail_map[reason],
                "signals": {
                    "person_present": person_present,
                    "looking_at_camera": looking_at_camera,
                    "phone_visible": phone_visible,
                },
            }
    else:
        line = raw.splitlines()[0].strip() if raw else ""
    status_part, reason = (line.split(":", 1) + [""])[:2]
    status = status_part.strip().upper()
    reason = reason.strip().lower().replace(" ", "_")

    if status.startswith("FOCUS"):
        status = "FOCUSED"
    elif status.startswith("DISTRACT"):
        status = "DISTRACTED"
    else:
        status = "DISTRACTED"

    if reason in {"away", "looking", "looking_away"}:
        reason = "not_looking"
    if reason not in {"phone", "not_looking", "no_person", "focused"}:
        reason = "focused" if status == "FOCUSED" else "unknown"
    if status == "FOCUSED":
        reason = "focused"

    detail_map = {
        "phone": "phone visible",
        "not_looking": "not looking at camera",
        "no_person": "no person in frame",
        "focused": "focused",
        "unknown": "unclear, treating as distracted",
    }

    return {
        "label": status,
        "distracted": status == "DISTRACTED",
        "reason": reason,
        "detail": detail_map.get(reason, reason),
    }


def _analyze_together(image_url: str) -> dict:
    _init_client()
    start = time.time()
    image_url = _normalize_image_url(image_url)
    messages = [
        {"role": "system", "content": TOGETHER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]
    try:
        response = client.chat.completions.create(
            model=TOGETHER_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=MAX_TOKENS,
            stream=True,
            response_format=TOGETHER_RESPONSE_FORMAT,
        )
        chunks = []
        for token in response:
            if not hasattr(token, "choices"):
                continue
            choice = (token.choices or [None])[0]
            if not choice or not hasattr(choice, "delta"):
                continue
            delta = choice.delta
            content = getattr(delta, "content", None)
            if content:
                chunks.append(content)
        raw = "".join(chunks).strip()
        parsed = _parse_response(raw)
        parsed["elapsed"] = round(time.time() - start, 2)
        parsed["raw"] = raw
        return parsed
    except Exception as exc:
        logger.error("Together request failed: %s", exc)
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
    return {"backend": "together", "model": TOGETHER_MODEL}


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
        result = _analyze_together(req.image)
        _last_result = {**result, "stale": False}
        logger.info(
            "Result %s reason=%s elapsed=%.2fs",
            result["label"],
            result["reason"],
            result["elapsed"],
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
    <title>Focus Detector</title>
    <style>
      :root {
        --bg: #f6f1e6;
        --bg-accent: #e8eef6;
        --card: rgba(255, 255, 255, 0.8);
        --border: rgba(15, 23, 42, 0.1);
        --text: #0f172a;
        --muted: #6b7280;
        --accent: #0f766e;
        --accent-dark: #0f3d35;
        --danger: #b42318;
        --success: #067647;
        --shadow: 0 20px 60px rgba(15, 23, 42, 0.12);
        --font: "Avenir Next", "Avenir", "Futura", "Trebuchet MS", sans-serif;
        --mono: ui-monospace, "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: var(--font);
        color: var(--text);
        background: radial-gradient(circle at top left, var(--bg-accent), transparent 60%),
                    radial-gradient(circle at 20% 10%, #fef3c7, transparent 55%),
                    var(--bg);
        min-height: 100vh;
      }
      .frame {
        max-width: 1100px;
        margin: 0 auto;
        padding: 32px 20px 48px;
        display: grid;
        gap: 24px;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        animation: rise 480ms ease-out;
      }
      @keyframes rise {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
      }
      header {
        grid-column: 1 / -1;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
      }
      h1 {
        margin: 0;
        font-size: clamp(24px, 4vw, 36px);
        letter-spacing: -0.03em;
      }
      .tag {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: var(--accent-dark);
        border: 1px solid var(--border);
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.6);
      }
      .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 20px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(12px);
        display: flex;
        flex-direction: column;
        gap: 16px;
      }
      .video-shell {
        position: relative;
        border-radius: 16px;
        overflow: hidden;
        background: #0b0b0b;
        min-height: 220px;
      }
      #video {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transform: scaleX(-1);
        display: block;
      }
      .status-pill {
        position: absolute;
        top: 16px;
        left: 16px;
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid var(--border);
      }
      .status-pill.FOCUSED { color: var(--success); }
      .status-pill.DISTRACTED { color: var(--danger); }
      .status-pill.UNKNOWN { color: var(--muted); }
      .metrics {
        display: grid;
        gap: 12px;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      }
      .metric {
        padding: 12px 14px;
        border-radius: 14px;
        border: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.7);
      }
      .metric .label {
        font-size: 12px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.12em;
      }
      .metric .value {
        font-size: 20px;
        font-weight: 700;
        margin-top: 6px;
      }
      .metric .value.stale { color: var(--danger); }
      .metric .value.live { color: var(--success); }
      .controls {
        display: grid;
        gap: 12px;
      }
      .controls button {
        border: none;
        border-radius: 12px;
        padding: 12px 16px;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
        transition: transform 160ms ease, box-shadow 160ms ease;
      }
      .controls button:active { transform: translateY(1px); }
      #start {
        background: var(--accent);
        color: white;
        box-shadow: 0 12px 24px rgba(15, 118, 110, 0.25);
      }
      #stop {
        background: rgba(15, 23, 42, 0.08);
        color: var(--text);
      }
      .slider {
        display: grid;
        gap: 6px;
      }
      .slider label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--muted);
        display: flex;
        justify-content: space-between;
      }
      input[type="range"] {
        width: 100%;
        accent-color: var(--accent);
      }
      .footer {
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        color: var(--muted);
        font-family: var(--mono);
      }
      .reason {
        font-size: 15px;
        line-height: 1.4;
        color: var(--accent-dark);
        font-weight: 600;
      }
      @media (max-width: 720px) {
        header { flex-direction: column; align-items: flex-start; }
      }
    </style>
  </head>
  <body>
    <main class="frame">
      <header>
        <div>
          <div class="tag">CodexVision Demo</div>
          <h1>Focus Detector</h1>
        </div>
        <div class="tag" id="backend">Loading...</div>
      </header>

      <section class="card">
        <div class="video-shell">
          <video id="video" autoplay playsinline></video>
          <div id="badge" class="status-pill UNKNOWN">Ready</div>
        </div>
        <div class="metrics">
          <div class="metric">
            <div class="label">Status</div>
            <div id="status" class="value">--</div>
          </div>
          <div class="metric">
            <div class="label">Reason</div>
            <div id="reason" class="value">--</div>
          </div>
          <div class="metric">
            <div class="label">Freshness</div>
            <div id="freshness" class="value">--</div>
          </div>
        </div>
        <div class="reason" id="detail">Waiting for camera...</div>
      </section>

      <section class="card">
        <div class="controls">
          <button id="start">Start</button>
          <button id="stop" disabled>Stop</button>
          <div class="slider">
            <label for="interval"><span>Polling Interval</span><span id="interval-val">2.5s</span></label>
            <input type="range" id="interval" min="500" max="10000" step="500" value="2500" />
          </div>
        </div>
        <div class="footer">
          <span id="latency">0ms</span>
          <span id="pipeline">webcam -> together</span>
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
            body: JSON.stringify({ image: canvas.toDataURL("image/jpeg", 0.6) }),
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

    parser = argparse.ArgumentParser(description="CodexVision focus demo")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    _init_client()
    logger.info("Starting server on %s:%s", args.host, args.port)
    if args.host in {"0.0.0.0", "127.0.0.1", "localhost"}:
        try:
            webbrowser.open(f"http://localhost:{args.port}")
        except Exception:
            pass
    uvicorn.run(app, host=args.host, port=args.port)

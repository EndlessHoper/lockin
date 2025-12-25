import atexit
import base64
import io
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PIL import Image

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_IMAGE_PROCESSOR_USE_FAST", "0")
os.environ.setdefault("TRANSFORMERS_USE_FAST", "0")
if not os.getenv("TMPDIR"):
    tmp_dir = ROOT / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

def _find_model_dir(patterns):
    for pattern in patterns:
        for match in sorted(ROOT.glob(pattern)):
            if match.is_dir():
                return str(match)
    return None

def _find_coreml_package(path: Path):
    for entry in path.glob("*.mlpackage"):
        return entry
    return None

def _ensure_coreml_vision_tower(model_path: Path, override: str | None):
    if not model_path.is_dir():
        return
    if _find_coreml_package(model_path):
        return
    candidates = []
    if override:
        candidates.append(Path(override))
    candidates.extend([
        ROOT / "llava-fastvithd_0.5b_stage3",
        ROOT / "llava-fastvithd_0.5b_stage3_llm.fp16",
    ])
    src = None
    for cand in candidates:
        if cand.is_dir() and cand.suffix == ".mlpackage":
            src = cand
        elif cand.is_dir():
            src = _find_coreml_package(cand)
        elif cand.suffix == ".mlpackage" and cand.exists():
            src = cand
        if src:
            break
    if not src:
        return
    dest = model_path / src.name
    if dest.exists():
        return
    try:
        os.symlink(src, dest)
    except OSError:
        shutil.copytree(src, dest)
    print(f"Using vision tower from {src}")

# GLOBAL STATE
model = None
processor = None
config = None
prompt = None
groq_client = None
_initialized = False
_lock = threading.Lock()
_last_result = None
_llamacpp_proc = None

# SETTINGS (hardcoded)
MODEL_PATH = "mlx-community/SmolVLM2-500M-Video-Instruct-mlx"
VISION_TOWER_PATH = None
MAX_SIDE = 384
MAX_TOKENS = 128
MAX_TOKENS_CLASSIFY = 32
BACKEND = "llamacpp"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_API_KEY = "fill in"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = ""
LLAMACPP_HOST = "127.0.0.1"
LLAMACPP_PORT = 8080
LLAMACPP_BASE_URL = f"http://{LLAMACPP_HOST}:{LLAMACPP_PORT}"
LLAMACPP_URL = f"{LLAMACPP_BASE_URL}/v1/chat/completions"
LLAMACPP_MODEL = ""
LLAMACPP_BIN = "/opt/homebrew/bin/llama-server"
LLAMACPP_AUTOSTART = True
LLAMACPP_START_TIMEOUT = 120
LLAMACPP_MODEL_PATH = str(ROOT / "models" / "Qwen3VL-2B-Instruct-Q4_K_M.gguf")
LLAMACPP_MMPROJ_PATH = str(ROOT / "models" / "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf")
LLAMACPP_ARGS = ["--image-min-tokens", "1024"]

# PROMPT - single shot classification, short output
PROMPT = (
    "Classify attention from the image.\n"
    "Rules: DISTRACTED if phone visible, looking away, eyes closed/asleep, or person not visible. "
    "If image is dark/blank/covered or unsure, choose DISTRACTED.\n"
    "Output exactly one line in this format:\n"
    "FOCUSED: focused\n"
    "or\n"
    "DISTRACTED: phone\n"
    "DISTRACTED: away\n"
    "DISTRACTED: eyes_closed\n"
    "DISTRACTED: no_person\n"
    "Choose exactly ONE reason token from: phone, away, eyes_closed, no_person, focused.\n"
    "Do not add extra words or separators.\n"
    "No other text."
)
OLLAMA_PROMPT = (
    "Classify attention from the image.\n"
    "Rules: DISTRACTED if phone visible, looking away, eyes closed/asleep, or person not visible. "
    "If image is dark/blank/covered or unsure, choose DISTRACTED.\n"
    "Return JSON only. Keys: state, reason.\n"
    "state must be FOCUSED or DISTRACTED.\n"
    "reason must be one of: phone, away, eyes_closed, no_person, focused."
)
OLLAMA_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "state": {"type": "string", "enum": ["FOCUSED", "DISTRACTED"]},
        "reason": {
            "type": "string",
            "enum": ["phone", "away", "eyes_closed", "no_person", "focused"],
        },
    },
    "required": ["state", "reason"],
}

def _resolve_llamacpp_bin():
    if Path(LLAMACPP_BIN).exists():
        return LLAMACPP_BIN
    return shutil.which(LLAMACPP_BIN)

def _llamacpp_is_ready():
    try:
        with urllib.request.urlopen(f"{LLAMACPP_BASE_URL}/v1/models", timeout=1):
            return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False

def _stop_llamacpp_server():
    global _llamacpp_proc
    if not _llamacpp_proc or _llamacpp_proc.poll() is not None:
        return
    _llamacpp_proc.terminate()
    try:
        _llamacpp_proc.wait(timeout=5)
    except Exception:
        _llamacpp_proc.kill()

def _ensure_llamacpp_server():
    global _llamacpp_proc
    if not LLAMACPP_AUTOSTART or _llamacpp_is_ready():
        return
    bin_path = _resolve_llamacpp_bin()
    if not bin_path:
        raise RuntimeError("llama-server not found. Set LLAMACPP_BIN in server.py.")
    model_path = Path(LLAMACPP_MODEL_PATH)
    if not model_path.exists():
        raise RuntimeError(f"Missing GGUF model at {model_path}. Update LLAMACPP_MODEL_PATH in server.py.")
    mmproj_path = Path(LLAMACPP_MMPROJ_PATH)
    if not mmproj_path.exists():
        raise RuntimeError(f"Missing mmproj at {mmproj_path}. Update LLAMACPP_MMPROJ_PATH in server.py.")
    cmd = [
        bin_path,
        "--model", str(model_path),
        "--mmproj", str(mmproj_path),
        "--host", LLAMACPP_HOST,
        "--port", str(LLAMACPP_PORT),
    ]
    cmd.extend(LLAMACPP_ARGS)
    print("Starting llama.cpp server...")
    _llamacpp_proc = subprocess.Popen(cmd)
    atexit.register(_stop_llamacpp_server)
    deadline = time.time() + LLAMACPP_START_TIMEOUT
    while time.time() < deadline:
        if _llamacpp_is_ready():
            return
        if _llamacpp_proc.poll() is not None:
            raise RuntimeError("llama-server exited before becoming ready.")
        time.sleep(0.25)
    raise RuntimeError("llama-server did not become ready in time.")

def _init_backend():
    global model, processor, config, prompt, groq_client, _initialized
    if _initialized: return

    if BACKEND == "groq":
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        print(f"Using Groq: {GROQ_MODEL}")
        _initialized = True
        return
    if BACKEND == "ollama":
        if not OLLAMA_MODEL:
            raise RuntimeError("Set OLLAMA_MODEL in server.py to your Ollama vision model.")
        print(f"Using Ollama: {OLLAMA_MODEL} @ {OLLAMA_URL}")
        _initialized = True
        return
    if BACKEND == "llamacpp":
        _ensure_llamacpp_server()
        print(f"Using llama.cpp server: {LLAMACPP_URL}")
        _initialized = True
        return

    def _load_model(extra_kwargs=None):
        kwargs = {}
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        return load(MODEL_PATH, **kwargs)

    print(f"Loading local model: {MODEL_PATH}")
    config = load_config(MODEL_PATH)
    if config.get("model_type") in {"llava_qwen2", "fastvlm"}:
        _ensure_coreml_vision_tower(Path(MODEL_PATH), VISION_TOWER_PATH)
    model_type = config.get("model_type")
    is_jina = model_type == "jvlm"
    try:
        load_kwargs = {"trust_remote_code": not is_jina}
        model, processor = _load_model(load_kwargs)
    except ValueError as e:
        if "parameters not in model" in str(e):
            print("Retrying load with only_llm=True (FastVLM checkpoint mapping).")
            retry_kwargs = {"only_llm": True, "trust_remote_code": not is_jina}
            model, processor = _load_model(retry_kwargs)
        else:
            raise

    if config.get("model_type") in {"llava_qwen2", "fastvlm"} and getattr(model, "vision_tower", None) is None:
        raise RuntimeError(
            "FastVLM requires a CoreML vision encoder (.mlpackage). "
            "Place one in the model directory or set CODEXVISION_VISION_TOWER. "
            "Run: python ml-fastvlm/model_export/export_vision_encoder.py "
            f"--model-path {MODEL_PATH}"
        )

    prompt = apply_chat_template(processor, config, PROMPT, num_images=1)
    print("Model loaded.")
    _initialized = True

@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_backend()
    yield

app = FastAPI(title="CodexVision", lifespan=lifespan)

class ImageRequest(BaseModel):
    image: str

def _resize_pil(image, max_side):
    if max_side <= 0: return image
    w, h = image.size
    scale = max_side / float(max(w, h))
    if scale >= 1.0: return image
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)

def _decode_image(data_url):
    if "," in data_url: data_url = data_url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data_url))).convert("RGB")

def _post_json(url, payload, timeout=60):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)

def _parse_response(text, elapsed):
    raw = (text or "").strip()
    line = raw.splitlines()[0].strip() if raw else ""
    status_part, reason = (line.split(":", 1) + [""])[:2] if line else ("", "")
    status_part = status_part.strip().upper()
    reason = reason.strip()

    if status_part.startswith("FOCUS"):
        status = "FOCUSED"
    elif status_part.startswith("DISTRACT"):
        status = "DISTRACTED"
    else:
        first = line.split()[0].upper() if line else ""
        if first.startswith("FOCUS"):
            status = "FOCUSED"
        elif first.startswith("DISTRACT"):
            status = "DISTRACTED"
        else:
            status = "DISTRACTED"

    if not reason:
        reason = "unknown"

    reason = reason.strip()
    if len(reason) > 80:
        reason = reason[:77] + "..."

    return {
        "label": status,
        "distracted": status == "DISTRACTED",
        "reason": "AI",
        "what_you_see": reason,
        "result": f"{status}: {reason}",
        "elapsed": round(elapsed, 2),
    }

def _analyze_groq(image_url):
    start = time.time()
    if not image_url.startswith("data:"): image_url = f"data:image/jpeg;base64,{image_url}"
    msgs = [{"role": "user", "content": [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": image_url}}]}]
    
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=msgs,
            temperature=0.0,
            max_tokens=MAX_TOKENS_CLASSIFY,
        )
        raw = completion.choices[0].message.content or ""
        print(f"\n{'='*50}")
        print(f"[GROQ] Raw: {raw!r}")
    except Exception as e:
        print(f"[GROQ] ERROR: {e}")
        return {"label": "ERROR", "distracted": False, "reason": "error", "what_you_see": str(e), "elapsed": 0}

    result = _parse_response(raw, time.time() - start)
    print(f"[GROQ] Result -> {result['label']} | {result['elapsed']}s")
    print(f"{'='*50}")
    return result

def _analyze_ollama(image_url):
    start = time.time()
    if not image_url.startswith("data:"):
        image_url = f"data:image/jpeg;base64,{image_url}"
    image_b64 = image_url.split(",", 1)[1]

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "user", "content": OLLAMA_PROMPT, "images": [image_b64]}
        ],
        "stream": False,
        "format": OLLAMA_SCHEMA,
        "options": {"temperature": 0},
    }
    try:
        resp = _post_json(f"{OLLAMA_URL}/api/chat", payload, timeout=120)
        raw = resp.get("message", {}).get("content", "")
        print(f"\n{'='*50}")
        print(f"[OLLAMA] Raw: {raw!r}")
        data = json.loads(raw)
        state = data.get("state", "DISTRACTED")
        reason = data.get("reason", "unknown")
        result = {
            "label": state,
            "distracted": state == "DISTRACTED",
            "reason": "AI",
            "what_you_see": reason,
            "result": f"{state}: {reason}",
            "elapsed": round(time.time() - start, 2),
        }
        print(f"[OLLAMA] Result -> {result['label']} | {result['elapsed']}s")
        print(f"{'='*50}")
        return result
    except Exception as e:
        print(f"[OLLAMA] ERROR: {e}")
        return {"label": "ERROR", "distracted": False, "reason": "error", "what_you_see": str(e), "elapsed": 0}

def _analyze_llamacpp(image_url):
    start = time.time()
    if not image_url.startswith("data:"):
        image_url = f"data:image/jpeg;base64,{image_url}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]
    payload = {
        "messages": messages,
        "temperature": 0,
        "max_tokens": MAX_TOKENS_CLASSIFY,
    }
    if LLAMACPP_MODEL:
        payload["model"] = LLAMACPP_MODEL
    try:
        resp = _post_json(LLAMACPP_URL, payload, timeout=120)
        choice = (resp.get("choices") or [{}])[0]
        raw = (choice.get("message") or {}).get("content", "")
        print(f"\n{'='*50}")
        print(f"[LLAMACPP] Raw: {raw!r}")
        result = _parse_response(raw, time.time() - start)
        print(f"[LLAMACPP] Result -> {result['label']} | {result['elapsed']}s")
        print(f"{'='*50}")
        return result
    except Exception as e:
        print(f"[LLAMACPP] ERROR: {e}")
        return {"label": "ERROR", "distracted": False, "reason": "error", "what_you_see": str(e), "elapsed": 0}

def _analyze_image(image):
    image = _resize_pil(image, MAX_SIDE)
    start = time.time()
    
    res = generate(model, processor, prompt, [image], max_tokens=MAX_TOKENS_CLASSIFY, temperature=0.0)
    raw = getattr(res, "text", str(res)).strip()
    print(f"\n{'='*50}")
    print(f"[LOCAL] Raw: {raw!r}")

    result = _parse_response(raw, time.time() - start)
    print(f"[LOCAL] Result -> {result['label']} | {result['elapsed']}s")
    print(f"{'='*50}")
    return result

@app.get("/")
def index(): return HTMLResponse(INDEX_HTML)

@app.get("/config")
def cfg():
    model = MODEL_PATH
    if BACKEND == "ollama":
        model = OLLAMA_MODEL or "unset"
    elif BACKEND == "llamacpp":
        model = LLAMACPP_MODEL or Path(LLAMACPP_MODEL_PATH).name
    return {"backend": BACKEND, "model": model}

@app.post("/analyze")
def analyze(req: ImageRequest):
    global _last_result
    if not _lock.acquire(blocking=False):
        print(f"[ANALYZE] Request skipped (busy), returning stale result")
        return {**_last_result, "stale": True} if _last_result else {"label": "BUSY", "stale": True}
    try:
        print(f"\n[ANALYZE] New request received ({BACKEND} backend)")
        if BACKEND == "groq":
            res = _analyze_groq(req.image)
        elif BACKEND == "ollama":
            res = _analyze_ollama(req.image)
        elif BACKEND == "llamacpp":
            res = _analyze_llamacpp(req.image)
        else:
            res = _analyze_image(_decode_image(req.image))
        _last_result = {**res, "stale": False}
        return _last_result
    finally: _lock.release()

INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>CodexVision</title>
    <style>
      :root {
        --bg: #09090b;
        --panel: #18181b;
        --border: #27272a;
        --text: #e4e4e7;
        --text-muted: #a1a1aa;
        --primary: #2563eb;
        --primary-fg: #ffffff;
        --danger: #ef4444;
        --success: #22c55e;
        --font-sans: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif;
      }
      body {
        margin: 0;
        font-family: var(--font-sans);
        background: var(--bg);
        color: var(--text);
        -webkit-font-smoothing: antialiased;
      }
      .wrap {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 350px;
        gap: 24px;
        padding: 24px;
        min-height: 100vh;
        box-sizing: border-box;
        max-width: 1600px;
        margin: 0 auto;
      }
      .panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px;
        display: flex;
        flex-direction: column;
        gap: 16px;
      }
      h1 { margin: 0; font-size: 20px; font-weight: 600; letter-spacing: -0.02em; }
      
      .video-wrap {
        position: relative;
        border-radius: 8px;
        overflow: hidden;
        background: #000;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        aspect-ratio: 4/3;
      }
      #video {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transform: scaleX(-1);
      }
      
      .badge {
        position: absolute;
        top: 16px;
        left: 16px;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        backdrop-filter: blur(8px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      }
      .badge.UNKNOWN { background: rgba(39, 39, 42, 0.8); color: var(--text-muted); }
      .badge.FOCUSED { background: rgba(34, 197, 94, 0.2); color: #86efac; border: 1px solid rgba(34, 197, 94, 0.3); }
      .badge.DISTRACTED { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.3); }

      .stat-card {
        background: rgba(0,0,0,0.2);
        border-radius: 8px;
        padding: 16px;
        border: 1px solid var(--border);
      }
      .stat-value { font-size: 24px; font-weight: 700; margin-bottom: 4px; }
      .stat-label { font-size: 13px; color: var(--text-muted); }
      
      .output-text { font-family: ui-monospace, monospace; font-size: 13px; line-height: 1.5; color: var(--text-muted); }

      .btn-group { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
      button {
        appearance: none; border: none; background: var(--primary); color: var(--primary-fg);
        padding: 12px; border-radius: 6px; font-size: 14px; font-weight: 500; cursor: pointer;
      }
      button:disabled { opacity: 0.5; }
      button.secondary { background: #27272a; color: var(--text); }

      .slider-group { display: flex; flex-direction: column; gap: 8px; }
      .slider-header { display: flex; justify-content: space-between; font-size: 13px; color: var(--text-muted); }
      
      .pipeline { display: flex; gap: 8px; margin-top: 8px; }
      .step { flex: 1; height: 4px; background: #27272a; border-radius: 2px; }
      .step.active { background: var(--primary); box-shadow: 0 0 8px var(--primary); }
      
      .meta-row {
        display: flex; justify-content: space-between; font-size: 12px; color: var(--text-muted);
        border-top: 1px solid var(--border); padding-top: 12px; margin-top: auto;
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="panel">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <h1>CodexVision</h1>
            <div id="badge" class="badge UNKNOWN">Ready</div>
        </div>
        <div class="video-wrap">
          <video id="video" autoplay playsinline></video>
        </div>
        <div class="pipeline">
            <div id="step-cap" class="step"></div>
            <div id="step-inf" class="step"></div>
            <div id="step-res" class="step"></div>
        </div>
      </div>

      <div class="panel">
        <div class="stat-card">
          <div id="status" class="stat-value">--</div>
          <div class="stat-label">Current Status</div>
        </div>
        <div class="stat-card">
           <div class="output-text" id="desc">Waiting...</div>
           <div class="stat-label" style="margin-top:8px">Observation</div>
        </div>
        <div class="btn-group">
            <button id="start">Start</button>
            <button id="stop" class="secondary" disabled>Stop</button>
        </div>
        <div class="slider-group">
            <div class="slider-header"><span>Interval</span><span id="interval-val">2.5s</span></div>
            <input type="range" id="interval" min="500" max="10000" step="500" value="2500">
        </div>
        <div class="meta-row">
            <span id="backend">Loading...</span>
            <span id="latency">0ms</span>
        </div>
      </div>
    </div>
    <canvas id="canvas" style="display: none"></canvas>
    <script>
      const video=document.getElementById("video"), canvas=document.getElementById("canvas"), statusEl=document.getElementById("status"), 
            descEl=document.getElementById("desc"), badge=document.getElementById("badge"), latencyEl=document.getElementById("latency"),
            backendEl=document.getElementById("backend"), startBtn=document.getElementById("start"), stopBtn=document.getElementById("stop"),
            intervalInput=document.getElementById("interval"), intervalVal=document.getElementById("interval-val"),
            steps={cap:document.getElementById("step-cap"), inf:document.getElementById("step-inf"), res:document.getElementById("step-res")};

      let stream, isRunning=false, currentInterval=2500, timeoutId;

      intervalInput.addEventListener("input", (e) => {
          currentInterval = parseInt(e.target.value);
          intervalVal.textContent = (currentInterval / 1000).toFixed(1) + "s";
      });

      function setStatus(label, desc, elapsed, resultLine) {
        const isDistracted = label === "DISTRACTED";
        badge.textContent = label; badge.className = `badge ${label}`;
        statusEl.textContent = label; statusEl.style.color = isDistracted ? "var(--danger)" : "var(--success)";
        const line = resultLine || (desc ? `${label}: ${desc}` : label);
        descEl.textContent = line; latencyEl.textContent = elapsed ? `${(elapsed * 1000).toFixed(0)}ms` : "-";
      }

      function setStep(step) {
          steps.cap.classList.toggle("active", step === "capture");
          steps.inf.classList.toggle("active", step === "inference");
          steps.res.classList.toggle("active", step === "result");
      }

      async function loop() {
          if (!isRunning) return;
          if (!video.videoWidth) { timeoutId = setTimeout(loop, 100); return; }
          setStep("capture");
          canvas.width = 384; canvas.height = 384 * (video.videoHeight / video.videoWidth);
          const ctx = canvas.getContext("2d"); ctx.translate(canvas.width, 0); ctx.scale(-1, 1); ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          try {
            setStep("inference");
            const res = await fetch("/analyze", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({ image: canvas.toDataURL("image/jpeg", 0.6) })});
            const data = await res.json();
            setStep("result");
            setStatus(data.label, data.what_you_see, data.elapsed, data.result);
          } catch (e) {}
          setStep("idle");
          if (isRunning) timeoutId = setTimeout(loop, currentInterval);
      }

      startBtn.onclick = async () => {
        try {
          stream = await navigator.mediaDevices.getUserMedia({video: true});
          video.srcObject = stream; isRunning = true; startBtn.disabled = true; stopBtn.disabled = false; loop();
        } catch (e) { alert(e.message); }
      };

      stopBtn.onclick = () => {
        isRunning = false; clearTimeout(timeoutId);
        if (stream) stream.getTracks().forEach(t => t.stop());
        startBtn.disabled = false; stopBtn.disabled = true; badge.className = "badge UNKNOWN"; badge.textContent = "Stopped";
      };

      (async () => {
          const res = await fetch("/config"); const data = await res.json();
          backendEl.textContent = `${data.backend} / ${data.model.split('/').pop()}`;
      })();
    </script>
  </body>
</html>
"""

if __name__ == "__main__":
    import argparse, uvicorn, webbrowser
    p = argparse.ArgumentParser()
    p.add_argument("-groq", action="store_true")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8081)
    args = p.parse_args()
    if args.groq: BACKEND = "groq"
    _init_backend()
    if args.host in {"0.0.0.0", "127.0.0.1", "localhost"}:
        try: webbrowser.open(f"http://localhost:{args.port}")
        except: pass
    uvicorn.run(app, host=args.host, port=args.port)

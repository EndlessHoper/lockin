# Lock In - Architecture & Phase 1 Testing

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Mac Native App                           │
│                      (Tauri / Electron)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   Webcam     │    │    Screen    │    │  Session State   │  │
│  │   Capture    │    │   Capture    │    │  (task, persona, │  │
│  │              │    │              │    │   history, etc)  │  │
│  └──────┬───────┘    └──────┬───────┘    └────────┬─────────┘  │
│         │                   │                     │             │
│         ▼                   ▼                     │             │
│  ┌──────────────────────────────────┐             │             │
│  │      Local Vision Model          │             │             │
│  │  (SmolVLM via llama.cpp / MLX)   │             │             │
│  │                                  │             │             │
│  │  Input: webcam frame + screen    │             │             │
│  │  Output: {distracted, reason,    │             │             │
│  │           confidence}            │             │             │
│  └──────────────┬───────────────────┘             │             │
│                 │                                 │             │
│                 ▼                                 │             │
│  ┌──────────────────────────────────┐             │             │
│  │       Distraction Detected?      │             │             │
│  │       (confidence > threshold)   │             │             │
│  └──────────────┬───────────────────┘             │             │
│                 │ YES                             │             │
│                 ▼                                 │             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    API Model Call                         │  │
│  │         (Opus 4.5 / GPT-5.2 / Grok / etc.)               │  │
│  │                                                           │  │
│  │  Input: persona + session context + distraction details   │  │
│  │  Output: callout line ("Twitter? Really? The essay...")   │  │
│  └──────────────┬────────────────────────────────────────────┘  │
│                 │                                               │
│                 ▼                                               │
│  ┌──────────────────────────────────┐                          │
│  │             TTS                  │                          │
│  │   (System / ElevenLabs / OpenAI) │                          │
│  └──────────────┬───────────────────┘                          │
│                 │                                               │
│                 ▼                                               │
│           🔊 Audio Output                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Validate Local Vision Model

**Goal:** Can SmolVLM (or similar) reliably detect distraction from webcam + screen captures?

### SmolVLM Options (as of SmolVLM2)

| Model | Params | Memory | Notes |
|-------|--------|--------|-------|
| SmolVLM2-256M-Instruct | 256M | <1GB VRAM | World's smallest VLM. Experimental but surprisingly capable. |
| SmolVLM2-500M-Instruct | 500M | ~1-2GB | Sweet spot — nearly matches 2.2B on many tasks |
| SmolVLM2-2.2B-Instruct | 2.2B | ~4-5GB | Best performance, still runs on M-series Macs easily |

All three have MLX support from day zero — ideal for Mac native inference.

The 256M model can run inference on a single image with under 1GB of GPU RAM. Even the 256M outperforms Idefics 80B from 17 months ago, so "small" doesn't mean weak.

**Recommendation:** Start testing with 500M. If it's accurate enough, ship with that. If not, bump to 2.2B. The 256M is worth trying but might struggle with nuanced detection.

### Detection Cases to Test

**Webcam detection:**
1. ✅ Looking at screen, engaged (baseline - NOT distracted)
2. 📱 Phone in hand / looking at phone
3. 👀 Looking away from screen (left, right, behind)
4. 😴 Head down / appears asleep
5. 🫥 Zoned out stare (eyes glazed, not focused)
6. 🗣️ Talking to someone else in room

**Screen detection:**
1. ✅ Work-related content (doc, IDE, PDF, etc.) - NOT distracted
2. 🐦 Social media (Twitter, Instagram, TikTok, Reddit)
3. 📺 Entertainment (YouTube, Netflix, games)
4. 💬 Messaging apps (Discord, iMessage, WhatsApp)
5. 📧 Email (ambiguous - could be work or distraction)
6. 🛒 Shopping sites

### Test Script Approach

```python
# Pseudocode for phase 1 testing

import cv2
from llama_cpp import Llama  # or mlx equivalent

# Load SmolVLM
model = load_smolvlm("smolvlm-500m-gguf")

# Capture sources
webcam = cv2.VideoCapture(0)
# screen capture via pyautogui or native API

def get_distraction_assessment(webcam_frame, screen_frame):
    prompt = """Look at these two images:
    Image 1: Webcam view of person at computer
    Image 2: Their screen contents
    
    Assess if this person is distracted from work.
    
    Output JSON only:
    {
      "distracted": true/false,
      "confidence": 0.0-1.0,
      "reason": "phone" | "social_media" | "entertainment" | "looking_away" | "zoned_out" | "messaging" | "none",
      "detail": "brief description"
    }
    """
    
    response = model.generate(
        prompt=prompt,
        images=[webcam_frame, screen_frame]
    )
    
    return parse_json(response)

# Test loop
while True:
    webcam_frame = capture_webcam()
    screen_frame = capture_screen()
    
    result = get_distraction_assessment(webcam_frame, screen_frame)
    
    print(f"Distracted: {result['distracted']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Reason: {result['reason']}")
    print(f"Detail: {result['detail']}")
    print("---")
    
    time.sleep(3)  # Sample every 3 seconds
```

### Success Criteria for Phase 1

**Must have:**
- [ ] Runs on M1/M2/M3 Mac without melting
- [ ] Inference time < 2 seconds per sample
- [ ] Correctly identifies phone pickup > 80% of time
- [ ] Correctly identifies social media on screen > 80% of time
- [ ] False positive rate < 20% (doesn't yell at you when you're focused)

**Nice to have:**
- [ ] Can distinguish between YouTube lecture vs YouTube entertainment
- [ ] Handles varied lighting conditions
- [ ] Works with multiple monitor setups

### Phase 1 Tasks

1. [ ] Get SmolVLM2-500M running locally via MLX
2. [ ] Build simple test harness: webcam capture → model → JSON output
3. [ ] Test detection prompt variations (see below)
4. [ ] Collect sample images for each distraction case (take ~20 photos of yourself)
5. [ ] Run accuracy tests, log results in a spreadsheet
6. [ ] If 500M isn't accurate enough, try 2.2B
7. [ ] Measure inference speed and resource usage

### Detection Prompt Variations to Test

**Simple binary:**
```
Is this person distracted from their computer work? Answer only: YES or NO
```

**Structured JSON:**
```
Look at this image of a person at their computer.
Is the person distracted? Output ONLY valid JSON:
{"distracted": true/false, "confidence": 0.0-1.0, "reason": "phone|looking_away|zoned_out|none"}
```

**Two-step (might be more reliable):**
```
Describe what this person is doing in one sentence.
```
Then parse the description for keywords (phone, looking away, etc.)

**With screen context (if passing both webcam + screen):**
```
Image 1: Webcam showing person at desk
Image 2: Their computer screen

Assess: Is this person focused on productive work?
Output JSON: {"focused": true/false, "webcam_issue": "none|phone|looking_away|zoned_out", "screen_issue": "none|social_media|entertainment|messaging"}
```

### Quick MLX Test Script

```python
# test_smolvlm_detection.py
# Run: python test_smolvlm_detection.py

import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.utils import load_image
import cv2
import json
import time

# Load model
model_path = "mlx-community/SmolVLM2-500M-Video-Instruct-mlx"
model, processor = load(model_path)

# Capture webcam frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

# Save temp image
cv2.imwrite("/tmp/webcam_test.jpg", frame)

# Test prompt
prompt = """Look at this image of a person at their computer.
Is the person distracted from work? Output ONLY valid JSON:
{"distracted": true/false, "reason": "phone|looking_away|zoned_out|none"}"""

# Generate
start = time.time()
response = generate(
    model,
    processor,
    prompt,
    ["/tmp/webcam_test.jpg"],
    max_tokens=100,
    temperature=0.1  # Low temp for consistent JSON
)
elapsed = time.time() - start

print(f"Response: {response}")
print(f"Inference time: {elapsed:.2f}s")

# Try to parse JSON
try:
    result = json.loads(response)
    print(f"Parsed: {result}")
except:
    print("Failed to parse JSON - might need prompt tuning")
```

---

## Phase 2: End-to-End Prototype (after Phase 1 validated)

Once we know local detection works:

1. [ ] Wire up API model call on distraction trigger
2. [ ] Add TTS output
3. [ ] Build minimal UI (session start, active call view, end session)
4. [ ] Test full loop: detect → generate callout → speak

---

## Open Questions

**Model questions:**
- SmolVLM2 seems like the move — 500M or 2.2B depending on accuracy needs
- MLX is probably the inference backend for Mac (native Apple Silicon support, SmolVLM2 has MLX ready from day zero)
- Single image (webcam OR screen) vs combined input? Probably test both

**Detection questions:**
- Sample rate: 2 sec vs 3 sec vs 5 sec?
- Confidence threshold: 0.7? 0.8? Configurable?
- Cooldown after callout: 30 sec? 60 sec?
- Should webcam and screen be assessed separately or together?

**UX questions:**
- How to handle ambiguous cases (email, YouTube lectures)?
- User whitelist for allowed sites/apps?
- "Thinking" mode - user is looking away but thinking about work?

---

## Resources

- SmolVLM2 announcement: https://huggingface.co/blog/smolvlm2
- SmolVLM 256M/500M announcement: https://huggingface.co/blog/smolervlm
- SmolVLM2-500M-Instruct model: https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
- MLX examples (Apple Silicon inference): https://github.com/ml-explore/mlx-examples
- MLX SmolVLM2 models: https://huggingface.co/mlx-community (search for SmolVLM2)
- Original webcam demo inspiration: https://github.com/ngxson/smolvlm-realtime-webcam
- HuggingFace SmolLM/SmolVLM repo: https://github.com/huggingface/smollm5

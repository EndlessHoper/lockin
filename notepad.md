# Lock In - Development Notes

## Current Goal: Demo + Article

**Article vision:**
- Proof of concept while building the full app
- Angle: "I'm impressed that these capabilities are now open and local"
- Shoutout to HuggingFace for making this accessible
- End with: "You might think 'but how is this useful?' - I'm exploring that with Lock In"
- Target audience: General (not just developers)
- Format: Both video demo + blog post

**Demo concept:**
- Show local VLM describing what it sees in real-time
- The "wow factor": small models running locally can now see and understand

**Article angles:**
- Just got a MacBook Pro M5 - perfect hook for "MLX on Apple Silicon" angle
- Can highlight how Apple's MLX framework makes local AI accessible
- "I just got an M5, here's what local vision models can do on it"

---

## File Structure

```
detector/
├── terminal_capture.py   # CLI webcam demo (press SPACE to capture)
├── server.py             # Browser-based focus detector
├── requirements.txt
└── Ministral_3B_WebGPU/  # UI reference (nice prompt suggestions pattern)

archive/            # Old experiments (moved)
```

---

## Models to Try (MLX, 4-bit quantized)

| Model | Size | Downloads | Notes |
|-------|------|-----------|-------|
| Qwen2-VL-2B-Instruct-4bit | 2B | 4K | Good balance speed/quality |
| Qwen2.5-VL-3B-Instruct-4bit | 3B | 1.4K | Newer version |
| Phi-3.5-vision-instruct-4bit | ~4B | 94 | Alternative architecture |
| SmolVLM2-500M-8bit-skip-vision | 500M | 59 | Smallest, fastest |

**SmolVLM issues:** Doesn't follow structured prompts well, outputs garbage like "1, 1, 2" instead of actual answers.

**Cleared from cache:** SmolVLM models removed to try Qwen2-VL instead.

---

## Learnings / Gotchas

### Image orientation matters!
- Webcam display is mirrored (feels natural like a mirror)
- BUT don't send mirrored image to model - text becomes unreadable
- Solution: flip for display only, send original to model

### mlx_vlm returns GenerationResult object
- Don't use `str(response)` - gives object dump
- Use `response.text` to get actual generated text

### Small models don't follow complex prompts
- Structured output (YES/NO format) doesn't work well with SmolVLM
- Better to ask simple questions and parse natural language response
- Qwen2-VL may be better at instruction following

### Image size affects speed dramatically
- Full 1080p webcam = very slow
- Resize to 384px = much faster
- Vision transformers scale quadratically with image size

### Model size vs speed tradeoff
- SmolVLM2 500M 8-bit: ~1.2s but poor instruction following
- SmolVLM2 2.2B 4-bit: ~4.5s, still mediocre output
- Try Qwen2-VL 2B 4-bit next

---

## UI Inspiration: Ministral_3B_WebGPU

Nice pattern for prompts:
```typescript
const PROMPTS = {
  default: "Describe what you see in one sentence.",
  suggestions: [
    "Describe what you see in one sentence.",
    "What is the color of my shirt?",
    "Identify any text or written content visible.",
    "What emotions or actions are being portrayed?",
    "Name the object I am holding in my hand.",
  ],
}
```

**The trick:** Prompt augmentation behind the scenes. The user sees/types a simple prompt like "Describe the image", but the model actually receives an enriched version:

> "Describe what is happening. Include details about colors, what the person is wearing, visible objects, actions being performed..."

The model gives a detailed, insightful response. User thinks "wow it understood so much from my simple question!" - but actually the system guided it with a richer prompt.

**Takeaway for our demo:** Consider invisible prompt enrichment. Simple user-facing prompts, detailed model-facing prompts.

---

## TODO

- [x] Clear SmolVLM models from cache
- [ ] Test Qwen2-VL-2B-Instruct-4bit
- [ ] Record demo video for article
- [ ] Write article draft
- [ ] Consider: continuous mode vs manual capture for demo

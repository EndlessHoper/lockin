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
- The "wow factor": small models (500M params) running locally can now see and understand

**Article angles:**
- Just got a MacBook Pro M5 - perfect hook for "MLX on Apple Silicon" angle
- Can highlight how Apple's MLX framework makes local AI accessible
- "I just got an M5, here's what local vision models can do on it"

---

## File Structure

```
detector/           # Current demo code
├── terminal_capture.py   # CLI webcam demo (press SPACE to capture)
├── server.py             # Browser-based focus detector
└── requirements.txt

archive/            # Old experiments (moved)
├── focus.py, gemmafocus.py, etc.
```

---

## Models Tested

| Model | Quantization | Speed | Notes |
|-------|--------------|-------|-------|
| SmolVLM2-500M-Video-Instruct-mlx | BF16 | Slow | Original, non-quantized |
| SmolVLM2-500M-Video-Instruct-mlx-8bit-skip-vision | 8-bit | Fast (~1.2s) | Current default |
| EZCon/SmolVLM2-2.2B-Instruct-4bit-mlx | 4-bit | TBD | Larger model, might be better quality |

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
- Structured output (YES/NO format) doesn't work well
- Better to ask simple questions and parse natural language response
- "Describe this image" works better than "Answer YES or NO to these 3 questions"

### Image size affects speed dramatically
- Full 1080p webcam = very slow
- Resize to 384px = much faster
- Vision transformers scale quadratically with image size

### Browser demo vs Python:
- Browser WebGPU + Q4 can be faster than Python MLX with BF16
- Quantization matters more than framework choice

---

## TODO

- [ ] Test the 2.2B 4-bit model for quality comparison
- [ ] Record demo video for article
- [ ] Write article draft
- [ ] Consider: continuous mode vs manual capture for demo

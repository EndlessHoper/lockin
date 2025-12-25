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
- Inspired by someone else's demo that had much faster inference
- The "wow factor": small models (500M params) running locally can now see and understand

**Current blocker:** Inference too slow to be impressive (was using non-quantized BF16 model)

**Article angles:**
- Just got a MacBook Pro M5 - perfect hook for "MLX on Apple Silicon" angle
- Can highlight how Apple's MLX framework makes local AI accessible
- "I just got an M5, here's what local vision models can do on it"

---

## File Guide

| File | Purpose |
|------|---------|
| `focus.py` | Testing distraction detector (Together AI) |
| `gemmafocus.py` | Testing distraction detector (LM Studio) |
| `whatyousee.py` | Model describes what it sees (LM Studio) |
| `localwhatyousee.py` | Model describes what it sees (local MLX) |
| `server.py` | Main backend with local MLX inference |

---

## Inference Speed Investigation

**Issue found:** `localwhatyousee.py` has `MAX_SIDE = 0` (no image resizing)

Compare:
- `server.py`: MAX_SIDE = 384 (resizes to max 384px)
- `localwhatyousee.py`: MAX_SIDE = 0 (full resolution!)

A 1080p webcam frame is ~25x more pixels than a 384px resized version. Vision transformers scale roughly quadratically with attention, so this could be a major factor.

**Other optimization ideas to try:**
- Set MAX_SIDE to 384 or 512
- Try the 256M model instead of 500M
- Reduce MAX_TOKENS further (32 vs 64)
- Check if mlx_vlm is using the GPU properly

---

## Old Notes

> ok. can you rewrite it to use together ai's hosted gemma model as per the code example below.

```python
from together import Together

client = Together()

response = client.chat.completions.create(
    model="google/gemma-3n-E4B-it",
    messages=[
      {
        "role": "user",
        "content": "What are some fun things to do in New York?"
      }
    ]
)
print(response.choices[0].message.content)
```

key is tgp_v1_kLnGH6eAJ9TWVttIv54R5rHBwfZMuIT1D3qx_us97yg
i want you to hardcode it. there is no risk. its my code. its not going anywehre.

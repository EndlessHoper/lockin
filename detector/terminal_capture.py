"""
Local Vision Model Demo - See What AI Sees

Runs a vision-language model locally on Apple Silicon using MLX.
Captures webcam frames and describes what it sees in real-time.

Requirements:
    pip install mlx-vlm pillow opencv-python

Usage:
    python terminal_capture.py
"""

import time
import cv2
from PIL import Image

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# Model options - uncomment the one you want to use
MODEL = "mlx-community/SmolVLM2-500M-Video-Instruct-mlx-8bit-skip-vision"
# MODEL = "EZCon/SmolVLM2-2.2B-Instruct-4bit-mlx"

# Generation settings
MAX_TOKENS = 64
TEMPERATURE = 0.0
MAX_IMAGE_SIZE = 384

PROMPT = "What do you see in this image? Describe it in one sentence."


def load_model():
    """Load the vision model and processor."""
    print(f"Loading model: {MODEL}")
    start = time.time()

    config = load_config(MODEL)
    model, processor = load(MODEL)

    print(f"Model loaded in {time.time() - start:.1f}s")
    return model, processor, config


def resize_image(image: Image.Image, max_side: int) -> Image.Image:
    """Resize image to fit within max_side while preserving aspect ratio."""
    w, h = image.size
    scale = max_side / max(w, h)
    if scale >= 1.0:
        return image
    return image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)


def capture_frame(cap) -> Image.Image | None:
    """Capture a frame from webcam and convert to PIL Image."""
    ret, frame = cap.read()
    if not ret:
        return None
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def describe_image(model, processor, config, image: Image.Image) -> tuple[str, float]:
    """Run inference on an image and return description + elapsed time."""
    image = resize_image(image, MAX_IMAGE_SIZE)
    prompt = apply_chat_template(processor, config, PROMPT, num_images=1)

    start = time.time()
    response = generate(
        model,
        processor,
        prompt,
        [image],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    elapsed = time.time() - start

    # Extract text from GenerationResult object
    if hasattr(response, 'text'):
        text = response.text.strip()
    else:
        text = str(response).strip()

    # Clean up - take first line only
    if "\n" in text:
        text = text.split("\n")[0]

    return text, elapsed


def main():
    print("=" * 60)
    print("Local Vision Model Demo")
    print("=" * 60)
    print()

    # Load model
    model, processor, config = load_model()
    print()

    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Webcam ready. Press 'q' to quit, SPACE to capture.")
    print()

    try:
        while True:
            # Show live preview
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror for display only (so it feels like a mirror)
            display_frame = cv2.flip(frame, 1)
            cv2.putText(
                display_frame, "SPACE: Capture | Q: Quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

            cv2.imshow("Local Vision Demo", display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                # Capture and analyze (use ORIGINAL unflipped frame for model)
                print("Capturing...")

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)

                # Run inference
                description, elapsed = describe_image(model, processor, config, image)

                print(f"[{elapsed:.2f}s] {description}")
                print()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()

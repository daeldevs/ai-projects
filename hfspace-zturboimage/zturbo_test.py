import os
import re
import shutil
from gradio_client import Client

def sanitize_filename(text):
    # Keep filename safe for Windows, Linux, macOS
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text[:60]  # limit length

def generate_image(prompt="Hello!!", h=1024, w=1024, steps=9, seed=42, randomize=True):
    client = Client("mrfakename/Z-Image-Turbo")

    print("Requesting image…")

    # Run API
    result = client.predict(
        prompt=prompt,
        height=int(h),
        width=int(w),
        num_inference_steps=int(steps),
        seed=int(seed),
        randomize_seed=bool(randomize),
        api_name="/generate_image"
    )

    image_path, seed_used = result

    print("Seed used:", seed_used)
    print("Temp path returned:", image_path)

    # Make sure it's a real file
    if not (isinstance(image_path, str) and os.path.exists(image_path)):
        print("ERROR: Space returned non-path:", image_path)
        return None

    # Create output directory
    out_dir = "generated"
    os.makedirs(out_dir, exist_ok=True)

    # Build filename
    base_name = sanitize_filename(prompt)
    out_file = f"{base_name}_seed{seed_used}.webp"
    out_path = os.path.join(out_dir, out_file)

    # Copy image from temp to output directory
    shutil.copy(image_path, out_path)

    print("Saved locally as:", out_path)
    return out_path


if __name__ == "__main__":
    generate_image(
        prompt="A futuristic neon city with flying cars",
        seed=42,
        randomize=False
    )

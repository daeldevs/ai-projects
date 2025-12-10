from gradio_client import Client

def debug_generate():
    client = Client("mrfakename/Z-Image-Turbo")

    print("Requesting image...\n")

    result = client.predict(
        prompt="A futuristic neon city with flying cars",
        height=1024,
        width=1024,
        num_inference_steps=9,
        seed=42,
        randomize_seed=False,
        api_name="/generate_image"
    )

    print("=== RAW RESULT ===")
    print(result)
    print("==================\n")

    # Unpack result
    image_info, seed_used = result

    print("=== TYPES ===")
    print("image_info type:", type(image_info))
    print("seed_used type:", type(seed_used))
    print("==================")

    return result


if __name__ == "__main__":
    debug_generate()

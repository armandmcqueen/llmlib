import enum

import typer
from PIL import Image
import subprocess
import base64
import io
import replicate
from io import BytesIO
import requests
from queue import Queue
import threading

app = typer.Typer()


def display_image(image_path):
    subprocess.run(["open", image_path])


def encode_image_webp(image: Image.Image, quality: int = 85) -> str:
    """
    Encode a Pillow image to base64 string in WebP format.

    Args:
        image (Image.Image): Pillow image object.
        quality (int): WebP quality setting (0-100). Default is 85.

    Returns:
        str: Base64 encoded string of the image in WebP format

    Raises:
        ValueError: If the input is not a valid Pillow image or quality is out of range.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a Pillow Image object")

    if not 0 <= quality <= 100:
        raise ValueError("Quality must be between 0 and 100")

    buffer = io.BytesIO()
    image.save(buffer, format="WebP", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return encoded


class FluxRatio(str, enum.Enum):
    r1_1 = "1:1"
    r2_3 = "2:3"
    r3_2 = "3:2"
    r4_5 = "4:5"
    r5_4 = "5:4"
    r9_16 = "9:16"
    r16_8 = "16:9"


# TODO: Enumerate possible ratios as enum
def flux_generate(description: str, ratio: FluxRatio) -> Image.Image:
    output_url = replicate.run(
        "black-forest-labs/flux-pro",
        input={
            "steps": 25,
            "prompt": description,
            "guidance": 5,
            "interval": 1,
            "aspect_ratio": ratio.value,
            "output_format": "webp",
            "output_quality": 80,
            "safety_tolerance": 5
        }
    )

    response = requests.get(output_url)
    img = Image.open(BytesIO(response.content))
    return img

def _flux_generate(prompt: str, ratio: FluxRatio, result_queue: Queue):
    # print(f"_flux_generate: Generating image with Flux: {prompt[:50]}...")
    img = flux_generate(prompt, ratio)
    result_queue.put(img)

# TODO: Make the ratio variable per prompt
# TODO: Note that the result list is not guaranteed to be in the same order as the input list
def parallel_flux_generate(prompts: list[str], ratio: FluxRatio) -> list[Image.Image]:
    """Generate multiple images in parallel using threads."""
    # print(f"Generating {versions} images with Flux...")
    # rich.inspect(prompt)
    threads = []
    result_queue = Queue()

    for prompt in prompts:
        thread = threading.Thread(target=_flux_generate, args=(prompt, ratio, result_queue))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    return results

@app.command()
def flux(description: str):
    typer.echo(f"Flux: {description}")
    output = flux_generate(description, ratio=FluxRatio.r1_1)
    # Write image to output.webp
    output.save("output.webp")
    # Display image
    display_image("output.webp")


if __name__ == "__main__":
    app()
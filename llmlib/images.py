from PIL import Image
import base64
import io


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

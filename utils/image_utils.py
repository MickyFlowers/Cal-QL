import io

import numpy as np
from PIL import Image


def compress_image(img_pil: Image.Image, format: str = "PNG")-> np.ndarray:
    with io.BytesIO() as output:
        img_pil.save(output, format="PNG")
        png_data = output.getvalue()
    return np.frombuffer(png_data, dtype='uint8')


def decompress_image(img_bytes: np.ndarray) -> Image.Image:
    with io.BytesIO(img_bytes.tobytes()) as input:
        img_pil = Image.open(input).convert('RGB')
    return img_pil

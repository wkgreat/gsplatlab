import numpy as np
from PIL import Image


def save_to_image(arr: np.ndarray, image_path: str, factor=255, flipY=False):
    image = (arr * factor).astype(np.uint8)
    if flipY:
        image = np.flipud(image)
    Image.fromarray(image).save(image_path)

from PIL import Image
import io

from models.model import get_label


def predict(data: bytes) -> str:
    image: Image.Image = Image.open(io.BytesIO(data))

    return get_label(image)

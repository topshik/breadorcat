from PIL import Image
import io


def predict(data: bytes) -> str:
    image: Image.Image = Image.open(io.BytesIO(data))
    image.resize((256, 256))
    return 'To be continued'

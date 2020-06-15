from aiohttp import web
import io
from PIL import Image
import torch
import torchvision
from torchvision import transforms

from models.model import BreadOr


class PredictionHandler:
    def __init__(self) -> None:
        vgg16 = torchvision.models.vgg16()
        self.breador_loaded = BreadOr(vgg16)
        self.breador_loaded.load_state_dict(torch.load('breador.pth', map_location='cpu'))
        self.breador_loaded.eval()

        self.transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

        self.prediction_map = dict(zip([0, 1, 2], ['bread', 'cat', 'other']))

    async def handle_predict(self, request) -> web.Response:
        data = await request.read()
        print(f'got {len(data)} bytes')

        image: Image.Image = Image.open(io.BytesIO(data))
        image = self.transform(image).unsqueeze(dim=0)
        probs = self.breador_loaded(image)
        prediction = self.prediction_map[int(probs.max(1)[1][0])]

        print(f'Prediction is {prediction}')
        return web.json_response({'prediction': prediction})

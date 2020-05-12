from aiohttp import web
from PIL import Image
import io

from backend import pipeline

@web.middleware
async def cors_middleware(request, handler):
    response: web.Response = await handler(request)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', '*')
    response.headers.add('Access-Control-Allow-Headers', '*')
    return response


async def handle_options(request) -> web.Response:
    return web.Response()


async def handle_predict(request: web.Request) -> web.Response:
    data = await request.read()
    print(f'got {len(data)} bytes')

    result = pipeline.predict(data)

    print(f'Prediction is {result}')
    return web.json_response({'prediction': result})


def create_app() -> web.Application:
    app = web.Application(middlewares=[cors_middleware])

    app.router.add_options('/{tail:.*}', handle_options)
    app.router.add_post('/predict/', handle_predict)

    return app

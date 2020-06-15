from typing import Tuple

from aiohttp import web

from backend.pipeline import PredictionHandler


@web.middleware
async def cors_middleware(request, handler):
    response: web.Response = await handler(request)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', '*')
    response.headers.add('Access-Control-Allow-Headers', '*')
    return response


async def handle_options(request) -> web.Response:
    return web.Response()


def create_app() -> web.Application:
    prediction_handler = PredictionHandler()
    app = web.Application(middlewares=[cors_middleware])

    app.router.add_options('/{tail:.*}', handle_options)
    app.router.add_post('/predict/', prediction_handler.handle_predict)

    return app

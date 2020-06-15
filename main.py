import asyncio
import argparse

from aiohttp import web
import uvloop

from backend import app

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int)
    args = parser.parse_args()

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    instance = app.create_app()
    web.run_app(app=instance, port=(args.port or 8080))

#!/usr/bin/env python
import random
import websockets
import asyncio
import os
import numpy as np
import cv2
import json

import base64


#!/usr/bin/env python

# WS server that sends messages at random intervals


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture('video2.mp4')

    def __del__(self):
        self.video.release()

    def get_frame(self):
        while True:
            ret, image = self.video.read()
            ret, jpeg = cv2.imencode('.jpg', image)
            base6 = base64.b64encode(jpeg.tobytes())
            yield base6.decode('utf-8')


def gen(camera):

    while True:
        image = next(camera.get_frame())

        yield(image)


async def time(websocket, path):
    camera = VideoCamera()
    i = 0
    while True:
        i = i+1
        data = next(gen(camera))
        js = {'data': data, "infos": [
            {
                "key": "1",
                "name": "John Brown",
                "age": 32,
                "address": "New York No. 1 Lake Park",
            },
            {
                "key": "2",
                "name": "Jim Green",
                "age": 42,
                "address": "London No. 1 Lake Park",
            },
            {
                "key": "3",
                "name": "Joe Black",
                "age": 32,
                "address": "Sidney No. 1 Lake Park",
            },
            {
                "key": "4",
                "name": "Jim Red",
                "age": 32,
                "address": "London No. 2 Lake Park",
            },
        ], }

        res_bytes = json.dumps(js).encode('utf-8')
        base6 = base64.b64encode(res_bytes)
        message = base6.decode('utf-8')

        await websocket.send(message)


start_server = websockets.serve(time, "127.0.0.1", 5678)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

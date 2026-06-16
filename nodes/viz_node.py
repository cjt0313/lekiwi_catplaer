"""Visualization node: bridges ZMQ messages to a WebSocket-based web frontend."""

import sys
import asyncio
import json
from pathlib import Path

sys.path.insert(0, ".")

import zmq
import zmq.asyncio
from aiohttp import web

from common.config import (
    FAKE_CAMERA_PUB, STATE_PUB, PLAY_PLAN_PUB,
    BASE_PLAN_PUB, ARM_PLAN_PUB, CONTROLLER_PUB,
    VIZ_HTTP_PORT,
)

ALL_PUBS = [
    FAKE_CAMERA_PUB, STATE_PUB, PLAY_PLAN_PUB,
    BASE_PLAN_PUB, ARM_PLAN_PUB, CONTROLLER_PUB,
]

state = {
    "robot_pose": None,
    "cat_bbox_3d": None,
    "occupancy_grid": None,
    "world_state": None,
    "play_target": None,
    "base_path": None,
    "arm_target": None,
    "robot_command": None,
}

grid_dirty = False
connected_ws: set[web.WebSocketResponse] = set()


async def zmq_listener():
    ctx = zmq.asyncio.Context()
    poller = zmq.asyncio.Poller()
    sockets = []

    for addr in ALL_PUBS:
        sock = ctx.socket(zmq.SUB)
        sock.connect(addr)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sockets.append(sock)
        poller.register(sock, zmq.POLLIN)

    global grid_dirty
    while True:
        events = await poller.poll(timeout=50)
        for sock, _ in events:
            parts = await sock.recv_multipart()
            if len(parts) != 2:
                continue
            topic = parts[0].decode()
            try:
                data = json.loads(parts[1])
                payload = data.get("payload")
            except (json.JSONDecodeError, KeyError):
                continue

            if topic in state:
                state[topic] = payload
                if topic == "occupancy_grid":
                    grid_dirty = True


async def broadcast_loop():
    global grid_dirty
    while True:
        if connected_ws:
            msg = {
                "robot_pose": state["robot_pose"],
                "cat_bbox_3d": state["cat_bbox_3d"],
                "world_state": state["world_state"],
                "play_target": state["play_target"],
                "base_path": state["base_path"],
                "arm_target": state["arm_target"],
                "robot_command": state["robot_command"],
            }
            if grid_dirty:
                msg["occupancy_grid"] = state["occupancy_grid"]
                grid_dirty = False

            payload = json.dumps(msg)
            dead = set()
            for ws in connected_ws:
                try:
                    await ws.send_str(payload)
                except Exception:
                    dead.add(ws)
            connected_ws -= dead

        await asyncio.sleep(0.1)


async def index_handler(request):
    html_path = Path(__file__).parent.parent / "viz" / "index.html"
    return web.FileResponse(html_path)


async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    connected_ws.add(ws)
    print(f"[Viz] WebSocket client connected ({len(connected_ws)} total)")

    # Send full state including grid on connect
    init_msg = json.dumps(state)
    try:
        await ws.send_str(init_msg)
    except Exception:
        pass

    async for _ in ws:
        pass

    connected_ws.discard(ws)
    print(f"[Viz] WebSocket client disconnected ({len(connected_ws)} total)")
    return ws


async def on_startup(app):
    app["zmq_task"] = asyncio.create_task(zmq_listener())
    app["broadcast_task"] = asyncio.create_task(broadcast_loop())


async def on_cleanup(app):
    app["zmq_task"].cancel()
    app["broadcast_task"].cancel()


def main():
    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", ws_handler)

    print(f"[Viz] Starting at http://localhost:{VIZ_HTTP_PORT}")
    web.run_app(app, host="0.0.0.0", port=VIZ_HTTP_PORT, print=None)


if __name__ == "__main__":
    main()

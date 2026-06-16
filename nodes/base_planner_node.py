"""Base planner node: A* path planning on 2D occupancy grid."""

import sys
import time
import math
import heapq

sys.path.insert(0, ".")

from common.config import (
    STATE_PUB, PLAY_PLAN_PUB, FAKE_CAMERA_PUB, BASE_PLAN_PUB,
    MAP_RESOLUTION, MAP_WIDTH, MAP_HEIGHT, MAP_ORIGIN, ROBOT_COLLISION_RADIUS,
)
from common.types import MsgType
from common.zmq_message import make_publisher, make_subscriber, publish, receive, ZmqMessage


def world_to_grid(x, y):
    col = int((x - MAP_ORIGIN[0]) / MAP_RESOLUTION)
    row = int((y - MAP_ORIGIN[1]) / MAP_RESOLUTION)
    return row, col


def grid_to_world(row, col):
    x = col * MAP_RESOLUTION + MAP_ORIGIN[0]
    y = row * MAP_RESOLUTION + MAP_ORIGIN[1]
    return x, y


def inflate_grid(grid_data):
    """Inflate obstacles by ROBOT_COLLISION_RADIUS to prevent collisions."""
    inflate_cells = int(math.ceil(ROBOT_COLLISION_RADIUS / MAP_RESOLUTION))
    inflated = list(grid_data)

    for r in range(MAP_HEIGHT):
        for c in range(MAP_WIDTH):
            if grid_data[r * MAP_WIDTH + c] == 1:
                for dr in range(-inflate_cells, inflate_cells + 1):
                    for dc in range(-inflate_cells, inflate_cells + 1):
                        if dr * dr + dc * dc <= inflate_cells * inflate_cells:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < MAP_HEIGHT and 0 <= nc < MAP_WIDTH:
                                inflated[nr * MAP_WIDTH + nc] = 1

    return inflated


def astar(grid_data, start, goal):
    sr, sc = start
    gr, gc = goal

    if not (0 <= sr < MAP_HEIGHT and 0 <= sc < MAP_WIDTH):
        return []
    if not (0 <= gr < MAP_HEIGHT and 0 <= gc < MAP_WIDTH):
        return []
    if grid_data[gr * MAP_WIDTH + gc] == 1:
        return []

    def heuristic(r, c):
        return math.sqrt((r - gr) ** 2 + (c - gc) ** 2)

    open_set = [(heuristic(sr, sc), 0, sr, sc)]
    came_from = {}
    g_score = {(sr, sc): 0}
    closed = set()

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_set:
        _, cost, r, c = heapq.heappop(open_set)

        if (r, c) in closed:
            continue
        closed.add((r, c))

        if r == gr and c == gc:
            path = []
            cur = (gr, gc)
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append((sr, sc))
            path.reverse()
            return path

        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < MAP_HEIGHT and 0 <= nc < MAP_WIDTH):
                continue
            if grid_data[nr * MAP_WIDTH + nc] == 1:
                continue
            if (nr, nc) in closed:
                continue

            move_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
            new_g = g_score[(r, c)] + move_cost

            if new_g < g_score.get((nr, nc), float("inf")):
                g_score[(nr, nc)] = new_g
                f = new_g + heuristic(nr, nc)
                heapq.heappush(open_set, (f, new_g, nr, nc))
                came_from[(nr, nc)] = (r, c)

    return []


class BasePlannerNode:
    def __init__(self):
        self.pub = make_publisher(BASE_PLAN_PUB)
        self.state_sub = make_subscriber(STATE_PUB, [MsgType.WORLD_STATE.value])
        self.play_sub = make_subscriber(PLAY_PLAN_PUB, [MsgType.PLAY_TARGET.value])
        self.cam_sub = make_subscriber(FAKE_CAMERA_PUB, [MsgType.OCCUPANCY_GRID.value])

        self.grid_data = None
        self.play_target = None
        self.robot_pose = None
        self.seq = 0
        self.planned = False

    def poll(self):
        for _ in range(10):
            msg = receive(self.cam_sub, timeout_ms=1)
            if not msg:
                break
            if msg.header.msg_type == MsgType.OCCUPANCY_GRID.value:
                self.grid_data = msg.payload["data"]

        msg = receive(self.play_sub, timeout_ms=1)
        if msg:
            self.play_target = msg.payload
            self.planned = False

        msg = receive(self.state_sub, timeout_ms=1)
        if msg:
            self.robot_pose = msg.payload.get("robot_pose")
            if msg.payload.get("state") == "SELECT_PLAY_TARGET":
                self.planned = False

    def plan_path(self):
        if self.planned:
            return
        if not self.grid_data or not self.play_target or not self.robot_pose:
            return

        start = world_to_grid(self.robot_pose["x"], self.robot_pose["y"])
        target = self.play_target["target_pose"]
        goal = world_to_grid(target["x"], target["y"])

        inflated = inflate_grid(self.grid_data)
        grid_path = astar(inflated, start, goal)

        if not grid_path:
            print("[BasePlanner] A* failed, no path found")
            payload = {"path": []}
        else:
            # Keep every 3rd point for smooth visual + last point
            sampled = grid_path[::3]
            if sampled[-1] != grid_path[-1]:
                sampled.append(grid_path[-1])

            world_path = []
            for i, (r, c) in enumerate(sampled):
                x, y = grid_to_world(r, c)
                if i < len(sampled) - 1:
                    nx, ny = grid_to_world(sampled[i + 1][0], sampled[i + 1][1])
                    yaw = math.atan2(ny - y, nx - x)
                else:
                    yaw = target["yaw"]
                world_path.append([x, y, yaw])

            payload = {"path": world_path}
            print(f"[BasePlanner] A* path found: {len(grid_path)} cells -> {len(world_path)} waypoints")

        msg = ZmqMessage.create(MsgType.BASE_PATH, payload, "base_planner_node", seq=self.seq)
        publish(self.pub, msg)
        self.seq += 1
        self.planned = True

    def run(self):
        print("[BasePlanner] Started")
        while True:
            self.poll()
            self.plan_path()
            time.sleep(0.1)


def main():
    time.sleep(0.5)
    node = BasePlannerNode()
    node.run()


if __name__ == "__main__":
    main()

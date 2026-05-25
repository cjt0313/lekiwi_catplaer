"""Play planner node: selects a playful peek position around the cat.

Algorithm:
  Sample candidates in an annulus around the cat, then score each by:
  - partial visibility (cat sees ~half of robot footprint)
  - proximity to obstacles (partial cover)
  - distance to desired play radius
  - travel cost from current robot position
  Pick the highest-scoring reachable free candidate.
"""

import sys
import time
import math

sys.path.insert(0, ".")

from common.config import (
    STATE_PUB, PLAY_PLAN_PUB, DESIRED_CAT_DISTANCE,
    MAP_RESOLUTION, MAP_WIDTH, MAP_HEIGHT, MAP_ORIGIN,
)
from common.types import MsgType
from common.zmq_message import make_publisher, make_subscriber, publish, receive, ZmqMessage

ROBOT_RADIUS = 0.18
IDEAL_OBSTACLE_DIST = 0.35
ANNULUS_RADII = [0.8, 1.0, 1.2]
NUM_ANGLES = 32


def world_to_grid(x, y):
    col = int((x - MAP_ORIGIN[0]) / MAP_RESOLUTION)
    row = int((y - MAP_ORIGIN[1]) / MAP_RESOLUTION)
    return row, col


def is_free(grid_data, row, col):
    if 0 <= row < MAP_HEIGHT and 0 <= col < MAP_WIDTH:
        return grid_data[row * MAP_WIDTH + col] == 0
    return False


def line_of_sight_free(x0, y0, x1, y1, grid_data):
    """Bresenham ray trace on the grid. Returns True if no occupied cell blocks the line.
    Skips the start cell (source may be on an obstacle edge)."""
    r0, c0 = world_to_grid(x0, y0)
    r1, c1 = world_to_grid(x1, y1)

    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dc - dr

    r, c = r0, c0
    first = True
    while True:
        if not first and not is_free(grid_data, r, c):
            return False
        first = False
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr

    return True


def compute_visible_ratio(cat_x, cat_y, x, y, grid_data):
    """Check how many of 5 footprint points around (x,y) have line-of-sight from cat."""
    points = [
        (x, y),
        (x + ROBOT_RADIUS, y),
        (x - ROBOT_RADIUS, y),
        (x, y + ROBOT_RADIUS),
        (x, y - ROBOT_RADIUS),
    ]
    visible = sum(
        1 for px, py in points
        if line_of_sight_free(cat_x, cat_y, px, py, grid_data)
    )
    return visible / len(points)


def clearance_to_obstacle(x, y, grid_data):
    """Find distance to nearest occupied cell within a search radius."""
    row, col = world_to_grid(x, y)
    search_cells = int(0.8 / MAP_RESOLUTION)
    min_dist = float("inf")

    for dr in range(-search_cells, search_cells + 1):
        for dc in range(-search_cells, search_cells + 1):
            nr, nc = row + dr, col + dc
            if 0 <= nr < MAP_HEIGHT and 0 <= nc < MAP_WIDTH:
                if grid_data[nr * MAP_WIDTH + nc] == 1:
                    dist = math.sqrt(dr * dr + dc * dc) * MAP_RESOLUTION
                    if dist < min_dist:
                        min_dist = dist

    return min_dist


class PlayPlannerNode:
    def __init__(self):
        self.pub = make_publisher(PLAY_PLAN_PUB)
        self.state_sub = make_subscriber(STATE_PUB, [MsgType.WORLD_STATE.value])
        self.seq = 0
        self.last_published_state = None

    def plan(self, world_state):
        state = world_state.get("state")
        if state != "SELECT_PLAY_TARGET":
            self.last_published_state = None
            return
        if self.last_published_state == "SELECT_PLAY_TARGET":
            return

        cat = world_state.get("cat")
        robot = world_state.get("robot_pose")
        grid = world_state.get("map")
        if not cat or not robot or not grid:
            return

        cat_x, cat_y = cat["center"][0], cat["center"][1]
        robot_x, robot_y = robot["x"], robot["y"]
        grid_data = grid["data"]

        best_candidate = None
        best_score = -float("inf")

        for r in ANNULUS_RADII:
            for i in range(NUM_ANGLES):
                theta = 2.0 * math.pi * i / NUM_ANGLES
                cx = cat_x + r * math.cos(theta)
                cy = cat_y + r * math.sin(theta)

                row, col = world_to_grid(cx, cy)
                if not is_free(grid_data, row, col):
                    continue

                clearance = clearance_to_obstacle(cx, cy, grid_data)
                if clearance < ROBOT_RADIUS:
                    continue

                visible_ratio = compute_visible_ratio(cat_x, cat_y, cx, cy, grid_data)
                if visible_ratio < 0.2 or visible_ratio > 0.9:
                    continue

                dist_to_cat = math.sqrt((cx - cat_x) ** 2 + (cy - cat_y) ** 2)
                travel_cost = math.sqrt((cx - robot_x) ** 2 + (cy - robot_y) ** 2)

                # Partial visibility: peaks at 0.5
                partial_vis_score = 1.0 - abs(visible_ratio - 0.5) / 0.5
                partial_vis_score = max(0.0, partial_vis_score)

                # Distance to cat: peaks at DESIRED_CAT_DISTANCE
                distance_score = 1.0 - abs(dist_to_cat - DESIRED_CAT_DISTANCE) / 0.5
                distance_score = max(0.0, distance_score)

                # Near obstacle for cover
                obstacle_score = 1.0 if 0.25 <= clearance <= 0.6 else 0.0

                # Safety clearance
                clearance_score = min(clearance / 0.6, 1.0)

                score = (
                    3.0 * partial_vis_score
                    + 2.0 * distance_score
                    + 1.5 * obstacle_score
                    + 1.0 * clearance_score
                    - 0.5 * travel_cost
                )

                if score > best_score:
                    best_score = score
                    best_candidate = (cx, cy)

        if best_candidate is None:
            print("[PlayPlanner] No valid candidate found")
            return

        tx, ty = best_candidate
        yaw = math.atan2(cat_y - ty, cat_x - tx)

        payload = {
            "target_pose": {"x": tx, "y": ty, "yaw": yaw},
            "desired_cat_distance": DESIRED_CAT_DISTANCE,
            "score": best_score,
        }
        msg = ZmqMessage.create(MsgType.PLAY_TARGET, payload, "play_planner_node", seq=self.seq)
        publish(self.pub, msg)
        self.seq += 1
        self.last_published_state = state
        print(f"[PlayPlanner] Target: ({tx:.2f}, {ty:.2f}, yaw={yaw:.2f}) score={best_score:.2f}")

    def run(self):
        print("[PlayPlanner] Started")
        while True:
            msg = receive(self.state_sub, timeout_ms=100)
            if msg:
                self.plan(msg.payload)


def main():
    time.sleep(0.5)
    node = PlayPlannerNode()
    node.run()


if __name__ == "__main__":
    main()

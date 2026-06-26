#!/usr/bin/env python
"""
Bridge between the cat-player pipeline and the real LeKiwi robot.

Subscribes to internal ROBOT_COMMAND messages (ZMQ SUB) and forwards
translated commands to the real robot (ZMQ PUSH on 192.168.100.1:5555).

Modes:
    --arm-only   Only send arm flirting motion (base stays still)
    --base-only  Only forward base velocity commands (arm at home)
    (default)    Both arm flirting + base path following

Usage:
    # Start robot host first (on robot or via SSH):
    #   python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_lekiwi --robot.cameras='{}' --host.connection_time_s=600

    python scripts/robot_bridge.py --arm-only    # Test arm flirting
    python scripts/robot_bridge.py --base-only   # Test path following
    python scripts/robot_bridge.py               # Full mode
"""

import argparse
import json
import math
import signal
import sys
import time

import zmq
import numpy as np

# Robot connection
REMOTE_IP = "192.168.100.1"
PORT_CMD = 5555
PORT_OBS = 5556
FPS = 30

# Path subscription (from viz_detection)
PERCEPTION_PUB = "tcp://127.0.0.1:5560"
PATH_TOPIC = "cat_bbox_3d"

# Control signal (from viz button)
CONTROL_SIGNAL_PUB = "tcp://127.0.0.1:5570"
CONTROL_SIGNAL_TOPIC = "bridge_control"

# Path following controller gains
KP_LINEAR = 0.5      # proportional gain for linear velocity
KP_ANGULAR = 1.0     # proportional gain for angular velocity
MAX_LINEAR_SPEED = 0.15   # m/s
MAX_ANGULAR_SPEED = 0.5   # rad/s
WAYPOINT_THRESHOLD = 0.08 # meters - switch to next waypoint when this close

# Arm joint names (LeKiwi protocol)
ARM_JOINTS = [
    "arm_shoulder_pan.pos",
    "arm_shoulder_lift.pos",
    "arm_elbow_flex.pos",
    "arm_wrist_flex.pos",
    "arm_wrist_roll.pos",
    "arm_gripper.pos",
]

# Home pose (degrees) - recorded from physical robot
HOME_POSE_DEG = [-0.1, -102.5, 96.8, -75.8, -0.5, 3.3]

# Arm flirting config (joint 4 = wrist_flex, index 3)
FLIRT_JOINT_INDEX = 3
FLIRT_CENTER_DEG = HOME_POSE_DEG[3]  # oscillate around home position
FLIRT_AMPLITUDE_DEG = 15.0
FLIRT_FREQUENCY_HZ = 1.0


class RobotBridge:
    def __init__(self, arm_enabled: bool, base_enabled: bool):
        self.arm_enabled = arm_enabled
        self.base_enabled = base_enabled
        self.running = True
        self.enabled = False  # starts disabled, enable via viz button
        self.arm_positions = list(HOME_POSE_DEG)
        self.base_vx = 0.0
        self.base_vy = 0.0
        self.base_wz = 0.0

        # Path following state
        self.path = []         # list of (x, y) waypoints in robot base frame
        self.waypoint_idx = 0  # current target waypoint

        # ZMQ setup
        self.ctx = zmq.Context()

        # PUSH to robot
        self.cmd_socket = self.ctx.socket(zmq.PUSH)
        self.cmd_socket.connect(f"tcp://{REMOTE_IP}:{PORT_CMD}")
        self.cmd_socket.setsockopt(zmq.CONFLATE, 1)

        # PULL observations from robot
        self.obs_socket = self.ctx.socket(zmq.PULL)
        self.obs_socket.connect(f"tcp://{REMOTE_IP}:{PORT_OBS}")
        self.obs_socket.setsockopt(zmq.CONFLATE, 1)

        # SUB to perception path (from viz_detection)
        self.path_socket = None
        if self.base_enabled:
            self.path_socket = self.ctx.socket(zmq.SUB)
            self.path_socket.connect(PERCEPTION_PUB)
            self.path_socket.setsockopt_string(zmq.SUBSCRIBE, PATH_TOPIC)

        # SUB to control signal from viz
        self.control_socket = self.ctx.socket(zmq.SUB)
        self.control_socket.connect(CONTROL_SIGNAL_PUB)
        self.control_socket.setsockopt_string(zmq.SUBSCRIBE, CONTROL_SIGNAL_TOPIC)

        # E-stop handler
        signal.signal(signal.SIGINT, self._estop)
        signal.signal(signal.SIGTERM, self._estop)

    def _estop(self, signum, frame):
        print("\n[E-STOP] Sending zero commands...")
        self.running = False
        self._send_stop()

    def _send_stop(self):
        action = {
            ARM_JOINTS[i]: HOME_POSE_DEG[i] for i in range(6)
        }
        action.update({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        try:
            self.cmd_socket.send_string(json.dumps(action))
            time.sleep(0.05)
            self.cmd_socket.send_string(json.dumps(action))
        except Exception:
            pass

    def _read_path(self):
        """Read latest path from viz_detection (non-blocking). Updates self.path."""
        if not self.path_socket:
            return
        while self.path_socket.poll(0):
            parts = self.path_socket.recv_multipart()
            if len(parts) == 2:
                data = json.loads(parts[1].decode("utf-8"))
                payload = data.get("payload", {})
                new_path = payload.get("path")
                if new_path and len(new_path) > 1:
                    self.path = new_path
                    self.waypoint_idx = 1  # skip first point (robot position)
                elif not payload.get("visible", False):
                    self.path = []
                    self.waypoint_idx = 0

    def _follow_path(self):
        """Proportional controller: compute velocity toward next waypoint.

        The path is in robot base frame (robot is at origin, facing +X).
        Each waypoint is (x, y) in meters from robot's current position.
        """
        if not self.path or self.waypoint_idx >= len(self.path):
            self.base_vx = 0.0
            self.base_vy = 0.0
            self.base_wz = 0.0
            return

        # Target waypoint (in base frame, robot at origin)
        wx, wy = self.path[self.waypoint_idx]

        # Distance to waypoint
        dist = math.sqrt(wx * wx + wy * wy)

        # If close enough, advance to next waypoint
        if dist < WAYPOINT_THRESHOLD:
            self.waypoint_idx += 1
            if self.waypoint_idx >= len(self.path):
                self.base_vx = 0.0
                self.base_vy = 0.0
                self.base_wz = 0.0
                return
            wx, wy = self.path[self.waypoint_idx]
            dist = math.sqrt(wx * wx + wy * wy)

        # Proportional control: drive toward waypoint
        # vx = forward (toward waypoint X), vy = strafe (toward waypoint Y)
        if dist > 0.01:
            self.base_vx = np.clip(KP_LINEAR * wx, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED)
            self.base_vy = np.clip(KP_LINEAR * wy, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED)
        else:
            self.base_vx = 0.0
            self.base_vy = 0.0

        # Angular: face the direction of travel
        angle_to_target = math.atan2(wy, wx)
        self.base_wz = np.clip(KP_ANGULAR * angle_to_target, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)

    def _read_control_signal(self):
        """Check for enable/disable from viz button (non-blocking)."""
        while self.control_socket.poll(0):
            parts = self.control_socket.recv_multipart()
            if len(parts) == 2:
                msg = json.loads(parts[1].decode("utf-8"))
                if msg.get("estop"):
                    print("\n[E-STOP] Triggered from GUI!")
                    self.enabled = False
                    self._send_stop()
                    continue
                new_state = msg.get("enabled", self.enabled)
                if new_state != self.enabled:
                    self.enabled = new_state
                    state_str = "ENABLED" if self.enabled else "DISABLED"
                    print(f"\n[CONTROL] Robot execution {state_str}")
                    if not self.enabled:
                        self._send_stop()

    def _compute_flirt(self, t: float):
        """Sinusoidal oscillation on the flirt joint."""
        angle = FLIRT_CENTER_DEG + FLIRT_AMPLITUDE_DEG * math.sin(
            2 * math.pi * FLIRT_FREQUENCY_HZ * t
        )
        self.arm_positions[FLIRT_JOINT_INDEX] = angle

    def _read_observation(self):
        """Read robot observation (non-blocking) to keep arm positions in sync."""
        try:
            msg = self.obs_socket.recv_string(zmq.NOBLOCK)
            obs = json.loads(msg)
            for i, joint in enumerate(ARM_JOINTS):
                if i != FLIRT_JOINT_INDEX or not self.arm_enabled:
                    self.arm_positions[i] = obs.get(joint, self.arm_positions[i])
        except zmq.Again:
            pass

    def connect(self) -> bool:
        """Wait for first observation from robot."""
        print(f"Connecting to LeKiwi at {REMOTE_IP}...")
        poller = zmq.Poller()
        poller.register(self.obs_socket, zmq.POLLIN)
        socks = dict(poller.poll(5000))
        if self.obs_socket not in socks:
            print("ERROR: Timeout waiting for robot. Is lekiwi_host running?")
            return False

        msg = self.obs_socket.recv_string()
        obs = json.loads(msg)
        for i, joint in enumerate(ARM_JOINTS):
            self.arm_positions[i] = obs.get(joint, self.arm_positions[i])
        print(f"Connected! Arm positions: {[f'{p:.1f}' for p in self.arm_positions]}")
        self._go_home()
        return True

    def _go_home(self):
        """Move arm to home position over ~1 second."""
        print("Moving arm to home position...")
        steps = FPS  # 30 steps = 1 second
        start_positions = list(self.arm_positions)
        for step in range(steps):
            t = (step + 1) / steps
            for i in range(6):
                self.arm_positions[i] = start_positions[i] + t * (HOME_POSE_DEG[i] - start_positions[i])
            action = {ARM_JOINTS[i]: self.arm_positions[i] for i in range(6)}
            action.update({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
            self.cmd_socket.send_string(json.dumps(action))
            time.sleep(1.0 / FPS)
        print("Arm at home.")

    def run(self):
        """Main control loop at 30 Hz."""
        mode_str = []
        if self.arm_enabled:
            mode_str.append("ARM_FLIRT")
        if self.base_enabled:
            mode_str.append("BASE_PATH")
        print(f"Running in mode: {' + '.join(mode_str)}")
        print("Press Ctrl+C for e-stop.\n")

        t_start = time.time()
        while self.running:
            t0 = time.perf_counter()
            t_elapsed = time.time() - t_start

            # Check for enable/disable from viz
            self._read_control_signal()

            if not self.enabled:
                time.sleep(1.0 / FPS)
                sys.stdout.write("\r[BRIDGE] DISABLED - waiting for enable signal   ")
                sys.stdout.flush()
                continue

            # Read path and compute velocity
            if self.base_enabled:
                self._read_path()
                self._follow_path()

            # Compute arm flirting
            if self.arm_enabled:
                self._compute_flirt(t_elapsed)

            # Convert base velocity: pipeline uses rad/s, robot uses deg/s
            theta_deg_s = math.degrees(self.base_wz) if self.base_enabled else 0.0

            # Build action
            action = {ARM_JOINTS[i]: self.arm_positions[i] for i in range(6)}
            action["x.vel"] = self.base_vx if self.base_enabled else 0.0
            action["y.vel"] = self.base_vy if self.base_enabled else 0.0
            action["theta.vel"] = theta_deg_s

            # Send to robot
            self.cmd_socket.send_string(json.dumps(action))

            # Read observation feedback
            self._read_observation()

            # Status line
            flirt_str = f"j4={self.arm_positions[FLIRT_JOINT_INDEX]:+6.1f}°" if self.arm_enabled else "arm:off"
            base_str = f"vx={self.base_vx:+.3f} vy={self.base_vy:+.3f} wz={theta_deg_s:+.1f}°/s"
            wp_str = f"wp={self.waypoint_idx}/{len(self.path)}" if self.path else "no path"
            sys.stdout.write(f"\r[BRIDGE] {flirt_str} | {base_str} | {wp_str}   ")
            sys.stdout.flush()

            # Maintain FPS
            elapsed = time.perf_counter() - t0
            time.sleep(max(1.0 / FPS - elapsed, 0.0))

        # Cleanup
        self._send_stop()
        print("\nStopped.")

    def close(self):
        self.obs_socket.close()
        self.cmd_socket.close()
        self.control_socket.close()
        if self.path_socket:
            self.path_socket.close()
        self.ctx.term()


def run_base_test(bridge: RobotBridge, speed: float = 0.1, duration: float = 3.0):
    """Drive forward, then stop. Simple test of base velocity control."""
    print(f"Base test: forward at {speed} m/s for {duration}s...")
    t_start = time.time()
    while bridge.running and (time.time() - t_start) < duration:
        t0 = time.perf_counter()
        action = {ARM_JOINTS[i]: bridge.arm_positions[i] for i in range(6)}
        action["x.vel"] = speed
        action["y.vel"] = 0.0
        action["theta.vel"] = 0.0
        bridge.cmd_socket.send_string(json.dumps(action))
        bridge._read_observation()
        elapsed = time.perf_counter() - t0
        time.sleep(max(1.0 / FPS - elapsed, 0.0))
    bridge._send_stop()
    print("Base test done.")


def main():
    parser = argparse.ArgumentParser(description="Bridge pipeline to real LeKiwi robot")
    parser.add_argument("--arm-only", action="store_true", help="Only arm flirting (no base)")
    parser.add_argument("--base-only", action="store_true", help="Only forward base velocity from pipeline (no arm)")
    parser.add_argument("--test-base", action="store_true", help="Test: drive forward 0.1 m/s for 3s then stop")
    parser.add_argument("--speed", type=float, default=0.1, help="Speed for --test-base (m/s, default 0.1)")
    parser.add_argument("--duration", type=float, default=3.0, help="Duration for --test-base (seconds, default 3)")
    args = parser.parse_args()

    if args.arm_only and args.base_only:
        print("ERROR: Cannot specify both --arm-only and --base-only")
        sys.exit(1)

    arm_enabled = not args.base_only and not args.test_base
    base_enabled = not args.arm_only

    bridge = RobotBridge(arm_enabled=arm_enabled, base_enabled=(base_enabled and not args.test_base))
    if not bridge.connect():
        sys.exit(1)

    if args.test_base:
        run_base_test(bridge, speed=args.speed, duration=args.duration)
        bridge.close()
        return

    try:
        bridge.run()
    finally:
        bridge.close()


if __name__ == "__main__":
    main()

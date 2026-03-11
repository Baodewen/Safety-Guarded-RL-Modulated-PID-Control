import argparse
import json
import math
import random
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


WEB_DIR = Path(__file__).parent / "web"
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "ppo_policy.pt"
LEGACY_MODEL_PATH = Path(__file__).parent / "models" / "ppo_policy.json"
OBS_KEYS = [
    "front_distance",
    "left_clearance",
    "right_clearance",
    "center_offset",
    "heading_error",
    "speed",
    "turn_rate",
    "nearest_obstacle_distance",
    "risk_trend",
]
ACTION_RANGES = {
    "speed_scale": (0.22, 1.20),
    "kp_scale": (0.75, 2.10),
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def encode_observation(obs: Dict[str, float]) -> List[float]:
    return [
        clamp(obs["front_distance"] / 4.0, 0.0, 1.5),
        clamp(obs["left_clearance"] / 2.0, 0.0, 1.5),
        clamp(obs["right_clearance"] / 2.0, 0.0, 1.5),
        clamp(obs["center_offset"], -1.5, 1.5),
        clamp(obs["heading_error"] / math.pi, -1.0, 1.0),
        clamp(obs["speed"] / 1.6, 0.0, 1.5),
        clamp(obs["turn_rate"] / 2.0, -1.5, 1.5),
        clamp(obs["nearest_obstacle_distance"] / 2.0, 0.0, 1.5),
        clamp(obs["risk_trend"], -1.0, 1.0),
    ]


def map_action_to_modulation(raw_action: List[float], action_ranges: Dict[str, Tuple[float, float]] | Dict[str, List[float]]) -> Dict[str, float]:
    speed_low, speed_high = action_ranges["speed_scale"]
    kp_low, kp_high = action_ranges["kp_scale"]
    speed_scale = speed_low + 0.5 * (float(raw_action[0]) + 1.0) * (speed_high - speed_low)
    kp_scale = kp_low + 0.5 * (float(raw_action[1]) + 1.0) * (kp_high - kp_low)
    return {
        "speed_scale": clamp(speed_scale, speed_low, speed_high),
        "kp_scale": clamp(kp_scale, kp_low, kp_high),
    }


class ActorOnly(nn.Module if nn is not None else object):
    def __init__(self, obs_dim: int, hidden_sizes: List[int], action_dim: int) -> None:
        if nn is None:
            raise RuntimeError("torch is not available")
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = obs_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.Tanh())
            in_dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.actor = nn.Linear(in_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor(self.backbone(obs)))


class TorchPolicyRuntime:
    def __init__(self, model_path: Path) -> None:
        if torch is None or nn is None:
            raise RuntimeError("torch is not installed")
        checkpoint = torch.load(model_path, map_location="cpu")
        self.hidden_sizes = list(checkpoint.get("hidden_sizes", [128, 128]))
        self.obs_keys = list(checkpoint.get("obs_keys", OBS_KEYS))
        self.action_ranges = checkpoint.get("action_ranges", ACTION_RANGES)
        self.metadata = checkpoint.get("metadata", {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorOnly(len(self.obs_keys), self.hidden_sizes, 2)
        self.model.load_state_dict(checkpoint["actor_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def infer(self, observation: Dict[str, float]) -> Dict[str, float]:
        obs_tensor = torch.tensor([encode_observation(observation)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            raw_action = self.model(obs_tensor).squeeze(0).detach().cpu().tolist()
        return map_action_to_modulation(raw_action, self.action_ranges)


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float
    vx: float = 0.0
    vy: float = 0.0
    dynamic: bool = False
    min_y: float = 0.0
    max_y: float = 0.0

    def step(self, dt: float) -> None:
        if not self.dynamic:
            return
        self.x += self.vx * dt
        self.y += self.vy * dt
        if self.y - self.radius < self.min_y or self.y + self.radius > self.max_y:
            self.vy *= -1.0
            self.y = clamp(self.y, self.min_y + self.radius, self.max_y - self.radius)


@dataclass
class RobotState:
    x: float
    y: float
    theta: float
    v: float = 0.0
    w: float = 0.0


@dataclass
class PIDController:
    kp: float = 2.4
    kd: float = 0.45
    speed_kp: float = 2.0
    max_speed: float = 1.45
    max_turn_rate: float = 1.9
    last_heading_error: float = 0.0

    def compute(self, target_speed: float, target_heading: float, robot: RobotState, dt: float, kp_scale: float) -> Dict[str, float]:
        heading_error = wrap_angle(target_heading - robot.theta)
        derivative = (heading_error - self.last_heading_error) / dt
        self.last_heading_error = heading_error

        effective_kp = self.kp * kp_scale
        raw_w = effective_kp * heading_error + self.kd * derivative
        turn_rate_cmd = clamp(raw_w, -self.max_turn_rate, self.max_turn_rate)

        speed_error = target_speed - robot.v
        accel_cmd = self.speed_kp * speed_error
        unclamped_speed = robot.v + accel_cmd * dt
        speed_cmd = clamp(unclamped_speed, 0.0, self.max_speed)

        return {
            "target_heading": target_heading,
            "heading_error": heading_error,
            "kp_effective": effective_kp,
            "turn_rate_cmd": turn_rate_cmd,
            "speed_cmd": speed_cmd,
        }


class CorridorEnv:
    def __init__(self) -> None:
        self.dt = 0.1
        self.robot_radius = 0.22
        self.base_target_speed = 1.28
        self.goal_tolerance = 0.4
        self.scene_name = "open_corridor"
        self.seed = 7
        self.random = random.Random(self.seed)
        self.corridor_length = 18.0
        self.corridor_width = 4.0
        self.obstacles: List[Obstacle] = []
        self.robot = RobotState(0.8, 0.0, 0.0)
        self.controller = PIDController()
        self.policy_mode = "safety_guarded_rl_pid"
        self.time_s = 0.0
        self.step_count = 0
        self.episode_status = "running"
        self.triggered_events: List[str] = []
        self.prev_front_distance = 99.0
        self.no_progress_steps = 0
        self.max_progress_x = self.robot.x
        self.path: List[Tuple[float, float]] = []
        self.policy_runtime = None
        self.model_info = self._load_model_info()
        self.reset(self.scene_name, self.policy_mode, self.seed)

    def _load_model_info(self) -> Dict[str, str]:
        if DEFAULT_MODEL_PATH.exists() and torch is not None:
            try:
                self.policy_runtime = TorchPolicyRuntime(DEFAULT_MODEL_PATH)
                metadata = self.policy_runtime.metadata
                return {
                    "source": "file",
                    "path": str(DEFAULT_MODEL_PATH),
                    "status": "loaded",
                    "name": metadata.get("algo", "ppo") + "-trained-policy",
                    "device": str(self.policy_runtime.device),
                }
            except Exception as exc:
                return {
                    "source": "file",
                    "path": str(DEFAULT_MODEL_PATH),
                    "status": f"load_failed: {exc}",
                    "name": "failed-policy",
                    "device": "cpu",
                }
        if LEGACY_MODEL_PATH.exists():
            return {
                "source": "file",
                "path": str(LEGACY_MODEL_PATH),
                "status": "legacy_placeholder_only",
                "name": "json-placeholder",
                "device": "cpu",
            }
        return {
            "source": "built-in",
            "path": "",
            "status": "fallback",
            "name": "heuristic-risk-modulator",
            "device": "cpu",
        }

    def list_scenes(self) -> List[str]:
        return ["open_corridor", "single_obstacle", "narrow_gap", "crossing_dynamic"]

    def list_policy_modes(self) -> List[str]:
        return ["fixed_pid", "rl_pid", "safety_guarded_rl_pid"]

    def reset(self, scene_name: str, policy_mode: str, seed: int) -> Dict[str, object]:
        self.scene_name = scene_name if scene_name in self.list_scenes() else "open_corridor"
        self.policy_mode = policy_mode if policy_mode in self.list_policy_modes() else "safety_guarded_rl_pid"
        self.seed = int(seed)
        self.random = random.Random(self.seed)
        self.time_s = 0.0
        self.step_count = 0
        self.episode_status = "running"
        self.triggered_events = []
        self.prev_front_distance = 99.0
        self.no_progress_steps = 0
        self.corridor_length, self.corridor_width, self.obstacles = self._build_scene(self.scene_name)
        self.robot = RobotState(0.8, 0.0, 0.0)
        self.controller = PIDController()
        self.max_progress_x = self.robot.x
        self.path = [(self.robot.x, self.robot.y)]
        return self.snapshot()

    def set_policy_mode(self, policy_mode: str) -> Dict[str, object]:
        if policy_mode in self.list_policy_modes():
            self.policy_mode = policy_mode
        return self.snapshot()

    def _build_scene(self, scene_name: str) -> Tuple[float, float, List[Obstacle]]:
        width = 4.0
        length = 18.0
        if scene_name == "open_corridor":
            return length, width, []
        if scene_name == "single_obstacle":
            return length, width, [
                Obstacle(x=8.0, y=0.65, radius=0.55),
                Obstacle(x=12.6, y=-0.75, radius=0.45),
            ]
        if scene_name == "narrow_gap":
            return length, width, [
                Obstacle(x=8.5, y=1.08, radius=0.62),
                Obstacle(x=8.7, y=-1.05, radius=0.62),
                Obstacle(x=12.4, y=0.0, radius=0.45),
            ]
        return length, width, [
            Obstacle(x=7.4, y=0.9, radius=0.45),
            Obstacle(x=11.0, y=-0.9, radius=0.45),
            Obstacle(x=9.8, y=-1.1, radius=0.36, vy=0.55, dynamic=True, min_y=-1.35, max_y=1.35),
        ]

    def _distance_to_obstacles(self, x: float, y: float) -> Tuple[float, float, float]:
        nearest = 99.0
        nearest_left = 99.0
        nearest_right = 99.0
        for obstacle in self.obstacles:
            dist = math.hypot(x - obstacle.x, y - obstacle.y) - obstacle.radius - self.robot_radius
            nearest = min(nearest, dist)
            if obstacle.y >= y:
                nearest_left = min(nearest_left, dist)
            else:
                nearest_right = min(nearest_right, dist)
        return nearest, nearest_left, nearest_right

    def _forward_distance(self, robot: RobotState) -> float:
        min_distance = self.corridor_length - robot.x
        for obstacle in self.obstacles:
            dx = obstacle.x - robot.x
            if dx <= 0.0:
                continue
            lateral = abs(obstacle.y - robot.y)
            corridor = obstacle.radius + self.robot_radius + 0.15
            if lateral <= corridor:
                min_distance = min(min_distance, max(0.0, dx - corridor))
        return min_distance

    def _local_heading_reference(self, robot: RobotState) -> float:
        attract_y = -0.9 * robot.y
        repel_y = 0.0
        for obstacle in self.obstacles:
            dx = obstacle.x - robot.x
            if -0.5 <= dx <= 3.0:
                dy = robot.y - obstacle.y
                distance_sq = max(dx * dx + dy * dy, 0.2)
                repel_y += (dy / math.sqrt(distance_sq)) * (1.1 / distance_sq)
        desired_y = attract_y + repel_y
        lookahead = 2.4
        return math.atan2(desired_y, lookahead)

    def _observation(self) -> Dict[str, float]:
        front_distance = self._forward_distance(self.robot)
        nearest_obs, nearest_left_obs, nearest_right_obs = self._distance_to_obstacles(self.robot.x, self.robot.y)
        left_boundary = self.corridor_width / 2.0 - (self.robot.y + self.robot_radius)
        right_boundary = self.corridor_width / 2.0 + (self.robot.y - self.robot_radius)
        left_clearance = min(left_boundary, nearest_left_obs)
        right_clearance = min(right_boundary, nearest_right_obs)
        target_heading = self._local_heading_reference(self.robot)
        heading_error = wrap_angle(target_heading - self.robot.theta)
        progress_remaining = self.corridor_length - self.robot.x
        return {
            "front_distance": round(front_distance, 4),
            "left_clearance": round(left_clearance, 4),
            "right_clearance": round(right_clearance, 4),
            "center_offset": round(self.robot.y / max(self.corridor_width / 2.0 - self.robot_radius, 0.1), 4),
            "heading_error": round(heading_error, 4),
            "speed": round(self.robot.v, 4),
            "turn_rate": round(self.robot.w, 4),
            "progress_remaining": round(progress_remaining, 4),
            "nearest_obstacle_distance": round(nearest_obs, 4),
            "risk_trend": round(front_distance - self.prev_front_distance, 4),
            "target_heading": round(target_heading, 4),
        }

    def _heuristic_modulation(self, observation: Dict[str, float]) -> Dict[str, float]:
        front = observation["front_distance"]
        nearest = observation["nearest_obstacle_distance"]
        center_mag = abs(observation["center_offset"])
        heading_mag = abs(observation["heading_error"])
        side_min = min(observation["left_clearance"], observation["right_clearance"])

        risk_front = 1.0 - clamp(front / 4.0, 0.0, 1.0)
        risk_near = 1.0 - clamp(nearest / 1.6, 0.0, 1.0)
        edge_risk = 1.0 - clamp(side_min / 1.3, 0.0, 1.0)
        directional_need = clamp(heading_mag / 0.9, 0.0, 1.0)

        speed_scale = clamp(1.12 - 0.52 * risk_front - 0.34 * risk_near - 0.16 * edge_risk, 0.22, 1.20)
        kp_scale = clamp(0.94 + 0.75 * directional_need + 0.32 * risk_near - 0.18 * center_mag, 0.75, 2.05)
        return {"speed_scale": speed_scale, "kp_scale": kp_scale, "source": self.model_info["name"]}

    def _policy_modulation(self, observation: Dict[str, float]) -> Dict[str, float]:
        if self.policy_mode == "fixed_pid":
            return {"speed_scale": 1.0, "kp_scale": 1.0, "source": "fixed"}
        if self.policy_runtime is not None and self.model_info.get("status") == "loaded":
            modulation = self.policy_runtime.infer(observation)
            modulation["source"] = self.model_info["name"]
            return modulation
        return self._heuristic_modulation(observation)

    def _safety_filter(self, raw_speed: float, raw_turn: float, observation: Dict[str, float]) -> Dict[str, object]:
        nearest = observation["nearest_obstacle_distance"]
        front = observation["front_distance"]
        side_min = min(observation["left_clearance"], observation["right_clearance"])
        risk_distance = min(nearest, front, side_min)

        speed_limit = raw_speed
        turn_limit = abs(raw_turn)
        override_reason = "none"
        override_strength = 0.0
        emergency_stop = False

        if risk_distance < 1.4:
            ratio = clamp((risk_distance - 0.35) / (1.4 - 0.35), 0.0, 1.0)
            speed_limit = min(speed_limit, 0.20 + 1.0 * ratio)
            turn_limit = min(turn_limit, 0.45 + 1.15 * ratio)
            override_reason = "risk_limit"
            override_strength = 1.0 - ratio

        if front < 0.38 or nearest < 0.30 or side_min < 0.18:
            speed_limit = 0.0
            turn_limit = min(turn_limit, 0.55)
            override_reason = "emergency_stop"
            override_strength = 1.0
            emergency_stop = True

        safe_speed = min(raw_speed, speed_limit)
        safe_turn = clamp(raw_turn, -turn_limit, turn_limit)
        overridden = abs(safe_speed - raw_speed) > 1e-6 or abs(safe_turn - raw_turn) > 1e-6
        return {
            "safe_speed_cmd": safe_speed,
            "safe_turn_cmd": safe_turn,
            "override_flag": overridden,
            "override_reason": override_reason if overridden else "none",
            "override_strength": round(override_strength if overridden else 0.0, 4),
            "emergency_stop": emergency_stop,
            "speed_limit": round(speed_limit, 4),
            "turn_limit": round(turn_limit, 4),
        }

    def _collision(self) -> bool:
        if abs(self.robot.y) + self.robot_radius >= self.corridor_width / 2.0:
            return True
        for obstacle in self.obstacles:
            if math.hypot(self.robot.x - obstacle.x, self.robot.y - obstacle.y) <= obstacle.radius + self.robot_radius:
                return True
        return False

    def step(self) -> Dict[str, object]:
        if self.episode_status != "running":
            return self.snapshot()

        for obstacle in self.obstacles:
            obstacle.step(self.dt)

        observation = self._observation()
        modulation = self._policy_modulation(observation)
        target_speed = self.base_target_speed * modulation["speed_scale"]
        pid_output = self.controller.compute(target_speed, observation["target_heading"], self.robot, self.dt, modulation["kp_scale"])

        raw_speed = pid_output["speed_cmd"]
        raw_turn = pid_output["turn_rate_cmd"]
        safety_status = {
            "safe_speed_cmd": raw_speed,
            "safe_turn_cmd": raw_turn,
            "override_flag": False,
            "override_reason": "none",
            "override_strength": 0.0,
            "emergency_stop": False,
            "speed_limit": raw_speed,
            "turn_limit": abs(raw_turn),
        }
        if self.policy_mode == "safety_guarded_rl_pid":
            safety_status = self._safety_filter(raw_speed, raw_turn, observation)

        self.robot.v = safety_status["safe_speed_cmd"]
        self.robot.w = safety_status["safe_turn_cmd"]
        self.robot.theta = wrap_angle(self.robot.theta + self.robot.w * self.dt)
        self.robot.x += self.robot.v * math.cos(self.robot.theta) * self.dt
        self.robot.y += self.robot.v * math.sin(self.robot.theta) * self.dt
        self.time_s += self.dt
        self.step_count += 1
        self.path.append((self.robot.x, self.robot.y))

        if self.robot.x > self.max_progress_x + 0.01:
            self.max_progress_x = self.robot.x
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1

        if safety_status["override_flag"]:
            self.triggered_events.append(safety_status["override_reason"])

        if self._collision():
            self.episode_status = "collision"
        elif self.robot.x >= self.corridor_length - self.goal_tolerance:
            self.episode_status = "success"
        elif self.no_progress_steps >= 140:
            self.episode_status = "stuck"

        self.prev_front_distance = observation["front_distance"]
        return self.snapshot(modulation=modulation, pid_output=pid_output, safety_status=safety_status, observation=observation)

    def snapshot(
        self,
        modulation: Dict[str, float] | None = None,
        pid_output: Dict[str, float] | None = None,
        safety_status: Dict[str, object] | None = None,
        observation: Dict[str, float] | None = None,
    ) -> Dict[str, object]:
        observation = observation or self._observation()
        modulation = modulation or self._policy_modulation(observation)
        target_speed = self.base_target_speed * modulation["speed_scale"]
        pid_output = pid_output or self.controller.compute(target_speed, observation["target_heading"], self.robot, self.dt, modulation["kp_scale"])
        safety_status = safety_status or {
            "safe_speed_cmd": pid_output["speed_cmd"],
            "safe_turn_cmd": pid_output["turn_rate_cmd"],
            "override_flag": False,
            "override_reason": "none",
            "override_strength": 0.0,
            "emergency_stop": False,
            "speed_limit": pid_output["speed_cmd"],
            "turn_limit": abs(pid_output["turn_rate_cmd"]),
        }

        nearest_obs, _, _ = self._distance_to_obstacles(self.robot.x, self.robot.y)
        side_min = min(observation["left_clearance"], observation["right_clearance"])

        return {
            "meta": {
                "time_s": round(self.time_s, 3),
                "step_count": self.step_count,
                "dt": self.dt,
                "scene_name": self.scene_name,
                "policy_mode": self.policy_mode,
                "seed": self.seed,
                "available_scenes": self.list_scenes(),
                "available_policy_modes": self.list_policy_modes(),
                "model_info": self.model_info,
            },
            "corridor": {
                "length": self.corridor_length,
                "width": self.corridor_width,
                "goal_x": self.corridor_length,
            },
            "robot": {
                "x": round(self.robot.x, 4),
                "y": round(self.robot.y, 4),
                "theta": round(self.robot.theta, 4),
                "v": round(self.robot.v, 4),
                "w": round(self.robot.w, 4),
                "radius": self.robot_radius,
                "path": [[round(x, 4), round(y, 4)] for x, y in self.path[-300:]],
            },
            "obstacles": [
                {
                    "x": round(obstacle.x, 4),
                    "y": round(obstacle.y, 4),
                    "radius": obstacle.radius,
                    "dynamic": obstacle.dynamic,
                    "vx": obstacle.vx,
                    "vy": obstacle.vy,
                }
                for obstacle in self.obstacles
            ],
            "observation_summary": observation,
            "rl_modulation": {
                "speed_scale": round(modulation["speed_scale"], 4),
                "kp_scale": round(modulation["kp_scale"], 4),
                "target_speed": round(target_speed, 4),
                "source": modulation["source"],
            },
            "pid_output": {
                "target_heading": round(pid_output["target_heading"], 4),
                "heading_error": round(pid_output["heading_error"], 4),
                "kp_effective": round(pid_output["kp_effective"], 4),
                "raw_speed_cmd": round(pid_output["speed_cmd"], 4),
                "raw_turn_cmd": round(pid_output["turn_rate_cmd"], 4),
            },
            "safe_action": {
                "speed_cmd": round(float(safety_status["safe_speed_cmd"]), 4),
                "turn_cmd": round(float(safety_status["safe_turn_cmd"]), 4),
            },
            "safety_status": {
                "override_flag": bool(safety_status["override_flag"]),
                "override_reason": safety_status["override_reason"],
                "override_strength": round(float(safety_status["override_strength"]), 4),
                "emergency_stop": bool(safety_status["emergency_stop"]),
                "speed_limit": round(float(safety_status["speed_limit"]), 4),
                "turn_limit": round(float(safety_status["turn_limit"]), 4),
                "nearest_obstacle_distance": round(nearest_obs, 4),
                "boundary_margin": round(side_min, 4),
                "override_count": len(self.triggered_events),
            },
            "episode_status": {
                "status": self.episode_status,
                "progress_x": round(self.robot.x, 4),
                "goal_x": self.corridor_length,
                "no_progress_steps": self.no_progress_steps,
                "triggered_events": self.triggered_events[-10:],
            },
        }


class DemoState:
    def __init__(self) -> None:
        self.env = CorridorEnv()
        self.lock = threading.Lock()

    def get_state(self) -> Dict[str, object]:
        with self.lock:
            return self.env.snapshot()

    def step(self) -> Dict[str, object]:
        with self.lock:
            return self.env.step()

    def reset(self, scene: str, policy_mode: str, seed: int) -> Dict[str, object]:
        with self.lock:
            return self.env.reset(scene, policy_mode, seed)

    def set_policy_mode(self, policy_mode: str) -> Dict[str, object]:
        with self.lock:
            return self.env.set_policy_mode(policy_mode)


APP_STATE = DemoState()


class DemoRequestHandler(BaseHTTPRequestHandler):
    server_version = "SafetyGuardedRLPIDDemo/0.2"

    def log_message(self, fmt: str, *args) -> None:
        return

    def _send_json(self, payload: Dict[str, object], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> Dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        if not raw:
            return {}
        return json.loads(raw)

    def _serve_static(self, relative_path: str) -> None:
        path = WEB_DIR / relative_path
        if path.is_dir():
            path = path / "index.html"
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        content_type = "text/plain; charset=utf-8"
        if path.suffix == ".html":
            content_type = "text/html; charset=utf-8"
        elif path.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif path.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"

        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            self._send_json(APP_STATE.get_state())
            return
        if parsed.path == "/" or parsed.path == "":
            self._serve_static("index.html")
            return
        self._serve_static(parsed.path.lstrip("/"))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json_body()
            if parsed.path == "/api/step":
                self._send_json(APP_STATE.step())
                return
            if parsed.path == "/api/reset":
                scene = str(payload.get("scene", "open_corridor"))
                policy_mode = str(payload.get("policy_mode", "safety_guarded_rl_pid"))
                seed = int(payload.get("seed", 7))
                self._send_json(APP_STATE.reset(scene, policy_mode, seed))
                return
            if parsed.path == "/api/set_policy_mode":
                policy_mode = str(payload.get("policy_mode", "safety_guarded_rl_pid"))
                self._send_json(APP_STATE.set_policy_mode(policy_mode))
                return
            self._send_json({"error": "Unknown endpoint"}, status=404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)


def main() -> None:
    parser = argparse.ArgumentParser(description="Safety-Guarded RL-PID demo server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DemoRequestHandler)
    print(f"Demo server listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

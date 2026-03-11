import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

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


class TrainCorridorEnv:
    def __init__(self, seed: int) -> None:
        self.base_seed = seed
        self.episode_index = 0
        self.random = random.Random(seed)
        self.dt = 0.1
        self.robot_radius = 0.22
        self.base_target_speed = 1.28
        self.goal_tolerance = 0.4
        self.max_steps = 260
        self.corridor_length = 18.0
        self.corridor_width = 4.0
        self.robot = RobotState(0.8, 0.0, 0.0)
        self.controller = PIDController()
        self.obstacles: List[Obstacle] = []
        self.scene_name = "open_corridor"
        self.step_count = 0
        self.prev_front_distance = 99.0
        self.max_progress_x = self.robot.x
        self.no_progress_steps = 0
        self.last_info: Dict[str, object] = {}
        self.reset()

    def _spawn_rng(self) -> random.Random:
        return random.Random(self.base_seed + self.episode_index * 9973)

    def _build_scene(self, rng: random.Random) -> Tuple[str, List[Obstacle]]:
        scene_name = rng.choice(["open_corridor", "single_obstacle", "narrow_gap", "crossing_dynamic"])
        obs: List[Obstacle] = []
        if scene_name == "single_obstacle":
            obs = [
                Obstacle(x=7.8 + rng.uniform(-0.4, 0.5), y=0.75 + rng.uniform(-0.25, 0.2), radius=0.52 + rng.uniform(-0.06, 0.05)),
                Obstacle(x=12.3 + rng.uniform(-0.4, 0.5), y=-0.8 + rng.uniform(-0.2, 0.2), radius=0.42 + rng.uniform(-0.05, 0.05)),
            ]
        elif scene_name == "narrow_gap":
            x_mid = 8.5 + rng.uniform(-0.35, 0.35)
            obs = [
                Obstacle(x=x_mid, y=1.05 + rng.uniform(-0.15, 0.12), radius=0.6 + rng.uniform(-0.04, 0.04)),
                Obstacle(x=x_mid + rng.uniform(-0.18, 0.18), y=-1.05 + rng.uniform(-0.12, 0.15), radius=0.6 + rng.uniform(-0.04, 0.04)),
                Obstacle(x=12.4 + rng.uniform(-0.5, 0.4), y=rng.uniform(-0.3, 0.3), radius=0.42 + rng.uniform(-0.04, 0.05)),
            ]
        elif scene_name == "crossing_dynamic":
            obs = [
                Obstacle(x=7.5 + rng.uniform(-0.4, 0.4), y=0.9 + rng.uniform(-0.2, 0.15), radius=0.44 + rng.uniform(-0.04, 0.04)),
                Obstacle(x=11.1 + rng.uniform(-0.5, 0.3), y=-0.85 + rng.uniform(-0.15, 0.15), radius=0.44 + rng.uniform(-0.04, 0.04)),
                Obstacle(
                    x=9.7 + rng.uniform(-0.2, 0.25),
                    y=-1.05 + rng.uniform(-0.12, 0.08),
                    radius=0.34 + rng.uniform(-0.03, 0.03),
                    vy=0.42 + rng.uniform(0.0, 0.25),
                    dynamic=True,
                    min_y=-1.35,
                    max_y=1.35,
                ),
            ]
        return scene_name, obs

    def reset(self) -> List[float]:
        self.episode_index += 1
        rng = self._spawn_rng()
        self.scene_name, self.obstacles = self._build_scene(rng)
        self.robot = RobotState(0.8, rng.uniform(-0.08, 0.08), rng.uniform(-0.04, 0.04))
        self.controller = PIDController()
        self.step_count = 0
        self.prev_front_distance = 99.0
        self.max_progress_x = self.robot.x
        self.no_progress_steps = 0
        self.last_info = {"scene_name": self.scene_name, "override_count": 0}
        obs = self._observation()
        self.prev_front_distance = obs["front_distance"]
        return encode_observation(obs)

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

    def _forward_distance(self) -> float:
        min_distance = self.corridor_length - self.robot.x
        for obstacle in self.obstacles:
            dx = obstacle.x - self.robot.x
            if dx <= 0.0:
                continue
            lateral = abs(obstacle.y - self.robot.y)
            corridor = obstacle.radius + self.robot_radius + 0.15
            if lateral <= corridor:
                min_distance = min(min_distance, max(0.0, dx - corridor))
        return min_distance

    def _local_heading_reference(self) -> float:
        attract_y = -0.9 * self.robot.y
        repel_y = 0.0
        for obstacle in self.obstacles:
            dx = obstacle.x - self.robot.x
            if -0.5 <= dx <= 3.0:
                dy = self.robot.y - obstacle.y
                distance_sq = max(dx * dx + dy * dy, 0.2)
                repel_y += (dy / math.sqrt(distance_sq)) * (1.1 / distance_sq)
        desired_y = attract_y + repel_y
        return math.atan2(desired_y, 2.4)

    def _observation(self) -> Dict[str, float]:
        front_distance = self._forward_distance()
        nearest_obs, nearest_left_obs, nearest_right_obs = self._distance_to_obstacles(self.robot.x, self.robot.y)
        left_boundary = self.corridor_width / 2.0 - (self.robot.y + self.robot_radius)
        right_boundary = self.corridor_width / 2.0 + (self.robot.y - self.robot_radius)
        left_clearance = min(left_boundary, nearest_left_obs)
        right_clearance = min(right_boundary, nearest_right_obs)
        target_heading = self._local_heading_reference()
        heading_error = wrap_angle(target_heading - self.robot.theta)
        return {
            "front_distance": front_distance,
            "left_clearance": left_clearance,
            "right_clearance": right_clearance,
            "center_offset": self.robot.y / max(self.corridor_width / 2.0 - self.robot_radius, 0.1),
            "heading_error": heading_error,
            "speed": self.robot.v,
            "turn_rate": self.robot.w,
            "progress_remaining": self.corridor_length - self.robot.x,
            "nearest_obstacle_distance": nearest_obs,
            "risk_trend": front_distance - self.prev_front_distance,
            "target_heading": target_heading,
        }

    def _map_action(self, raw_action: Tuple[float, float]) -> Tuple[float, float]:
        speed_low, speed_high = ACTION_RANGES["speed_scale"]
        kp_low, kp_high = ACTION_RANGES["kp_scale"]
        speed_scale = speed_low + 0.5 * (float(raw_action[0]) + 1.0) * (speed_high - speed_low)
        kp_scale = kp_low + 0.5 * (float(raw_action[1]) + 1.0) * (kp_high - kp_low)
        return speed_scale, kp_scale

    def _safety_filter(self, raw_speed: float, raw_turn: float, obs: Dict[str, float]) -> Dict[str, float]:
        nearest = obs["nearest_obstacle_distance"]
        front = obs["front_distance"]
        side_min = min(obs["left_clearance"], obs["right_clearance"])
        risk_distance = min(nearest, front, side_min)

        speed_limit = raw_speed
        turn_limit = abs(raw_turn)
        override_strength = 0.0
        emergency_stop = False

        if risk_distance < 1.4:
            ratio = clamp((risk_distance - 0.35) / (1.4 - 0.35), 0.0, 1.0)
            speed_limit = min(speed_limit, 0.20 + 1.0 * ratio)
            turn_limit = min(turn_limit, 0.45 + 1.15 * ratio)
            override_strength = max(override_strength, 1.0 - ratio)

        if front < 0.38 or nearest < 0.30 or side_min < 0.18:
            speed_limit = 0.0
            turn_limit = min(turn_limit, 0.55)
            override_strength = 1.0
            emergency_stop = True

        safe_speed = min(raw_speed, speed_limit)
        safe_turn = clamp(raw_turn, -turn_limit, turn_limit)
        override_flag = abs(safe_speed - raw_speed) > 1e-6 or abs(safe_turn - raw_turn) > 1e-6
        return {
            "safe_speed": safe_speed,
            "safe_turn": safe_turn,
            "override_flag": float(override_flag),
            "override_strength": override_strength,
            "emergency_stop": float(emergency_stop),
        }

    def _collision(self) -> bool:
        if abs(self.robot.y) + self.robot_radius >= self.corridor_width / 2.0:
            return True
        for obstacle in self.obstacles:
            if math.hypot(self.robot.x - obstacle.x, self.robot.y - obstacle.y) <= obstacle.radius + self.robot_radius:
                return True
        return False

    def step(self, raw_action: Tuple[float, float]) -> Tuple[List[float], float, bool, Dict[str, object]]:
        for obstacle in self.obstacles:
            obstacle.step(self.dt)

        obs = self._observation()
        speed_scale, kp_scale = self._map_action(raw_action)
        target_speed = self.base_target_speed * speed_scale
        pid = self.controller.compute(target_speed, obs["target_heading"], self.robot, self.dt, kp_scale)
        safe = self._safety_filter(pid["speed_cmd"], pid["turn_rate_cmd"], obs)

        prev_x = self.robot.x
        self.robot.v = safe["safe_speed"]
        self.robot.w = safe["safe_turn"]
        self.robot.theta = wrap_angle(self.robot.theta + self.robot.w * self.dt)
        self.robot.x += self.robot.v * math.cos(self.robot.theta) * self.dt
        self.robot.y += self.robot.v * math.sin(self.robot.theta) * self.dt
        self.step_count += 1

        if self.robot.x > self.max_progress_x + 0.01:
            self.max_progress_x = self.robot.x
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1

        collision = self._collision()
        success = self.robot.x >= self.corridor_length - self.goal_tolerance
        stuck = self.no_progress_steps >= 140 or self.step_count >= self.max_steps
        done = collision or success or stuck

        next_obs = self._observation()
        progress = self.robot.x - prev_x
        side_min = min(next_obs["left_clearance"], next_obs["right_clearance"])
        nearest = next_obs["nearest_obstacle_distance"]

        reward = 8.0 * progress
        reward -= 0.02
        reward -= 0.05 * abs(next_obs["center_offset"])
        reward -= 0.04 * abs(self.robot.w)
        reward -= 0.05 * max(0.0, 0.95 - nearest)
        reward -= 0.04 * max(0.0, 0.85 - side_min)
        reward -= 0.10 * safe["override_strength"]

        if success:
            reward += 40.0
        if collision:
            reward -= 35.0
        if stuck and not success and not collision:
            reward -= 12.0

        self.prev_front_distance = next_obs["front_distance"]
        self.last_info = {
            "scene_name": self.scene_name,
            "override_count": self.last_info.get("override_count", 0) + int(safe["override_flag"]),
            "success": success,
            "collision": collision,
        }

        if done:
            terminal_info = dict(self.last_info)
            terminal_info.update({"episode_steps": self.step_count, "episode_progress": self.robot.x})
            next_state = self.reset()
            return next_state, reward, True, terminal_info

        return encode_observation(next_obs), reward, False, {"scene_name": self.scene_name}


class VecEnv:
    def __init__(self, num_envs: int, seed: int) -> None:
        self.envs = [TrainCorridorEnv(seed + i * 17) for i in range(num_envs)]

    def reset(self) -> torch.Tensor:
        obs = [env.reset() for env in self.envs]
        return torch.tensor(obs, dtype=torch.float32)

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, object]]]:
        next_obs = []
        rewards = []
        dones = []
        infos: List[Dict[str, object]] = []
        action_np = actions.detach().cpu().tolist()
        for env, action in zip(self.envs, action_np):
            obs, rew, done, info = env.step((action[0], action[1]))
            next_obs.append(obs)
            rewards.append(rew)
            dones.append(float(done))
            infos.append(info)
        return (
            torch.tensor(next_obs, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
            infos,
        )


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: List[int], action_dim: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = obs_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.Tanh())
            in_dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.actor = nn.Linear(in_dim, action_dim)
        self.critic = nn.Linear(in_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.35))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        mean = torch.tanh(self.actor(features))
        value = self.critic(features).squeeze(-1)
        return mean, value

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self(obs)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = torch.clamp(dist.rsample(), -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, value


class ActorOnly(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: List[int], action_dim: int) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone PPO trainer for Safety-Guarded RL-PID modulation")
    parser.add_argument("--total-updates", type=int, default=220)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps-per-rollout", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save-path", type=str, default="models/ppo_policy.pt")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=8)
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_policy(policy: ActorOnly, device: torch.device, episodes: int, seed: int) -> Dict[str, float]:
    env = TrainCorridorEnv(seed + 999)
    success = 0
    collision = 0
    total_progress = 0.0
    total_steps = 0.0
    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = policy(obs_tensor).squeeze(0).cpu()
            obs, _, done, info = env.step((float(action[0]), float(action[1])))
            if done:
                success += int(info.get("success", False))
                collision += int(info.get("collision", False))
                total_progress += float(info.get("episode_progress", 0.0))
                total_steps += float(info.get("episode_steps", 0.0))
    return {
        "success_rate": success / max(episodes, 1),
        "collision_rate": collision / max(episodes, 1),
        "avg_progress": total_progress / max(episodes, 1),
        "avg_steps": total_steps / max(episodes, 1),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    num_steps = args.steps_per_rollout
    num_envs = args.num_envs
    batch_size = num_steps * num_envs
    minibatch_size = min(args.minibatch_size, batch_size)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    vec_env = VecEnv(num_envs, args.seed)
    model = ActorCritic(len(OBS_KEYS), args.hidden_sizes, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    actor_export = ActorOnly(len(OBS_KEYS), args.hidden_sizes, 2).to(device)

    obs = vec_env.reset().to(device)
    global_step = 0
    start_time = time.time()
    best_success = -1.0

    for update in range(1, args.total_updates + 1):
        obs_buf = torch.zeros((num_steps, num_envs, len(OBS_KEYS)), dtype=torch.float32, device=device)
        actions_buf = torch.zeros((num_steps, num_envs, 2), dtype=torch.float32, device=device)
        logprob_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        rewards_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        dones_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        values_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

        for step in range(num_steps):
            global_step += num_envs
            obs_buf[step] = obs
            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(obs)
            actions_buf[step] = action
            logprob_buf[step] = logprob
            values_buf[step] = value

            next_obs, reward, done, _ = vec_env.step(action.cpu())
            obs = next_obs.to(device)
            rewards_buf[step] = reward.to(device)
            dones_buf[step] = done.to(device)

        with torch.no_grad():
            _, next_value = model(obs)
            advantages = torch.zeros_like(rewards_buf, device=device)
            last_gae = torch.zeros(num_envs, dtype=torch.float32, device=device)
            for step in reversed(range(num_steps)):
                next_non_terminal = 1.0 - dones_buf[step]
                next_values = next_value if step == num_steps - 1 else values_buf[step + 1]
                delta = rewards_buf[step] + args.gamma * next_values * next_non_terminal - values_buf[step]
                last_gae = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae
                advantages[step] = last_gae
            returns = advantages + values_buf

        b_obs = obs_buf.reshape((-1, len(OBS_KEYS)))
        b_actions = actions_buf.reshape((-1, 2))
        b_logprob = logprob_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        indices = torch.arange(batch_size, device=device)

        for _ in range(args.epochs):
            shuffled = indices[torch.randperm(batch_size, device=device)]
            for start in range(0, batch_size, minibatch_size):
                mb_idx = shuffled[start:start + minibatch_size]
                _, new_logprob, entropy, new_value = model.get_action_and_value(b_obs[mb_idx], b_actions[mb_idx])
                logratio = new_logprob - b_logprob[mb_idx]
                ratio = logratio.exp()

                pg_loss1 = -b_advantages[mb_idx] * ratio
                pg_loss2 = -b_advantages[mb_idx] * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                value_loss_unclipped = (new_value - b_returns[mb_idx]) ** 2
                value_clipped = b_values[mb_idx] + torch.clamp(new_value - b_values[mb_idx], -args.clip_coef, args.clip_coef)
                value_loss_clipped = (value_clipped - b_returns[mb_idx]) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss + args.vf_coef * value_loss - args.ent_coef * entropy_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

        actor_export.backbone.load_state_dict(model.backbone.state_dict())
        actor_export.actor.load_state_dict(model.actor.state_dict())

        if update % args.save_every == 0 or update == args.total_updates:
            eval_metrics = evaluate_policy(actor_export, device, args.eval_episodes, args.seed + update)
            checkpoint = {
                "format": "sg_rl_pid_ppo_v1",
                "obs_keys": OBS_KEYS,
                "hidden_sizes": args.hidden_sizes,
                "action_ranges": ACTION_RANGES,
                "actor_state_dict": actor_export.state_dict(),
                "metadata": {
                    "algo": "ppo",
                    "global_step": global_step,
                    "update": update,
                    "device": str(device),
                    "success_rate": eval_metrics["success_rate"],
                    "collision_rate": eval_metrics["collision_rate"],
                    "avg_progress": eval_metrics["avg_progress"],
                    "avg_steps": eval_metrics["avg_steps"],
                    "timestamp": int(time.time()),
                },
            }
            torch.save(checkpoint, save_path)
            if eval_metrics["success_rate"] >= best_success:
                best_success = eval_metrics["success_rate"]
                torch.save(checkpoint, save_path.with_name(save_path.stem + "_best.pt"))

            elapsed = time.time() - start_time
            sps = int(global_step / max(elapsed, 1e-6))
            print(
                json.dumps(
                    {
                        "update": update,
                        "global_step": global_step,
                        "sps": sps,
                        "success_rate": round(eval_metrics["success_rate"], 4),
                        "collision_rate": round(eval_metrics["collision_rate"], 4),
                        "avg_progress": round(eval_metrics["avg_progress"], 4),
                        "avg_steps": round(eval_metrics["avg_steps"], 2),
                        "save_path": str(save_path),
                    },
                    ensure_ascii=False,
                )
            )

    print(f"Training complete. Model saved to {save_path}")


if __name__ == "__main__":
    main()

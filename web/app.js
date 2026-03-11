const state = {
  running: false,
  speedMultiplier: 2,
  timer: null,
};

const els = {
  sceneSelect: document.getElementById("sceneSelect"),
  policySelect: document.getElementById("policySelect"),
  seedInput: document.getElementById("seedInput"),
  speedRange: document.getElementById("speedRange"),
  speedLabel: document.getElementById("speedLabel"),
  startBtn: document.getElementById("startBtn"),
  pauseBtn: document.getElementById("pauseBtn"),
  resetBtn: document.getElementById("resetBtn"),
  canvas: document.getElementById("simCanvas"),
  episodeStatus: document.getElementById("episodeStatus"),
  overrideStatus: document.getElementById("overrideStatus"),
  speedScale: document.getElementById("speedScale"),
  kpScale: document.getElementById("kpScale"),
  targetSpeed: document.getElementById("targetSpeed"),
  currentSpeed: document.getElementById("currentSpeed"),
  policySource: document.getElementById("policySource"),
  targetHeading: document.getElementById("targetHeading"),
  headingError: document.getElementById("headingError"),
  rawSpeedCmd: document.getElementById("rawSpeedCmd"),
  rawTurnCmd: document.getElementById("rawTurnCmd"),
  safeSpeedCmd: document.getElementById("safeSpeedCmd"),
  safeTurnCmd: document.getElementById("safeTurnCmd"),
  overrideReason: document.getElementById("overrideReason"),
  frontDistance: document.getElementById("frontDistance"),
  leftClearance: document.getElementById("leftClearance"),
  rightClearance: document.getElementById("rightClearance"),
  centerOffset: document.getElementById("centerOffset"),
  nearestObstacle: document.getElementById("nearestObstacle"),
  boundaryMargin: document.getElementById("boundaryMargin"),
  overrideStrength: document.getElementById("overrideStrength"),
  overrideCount: document.getElementById("overrideCount"),
  timeValue: document.getElementById("timeValue"),
  stepValue: document.getElementById("stepValue"),
  progressValue: document.getElementById("progressValue"),
  modelValue: document.getElementById("modelValue"),
};

const ctx = els.canvas.getContext("2d");

function fmt(value, digits = 2) {
  return Number(value).toFixed(digits);
}

function setSelectOptions(selectEl, values, selectedValue) {
  selectEl.innerHTML = "";
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    option.selected = value === selectedValue;
    selectEl.appendChild(option);
  });
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `HTTP ${response.status}`);
  }
  return response.json();
}

function updateDashboard(snapshot) {
  const meta = snapshot.meta;
  const robot = snapshot.robot;
  const obs = snapshot.observation_summary;
  const rl = snapshot.rl_modulation;
  const pid = snapshot.pid_output;
  const safe = snapshot.safe_action;
  const safety = snapshot.safety_status;
  const episode = snapshot.episode_status;

  els.episodeStatus.textContent = episode.status;
  els.overrideStatus.textContent = safety.override_flag ? safety.override_reason : "none";
  els.speedScale.textContent = fmt(rl.speed_scale, 2);
  els.kpScale.textContent = fmt(rl.kp_scale, 2);
  els.targetSpeed.textContent = `${fmt(rl.target_speed)} m/s`;
  els.currentSpeed.textContent = `${fmt(robot.v)} m/s`;
  els.policySource.textContent = rl.source;
  els.targetHeading.textContent = `${fmt(pid.target_heading)} rad`;
  els.headingError.textContent = `${fmt(pid.heading_error)} rad`;
  els.rawSpeedCmd.textContent = `${fmt(pid.raw_speed_cmd)} m/s`;
  els.rawTurnCmd.textContent = `${fmt(pid.raw_turn_cmd)} rad/s`;
  els.safeSpeedCmd.textContent = `${fmt(safe.speed_cmd)} m/s`;
  els.safeTurnCmd.textContent = `${fmt(safe.turn_cmd)} rad/s`;
  els.overrideReason.textContent = safety.override_reason;
  els.frontDistance.textContent = `${fmt(obs.front_distance)} m`;
  els.leftClearance.textContent = `${fmt(obs.left_clearance)} m`;
  els.rightClearance.textContent = `${fmt(obs.right_clearance)} m`;
  els.centerOffset.textContent = fmt(obs.center_offset, 3);
  els.nearestObstacle.textContent = `${fmt(safety.nearest_obstacle_distance)} m`;
  els.boundaryMargin.textContent = `${fmt(safety.boundary_margin)} m`;
  els.overrideStrength.textContent = fmt(safety.override_strength);
  els.overrideCount.textContent = String(safety.override_count);
  els.timeValue.textContent = `${fmt(meta.time_s)} s`;
  els.stepValue.textContent = String(meta.step_count);
  els.progressValue.textContent = `${fmt(episode.progress_x)} / ${fmt(episode.goal_x)} m`;
  els.modelValue.textContent = `${meta.model_info.name} (${meta.model_info.status})`;

  els.episodeStatus.style.color = episode.status === "running" ? "#182126" : (episode.status === "success" ? "#0f766e" : "#b91c1c");
  els.overrideStatus.style.color = safety.override_flag ? "#b91c1c" : "#182126";
}

function worldToCanvas(snapshot, x, y) {
  const padding = 40;
  const usableWidth = els.canvas.width - padding * 2;
  const usableHeight = els.canvas.height - padding * 2;
  const scaleX = usableWidth / snapshot.corridor.length;
  const scaleY = usableHeight / snapshot.corridor.width;
  const scale = Math.min(scaleX, scaleY);
  const offsetX = padding;
  const offsetY = els.canvas.height / 2;
  return {
    x: offsetX + x * scale,
    y: offsetY - y * scale,
    scale,
  };
}

function draw(snapshot) {
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  const center = worldToCanvas(snapshot, 0, 0);
  const top = worldToCanvas(snapshot, 0, snapshot.corridor.width / 2);
  const bottom = worldToCanvas(snapshot, 0, -snapshot.corridor.width / 2);

  ctx.fillStyle = "#21343a";
  ctx.fillRect(center.x, top.y, snapshot.corridor.length * center.scale, bottom.y - top.y);

  const stripeGap = 80;
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.lineWidth = 2;
  for (let x = center.x; x <= center.x + snapshot.corridor.length * center.scale; x += stripeGap) {
    ctx.beginPath();
    ctx.moveTo(x, top.y + 20);
    ctx.lineTo(x + 32, top.y + 20);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, bottom.y - 20);
    ctx.lineTo(x + 32, bottom.y - 20);
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(236, 253, 245, 0.8)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(center.x, top.y);
  ctx.lineTo(center.x + snapshot.corridor.length * center.scale, top.y);
  ctx.moveTo(center.x, bottom.y);
  ctx.lineTo(center.x + snapshot.corridor.length * center.scale, bottom.y);
  ctx.stroke();

  ctx.setLineDash([8, 8]);
  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(center.x, center.y);
  ctx.lineTo(center.x + snapshot.corridor.length * center.scale, center.y);
  ctx.stroke();
  ctx.setLineDash([]);

  const frontDistance = snapshot.observation_summary.front_distance;
  const safeZoneRadius = Math.max(0.6, Math.min(2.0, frontDistance)) * center.scale;
  const robotCanvas = worldToCanvas(snapshot, snapshot.robot.x, snapshot.robot.y);
  ctx.fillStyle = "rgba(245, 158, 11, 0.12)";
  ctx.beginPath();
  ctx.arc(robotCanvas.x, robotCanvas.y, safeZoneRadius, 0, Math.PI * 2);
  ctx.fill();

  snapshot.obstacles.forEach((obstacle) => {
    const p = worldToCanvas(snapshot, obstacle.x, obstacle.y);
    ctx.fillStyle = obstacle.dynamic ? "#8b5cf6" : "#ef4444";
    ctx.beginPath();
    ctx.arc(p.x, p.y, obstacle.radius * p.scale, 0, Math.PI * 2);
    ctx.fill();
  });

  if (snapshot.robot.path.length > 1) {
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 3;
    ctx.beginPath();
    snapshot.robot.path.forEach(([x, y], idx) => {
      const p = worldToCanvas(snapshot, x, y);
      if (idx === 0) {
        ctx.moveTo(p.x, p.y);
      } else {
        ctx.lineTo(p.x, p.y);
      }
    });
    ctx.stroke();
  }

  ctx.fillStyle = "#14b8a6";
  ctx.beginPath();
  ctx.arc(robotCanvas.x, robotCanvas.y, snapshot.robot.radius * robotCanvas.scale, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = "#ecfeff";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(robotCanvas.x, robotCanvas.y);
  ctx.lineTo(
    robotCanvas.x + Math.cos(snapshot.robot.theta) * 26,
    robotCanvas.y - Math.sin(snapshot.robot.theta) * 26
  );
  ctx.stroke();

  ctx.strokeStyle = "rgba(20, 184, 166, 0.9)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(robotCanvas.x, robotCanvas.y);
  ctx.lineTo(
    robotCanvas.x + Math.cos(snapshot.pid_output.target_heading) * 34,
    robotCanvas.y - Math.sin(snapshot.pid_output.target_heading) * 34
  );
  ctx.stroke();

  const goalX = worldToCanvas(snapshot, snapshot.corridor.goal_x, 0).x;
  ctx.fillStyle = "rgba(16, 185, 129, 0.25)";
  ctx.fillRect(goalX - 8, top.y, 16, bottom.y - top.y);
}

function hydrateControls(snapshot) {
  if (!els.sceneSelect.options.length) {
    setSelectOptions(els.sceneSelect, snapshot.meta.available_scenes, snapshot.meta.scene_name);
  }
  if (!els.policySelect.options.length) {
    setSelectOptions(els.policySelect, snapshot.meta.available_policy_modes, snapshot.meta.policy_mode);
  }
}

async function fetchState() {
  const snapshot = await api("/api/state");
  hydrateControls(snapshot);
  updateDashboard(snapshot);
  draw(snapshot);
}

async function resetSimulation() {
  const snapshot = await api("/api/reset", {
    method: "POST",
    body: JSON.stringify({
      scene: els.sceneSelect.value,
      policy_mode: els.policySelect.value,
      seed: Number(els.seedInput.value || 7),
    }),
  });
  updateDashboard(snapshot);
  draw(snapshot);
}

async function runStep() {
  if (!state.running) return;
  try {
    const snapshot = await api("/api/step", {
      method: "POST",
      body: "{}",
    });
    updateDashboard(snapshot);
    draw(snapshot);
    if (snapshot.episode_status.status !== "running") {
      state.running = false;
      return;
    }
  } catch (error) {
    console.error(error);
    state.running = false;
    return;
  }
  scheduleNextStep();
}

function scheduleNextStep() {
  if (!state.running) return;
  const delay = Math.max(40, 220 / state.speedMultiplier);
  state.timer = window.setTimeout(runStep, delay);
}

function startSimulation() {
  if (state.running) return;
  state.running = true;
  scheduleNextStep();
}

function pauseSimulation() {
  state.running = false;
  if (state.timer) {
    window.clearTimeout(state.timer);
    state.timer = null;
  }
}

els.startBtn.addEventListener("click", startSimulation);
els.pauseBtn.addEventListener("click", pauseSimulation);
els.resetBtn.addEventListener("click", async () => {
  pauseSimulation();
  await resetSimulation();
});

els.policySelect.addEventListener("change", async () => {
  pauseSimulation();
  const snapshot = await api("/api/set_policy_mode", {
    method: "POST",
    body: JSON.stringify({ policy_mode: els.policySelect.value }),
  });
  updateDashboard(snapshot);
  draw(snapshot);
});

els.sceneSelect.addEventListener("change", async () => {
  pauseSimulation();
  await resetSimulation();
});

els.seedInput.addEventListener("change", async () => {
  pauseSimulation();
  await resetSimulation();
});

els.speedRange.addEventListener("input", () => {
  state.speedMultiplier = Number(els.speedRange.value);
  els.speedLabel.textContent = `${state.speedMultiplier}x`;
});

fetchState().catch((error) => {
  console.error(error);
  alert("无法连接本地演示服务，请先运行 launch_demo.bat");
});

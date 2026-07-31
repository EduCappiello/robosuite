# XLeRobot in simulation — branch `xlerobot-17dof`

The 17-DoF XLeRobot (2 × SO-101 arms + 2-DoF head + omni base) as a robosuite
v1.5.2 robot, with per-arm force ground truth and the estimator-validation
harnesses. Built on a copy of `patricia-xlerobot` (all coffee tasks included).

What this branch adds on top of Patricia's:
- `XLeRobot` robot model (17-dim action), verbatim copies of the **calibrated**
  SO-101 chains, per-arm wrist F/T sensors, head with actuators + camera,
  simulated GY-91 IMU (accelerometer + gyro) at the 2nd-tray-layer centre.
- Per-arm ground truth: `get_ground_truth_dynamics(..., arm="right"/"left")`
  with exact cross-arm isolation; per-arm motor signals.
- Foundation fixes (out-of-range init pose, phantom convex-hull self-contacts)
  — the GT identity tests are green again.
- Validation so far: estimator-vs-GT ≤ 0.0005 N·m per joint on both mounted
  arms (0–200 g); E-FC pilot (rotation, not translation, drives false
  contacts). See `robosuite_private/JOURNAL.md` (2026-07-30/31) and the
  `mujoco-sim2real` journal E06–E11.

---

## 0. Prerequisites

```bash
# robosuite side (this repo/branch)
cd ~/Documents/dev/robosuite && git fetch fork
git worktree add ~/Documents/dev/robosuite-xlerobot xlerobot-17dof   # if not present

# lerobot side (estimator scripts): ~/Documents/dev/lerobot-1-coffee @ Coffee_Automata
conda activate lerobot          # py3.12: mujoco 3.9, pinocchio, casadi
```

**⚠ The import gotcha (E08):** the conda env's *editable* robosuite points at
the MAIN worktree (branch `SOARM101`), and `python /abs/path/script.py` puts
the *script's* dir — not your cwd — on `sys.path`. Always pin both trees:

```bash
export PYTHONPATH=~/Documents/dev/lerobot-1-coffee/src:~/Documents/dev/robosuite-xlerobot
```

Every harness prints where `robosuite` / `robosuite_private` / `lerobot`
resolved from — **check those three lines before trusting a run.**

---

## 1. Sanity: the test suites (~10 s)

```bash
cd ~/Documents/dev/robosuite-xlerobot
conda run -n lerobot python -m pytest \
    tests/test_robots/test_xlerobot.py \
    tests/test_robosuite_private/test_xlerobot_gt.py -q
# expect: 12 passed
```

---

## 2. Look at it

Offscreen snapshot (headless-safe):

```bash
cd ~/Documents/dev/robosuite-xlerobot && conda run -n lerobot python -c "
import imageio.v2 as imageio, robosuite as suite
env = suite.make('Lift', robots=['XLeRobot'], has_renderer=False,
                 has_offscreen_renderer=True, use_camera_obs=True,
                 camera_names=['frontview'], camera_heights=768,
                 camera_widths=1024, initialization_noise=None, control_freq=20)
obs = env.reset()
imageio.imwrite('/tmp/xlerobot.png', obs['frontview_image'][::-1])
env.close(); print('wrote /tmp/xlerobot.png')"
```

Interactive MuJoCo viewer (needs a display):

```bash
cd ~/Documents/dev/robosuite-xlerobot && conda run -n lerobot python -c "
import numpy as np, robosuite as suite
env = suite.make('Lift', robots=['XLeRobot'], has_renderer=True,
                 has_offscreen_renderer=False, use_camera_obs=False,
                 ignore_done=True, initialization_noise=None, control_freq=20)
env.reset()
for _ in range(2000):
    env.step(np.zeros(env.action_spec[0].shape))
    env.render()
env.close()"
```

---

## 3. Drive it from Python

Action layout (composite order): **[right arm 5, left arm 5, head 2,
base vel 3, right grip 1, left grip 1]** — don't hand-build it, use
`create_action_vector`:

```python
import numpy as np
from robosuite_private.sim_api import XLeRobotSim
from robosuite_private.motor_signals import read_motor_signals
from robosuite_private.dynamics_gt import get_ground_truth_dynamics

sim = XLeRobotSim({"control_freq": 30})   # Lift scene, deterministic reset
sim.reset()
robot = sim.robot

hold = {a: read_motor_signals(robot, kt=0.5, arm=a)["pos"][:5] for a in ("right", "left")}
action = robot.create_action_vector({
    "right": hold["right"], "left": hold["left"],
    "head": np.zeros(2),
    "base": np.array([0.2, 0.0, 0.0]),        # drive forward 0.2 m/s
    "right_gripper": np.zeros(1), "left_gripper": np.zeros(1),
})
for _ in range(60):
    sim.step(action)

# per-arm force ground truth (6,) each: 5 arm joints + gripper
model, data = sim.env.sim.model._model, sim.env.sim.data._data
gt = get_ground_truth_dynamics(model, data, robot, grouped=True, arms=("right", "left"))
print(gt["right"]["ext"]["tau_ext_contact"])   # exactly 0 in free space

# simulated GY-91 (specific force + angular rate, imu-body frame)
print(robot.get_sensor_measurement("robot0_imu_accel"),
      robot.get_sensor_measurement("robot0_imu_gyro"))
```

Notes: base command is body-frame `(x.vel, y.vel, yaw.vel)` — same interface
as the real host (kiwi mixing stays outside the sim). Default controllers:
arms/head `JOINT_POSITION` (absolute radians), grippers `GRIP`, base
`JOINT_VELOCITY`. For estimator work use the flattened config from
`_controller_configs()` in `xlerobot_estimator_vs_truth.py`.

---

## 4. The validation harnesses (lerobot side)

All in `~/Documents/dev/lerobot-1-coffee/scripts/sim_validation/`; each prints
its import provenance first. Run with the §0 PYTHONPATH pin, any cwd:

```bash
P=~/Documents/dev/lerobot-1-coffee/src:~/Documents/dev/robosuite-xlerobot

# Desk-arm grading (model check, known loads, motion, Coulomb-free + DLS λ
# sweep; LPF + deployed EKF). ~3 min.
PYTHONPATH=$P conda run -n lerobot python \
  ~/Documents/dev/lerobot-1-coffee/scripts/sim_validation/estimator_vs_truth.py
# expect (phase 3): τ_ext RMSE ≤ 0.003 N·m @ 200 g; DLS slope −0.79 @ λ=0.05,
#                   −0.997 @ λ=5e-3; gravity gap ≤ ~0.01 N·m

# Per-mounted-arm grading + mount-TILT sweep. ~1 min.
PYTHONPATH=$P conda run -n lerobot python \
  ~/Documents/dev/lerobot-1-coffee/scripts/sim_validation/xlerobot_estimator_vs_truth.py
# expect: RMSE ≤ 0.0005 N·m every joint, both arms, 0–200 g;
#         tilt: 0.11 N phantom @ 2°, 0.28 N @ 5°

# E-FC false-contact pilot (base maneuvers, arms held, GT contact ≡ 0). ~1 min.
PYTHONPATH=$P conda run -n lerobot python \
  ~/Documents/dev/lerobot-1-coffee/scripts/sim_validation/xlerobot_efc_sim.py
# expect: translation benign at 0.05 N·m threshold; rotation sustains
#         ~0.44 N·m phantom (vibration-dominated — journal E11)

# Per-term error decomposition (debugging tool for any new gap)
PYTHONPATH=$P:~/Documents/dev/lerobot-1-coffee/scripts/sim_validation \
  conda run -n lerobot python \
  ~/Documents/dev/lerobot-1-coffee/scripts/sim_validation/diag_load_residual.py
```

---

## 5. Caveats

- **Mount transforms are CAD estimates** (upstream @ 3d14695e): arms
  (0.1352, ∓0.15, 0.8215) yaw 0; head (−0.125, 0, 0.945); IMU (0, 0, 0.42).
  Verify on the build before trusting platform force data (doc §3.3; the
  wrench channel is the sensitive detector — 0.11 N per 2° of tilt).
- Base = 3 **virtual planar joints**; wheels not simulated; chassis inertials
  are placeholders (Tier-2 item; irrelevant while the base is parked).
- Self-collision excludes `gripper↔shoulder` and `moving_jaw↔shoulder`
  (convex-hull phantoms). `shoulder↔wrist` is a REAL pair and stays
  collidable — grading holds use a raised pose to clear it.
- Grading uses `robot_id="parity"` + static gate OFF for model-fidelity
  phases; deployment-parity (gate auto, friction on) is the phase-1 row.
- Task envs (`cupPnP_task1/2/3/5`) still use the single desk arm —
  re-parenting onto XLeRobot is M4 (planned with Patricia; journal E10).

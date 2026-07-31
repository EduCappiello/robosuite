# XLeRobot in simulation — branch `xlerobot-17dof`

The 17-DoF XLeRobot (2 × SO-101 arms + 2-DoF head + omni base) as a robosuite
v1.5.2 robot, with per-arm force ground truth, teleoperation from two real
SO-101 leader arms + keyboard, and the estimator-validation harnesses.
Built on a copy of `patricia-xlerobot` (all coffee tasks included).

Added on top of Patricia's branch:
- `XLeRobot` robot model (17-dim action), verbatim copies of the **calibrated**
  SO-101 chains, per-arm wrist F/T sensors, head with actuators + camera,
  simulated GY-91 IMU (accelerometer + gyro) at the 2nd tray layer, 3-layer
  cart geometry.
- Per-arm ground truth: `get_ground_truth_dynamics(..., arm="right"/"left")`
  with exact cross-arm isolation; per-arm motor signals.
- `xlerobot_sim` lerobot follower — same 17 action keys as the real robot, so
  the leader-arms + keyboard teleop drives the sim unchanged.
- Foundation fixes (out-of-range init pose, phantom convex-hull self-contacts).
- Results: estimator-vs-GT ≤ 0.0005 N·m per joint on both mounted arms
  (0–200 g); E-FC pilot (rotation, not translation, drives false contacts).
  See `robosuite_private/JOURNAL.md` and journals `mujoco-sim2real` E06–E11.

---

## 0. Prerequisites

```bash
# robosuite side (this repo/branch)
cd ~/Documents/dev/robosuite && git fetch fork
git worktree add ~/Documents/dev/robosuite-xlerobot xlerobot-17dof   # if not present

# lerobot side: ~/Documents/dev/lerobot-1-coffee @ Coffee_Automata
conda activate lerobot          # py3.12: mujoco 3.9, pinocchio, casadi
```

**⚠ Import gotcha (journal E08):** the conda env's *editable* robosuite points
at the MAIN worktree (branch `SOARM101`), and `python /abs/path/script.py` puts
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

## 2. Look at it (the cart / tray)

The cart is a **3-layer RÅSKOG-style stand-in** (primitive geometry, not CAD):

| element | height (top surface) | notes |
|---|---|---|
| layer 1 (bottom basket) | z = 0.058 m | battery / compute |
| **layer 2 (tray deck)** | **z = 0.428 m** | **GY-91 IMU mounts here (green marker); cup-holder tray goes here** |
| layer 3 (arm deck) | z = 0.816 m | both arm bases bolt here |
| footprint | 0.35 × 0.45 m | corner posts + 3 omni wheels (visual only) |

The **cup-holder tray with slots 1–3 is not modelled yet** — it is printed
hardware whose design isn't frozen. Add it as geoms on layer 2 (`tray_layer2_*`
in `robot.xml`) once the holder exists.

Product shot (any angle, no scene needed):

```bash
cd ~/Documents/dev/robosuite-xlerobot/robosuite/models/assets/robots/XLeRobot
conda run -n lerobot python -c "
import mujoco, imageio.v2 as imageio, pathlib
xml = pathlib.Path('robot.xml').read_text().replace(
    '<mujoco model=\"xlerobot\">',
    '<mujoco model=\"xlerobot\"><visual><global offwidth=\"1200\" offheight=\"900\"/></visual>')
pathlib.Path('_r.xml').write_text(xml)
m = mujoco.MjModel.from_xml_path('_r.xml'); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 900, 1200); cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
cam.lookat[:] = [0,0,0.48]; cam.distance = 2.2; cam.azimuth = 135; cam.elevation = -18
r.update_scene(d, cam); imageio.imwrite('/tmp/cart_iso.png', r.render())
import os; os.remove('_r.xml'); print('wrote /tmp/cart_iso.png')"
```

In a scene (the robot is at the table edge; use the viewer to orbit):

```bash
cd ~/Documents/dev/robosuite-xlerobot && conda run -n lerobot python -c "
import numpy as np, robosuite as suite
env = suite.make('Lift', robots=['XLeRobot'], has_renderer=True,
                 has_offscreen_renderer=False, use_camera_obs=False,
                 ignore_done=True, initialization_noise=None, control_freq=20)
env.reset()
for _ in range(3000):
    env.step(np.zeros(env.action_spec[0].shape)); env.render()
env.close()"
```

---

## 3. Platform geometry (for matching the real build)

All values are in the robot base frame **B**: `+x` forward, `+y` left, `z` up,
origin on the floor under the cart centre. Source: upstream Vector-Wangel
XLeRobot CAD @ `3d14695e`, remapped. Registry: `XLEROBOT_FRAME_TREE` in
`robosuite/models/robots/manipulators/xlerobot_robot.py` (guarded by a test).

| frame | x [m] | y [m] | z [m] |
|---|---|---|---|
| **right arm base** | +0.1352 | **−0.150** | 0.8215 |
| **left arm base** | +0.1352 | **+0.150** | 0.8215 |
| head pan axis | −0.125 | 0 | 0.945 |
| head tilt / camera | −0.075 | 0 | 1.125 |
| IMU (2nd tray layer) | 0 | 0 | 0.420 |

**→ The two arm bases are 300 mm apart (0.30 m), purely lateral.** Both sit
+135.2 mm forward of the cart centre and 821.5 mm above the floor, i.e. **5.5 mm
above the arm-deck surface** (mounting-plate thickness), both facing **straight
forward with zero yaw** (no toe-in/toe-out).

Two caveats worth stating plainly:

1. **These are CAD estimates, not measurements of your build.** They are the
   Tier-1 items to verify on hardware (Project_definition §3.3 Validations A/C).
   The tilt sweep says the *wrench* channel is the sensitive detector: about
   **0.11 N of phantom force per 2° of mount tilt**, 0.28 N at 5°.
2. **Leader-arm spacing does not need to match.** SO-101 leader→follower
   teleoperation is a *joint-space* mapping (each leader joint angle is copied
   to the matching follower joint), so the physical distance between your two
   leader arms is purely ergonomic — put them wherever your hands are
   comfortable. What must match the sim is the **follower** mounting: 300 mm
   apart, same height, both facing forward. If you change the real mounting,
   update `XLEROBOT_FRAME_TREE` (and the MJCF) to match — the drift-guard test
   keeps the two in sync.

---

## 4. Teleoperation: two leader arms + keyboard

The sim follower `xlerobot_sim` publishes **exactly the 17 action keys of the
real `XLerobot`**, so `xlerobot_leader_keyboard` drives it with no changes:

```
left_arm_{shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper}.pos
right_arm_{...}.pos          head_motor_1.pos  head_motor_2.pos
x.vel   y.vel   theta.vel
```

Units mirror the real bus: arm/head positions in **degrees**, grippers
**0–100 %**, base `x/y` in **m/s** and `theta` in **deg/s**.

```bash
export PYTHONPATH=~/Documents/dev/lerobot-1-coffee/src:~/Documents/dev/robosuite-xlerobot

lerobot-teleoperate \
  --robot.type=xlerobot_sim \
  --robot.has_renderer=true \
  --teleop.type=xlerobot_leader_keyboard \
  --teleop.left_arm_port=/dev/ttyACM2 \
  --teleop.right_arm_port=/dev/ttyACM3
```

Keyboard (same bindings as the real rig, from the teleop config):

| keys | action |
|---|---|
| `w` / `s` | base forward / backward |
| `a` / `d` | base strafe left / right |
| `q` / `e` | base rotate left / right |
| `n` / `m` | speed level up / down (0.1 / 0.2 / 0.3 m/s, 30/60/90 °/s) |
| `i` / `k` | head tilt up / down |
| `j` / `l` | head pan left / right |

Recording works the same way (`lerobot-record` with the same flags) because the
observation dict carries the identical keys.

**Sim-specific base behaviour, measured not assumed:**
- Command → motion is **1:1**: `x.vel = 0.2` gives 0.200 m/s, `theta.vel = 30`
  gives 30.0 °/s.
- `x.vel` is **body-frame** ("forward" = wherever the robot is facing), matching
  the real robot — robosuite's velocity controller does the body→world rotation
  internally, and the follower rotates the observation back to body frame.
- robosuite's planar base joints ship with `frictionloss=250`, which imposes a
  ~0.25 m/s dead-band. The follower zeroes it by default
  (`zero_base_friction=True`) so the velocity servo tracks the command; set it
  `False` to reproduce E-FC runs, which keep that braking on purpose.

Driving it from Python directly (no hardware needed):

```python
import numpy as np
from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.xlerobot_sim import XLerobotSimConfig, ARM_JOINTS

robot = make_robot_from_config(XLerobotSimConfig(control_freq=30, has_renderer=True))
robot.connect()
obs = robot.get_observation()

action = {f"{a}_arm_{j}.pos": (50.0 if j == "gripper" else obs[f"{a}_arm_{j}.pos"])
          for a in ("left", "right") for j in ARM_JOINTS}
action |= {"head_motor_1.pos": 0.0, "head_motor_2.pos": 0.0,
           "x.vel": 0.2, "y.vel": 0.0, "theta.vel": 15.0}      # drive + turn
for _ in range(120):
    robot.send_action(action)

gt = robot.read_ground_truth()          # sim-only per-arm force truth
print(gt["left"]["ext"]["tau_ext_contact"])   # exactly 0 in free space
robot.disconnect()
```

---

## 5. Direct robosuite access

Composite action order is **[right arm 5, left arm 5, head 2, base 3,
right grip 1, left grip 1]** — build it with `create_action_vector`, never by
hand:

```python
import numpy as np
from robosuite_private.sim_api import XLeRobotSim
from robosuite_private.motor_signals import read_motor_signals

sim = XLeRobotSim({"control_freq": 30}); sim.reset()
robot = sim.robot
hold = {a: read_motor_signals(robot, kt=0.5, arm=a)["pos"][:5] for a in ("right", "left")}
action = robot.create_action_vector({
    "right": hold["right"], "left": hold["left"], "head": np.zeros(2),
    "base": np.array([0.2, 0.0, 0.0]),
    "right_gripper": np.zeros(1), "left_gripper": np.zeros(1)})
for _ in range(60):
    sim.step(action)
print(robot.get_sensor_measurement("robot0_imu_accel"),
      robot.get_sensor_measurement("robot0_imu_gyro"))
```

---

## 6. Validation harnesses (lerobot side)

In `~/Documents/dev/lerobot-1-coffee/scripts/sim_validation/`; each prints its
import provenance first. Run with the §0 PYTHONPATH pin, any cwd:

```bash
P=~/Documents/dev/lerobot-1-coffee/src:~/Documents/dev/robosuite-xlerobot
S=~/Documents/dev/lerobot-1-coffee/scripts/sim_validation

# Desk-arm grading (model check, known loads, motion, Coulomb-free + DLS λ
# sweep; LPF + deployed EKF). ~3 min.
PYTHONPATH=$P conda run -n lerobot python $S/estimator_vs_truth.py
# expect: τ_ext RMSE ≤ 0.003 N·m @ 200 g; DLS slope −0.79 @ λ=0.05, −0.997 @ 5e-3

# Per-mounted-arm grading + mount-TILT sweep. ~1 min.
PYTHONPATH=$P conda run -n lerobot python $S/xlerobot_estimator_vs_truth.py
# expect: RMSE ≤ 0.0005 N·m every joint, both arms, 0–200 g;
#         tilt: 0.11 N phantom @ 2°, 0.28 N @ 5°

# E-FC false-contact pilot (base maneuvers, arms held, GT contact ≡ 0). ~1 min.
PYTHONPATH=$P conda run -n lerobot python $S/xlerobot_efc_sim.py
# expect: translation benign at the 0.05 N·m threshold; rotation sustains
#         ~0.44 N·m phantom (vibration-dominated — journal E11)

# Per-term error decomposition (debugging tool for any new gap)
PYTHONPATH=$P:$S conda run -n lerobot python $S/diag_load_residual.py
```

---

## 7. Caveats

- **Mount transforms are CAD estimates** — verify on the build (§3).
- The base is 3 **virtual planar joints** (robosuite + upstream convention);
  physical wheels are visual only. Chassis inertials are placeholders
  (irrelevant while the base is parked; Tier-2 item).
- Body tree note: `robot0_base` is a *static* outer shell; everything real
  (`manipulator_mount` → arms, head, IMU, cart geometry) hangs off
  `mobilebase0_support` and moves with the base. Probe motion on
  `manipulator_mount`, not `robot0_base`.
- Self-collision excludes `gripper↔shoulder` and `moving_jaw↔shoulder`
  (convex-hull phantoms). `shoulder↔wrist` is a REAL pair and stays collidable
  — grading holds use a raised pose to clear it.
- The cup-holder tray (slots 1–3) is not modelled yet (§2).
- Task envs (`cupPnP_task1/2/3/5`) still use the single desk arm — re-parenting
  them onto XLeRobot is M4 (planned with Patricia; journal E10).

# XLeRobot in simulation — branch `xlerobot-17dof`

> **Start here (Patricia).** Everything you need, and where it lives:
>
> | you want to… | do this |
> |---|---|
> | check the branch is sane | §1 — `pytest …test_xlerobot.py …test_xlerobot_gt.py …test_soarm101.py -q` → 26 pass |
> | see the robot | §2 product shot, or `suite.make('Room128', robots=['XLeRobot'])` |
> | **put the robot in Room 128** | **§10** — `--robot.env=Room128`, or `suite.make('Room128', …)` |
> | teleoperate a task | §8 — one command per task, only `--robot.env` changes |
> | record data that matches the real set | §9 — set `--robot.record_motor_telemetry=true` on **both** sides |
> | know what moved and why | §3 (frames), §2 (cart/holder), §9 (schema) |
>
> **Files.** Robot model `robosuite/models/assets/robots/XLeRobot/robot.xml` (+ CAD
> under `assets/`); frame registry `robosuite/models/robots/manipulators/xlerobot_robot.py`;
> arena `robosuite/models/arenas/room128_arena.py`; envs
> `robosuite/environments/manipulation/{room128,cup_pnp_task1,2,3,5}.py`; sim follower
> `lerobot/robots/xlerobot_sim/` in the lerobot repo.
>
> **Scripts** (all in `robosuite/scripts/`):
> `usd_to_room128_arena.py` regenerates the arena XML from the USD (needs `usd-core`
> in a throwaway venv — it is not a project dependency); `gen_cup_holder.py` regenerates
> the foam cup-holder geoms from the measured dimensions. Both print/write generated
> output — edit the script, not the XML.
>
> **Three traps that cost real time.** (1) Always run from the repo root or with the
> §0 `PYTHONPATH` exported, or robosuite silently resolves to the main worktree and
> *every* env reports "not found". (2) `robot0_base` is a **static** shell — probe
> motion on `mobilebase0_support`, or the robot looks frozen while it is driving.
> (3) `record_motor_telemetry` must match on real and sim or the datasets differ.


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

The cart is the **real IKEA RÅSKOG mesh** from upstream CAD (see below); the
heights the frame tree and collision shell reference are:

| element | height (top surface) | notes |
|---|---|---|
| layer 1 (bottom basket) | z ≈ 0.06 m | battery / compute |
| **layer 2 (tray deck)** | **z = 0.428 m** | **GY-91 IMU mounts here (green marker); cup-holder tray goes here** |
| cart rim (mesh top) | z = 0.775 m | true RÅSKOG height |
| layer 3 (arm deck) | z = 0.816 m | mounting plate; both arm bases bolt here |
| footprint | 0.392 × 0.467 m | from the mesh; casters + 3 omni wheels (visual only) |

**The printed foam cup holder is modelled**, from measurements off the build:
block **29.5 (L, along y) × 18 (W, along x) × 7 cm**, **six holes of diameter
7 cm** in 2 rows × 3 columns, 2 cm between holes, 2.25 cm to the short ends. It
rests on the **top** basket floor (z = 0.688, found by raycasting down inside the
basket — the three floors are 0.688 / 0.408 / 0.138), so its top sits at 0.758,
~1.7 cm under the rim, forward of the arm mounting plate.

Two things fell out of the measurements and are worth recording:

- The gaps (3 cm right / 6.5 cm left) sum to **39 cm**, and the basket interior
  *at that height* measures **39.24 cm**. They agree to 2 mm, which confirms the
  gaps were taken at the foam's own level rather than at the rim (44 cm). The 3 cm
  right gap is the anchor; the left lands at 6.7 cm.
- Across x, the mounting plate leaves **17.96 cm** to the front wall — which is
  why the block is 18 cm wide. It fills that gap exactly.

MuJoCo has no CSG, so the slab is a lattice of boxes around square apertures with
each corner chamfered at 45°: a regular octagon inscribing the 7 cm circle. 35
geoms instead of the ~100 a true ring tessellation needs, round enough to read
correctly and to retain a cup. It is **collidable** — holding cups is the point.

Adding it exposed a bug in the collision shell. The old `tray_layer3_col`, a solid
plate at z = 0.810, **capped the basket**: a cup dropped at a slot settled on an
invisible lid at z = 0.821 and never reached the holder. The real cart has no such
deck — just a rim, the rear mounting plate and the mast — so that plate is now an
open basket (floor 0.688, walls to the 0.775 rim) plus separate `deck_plate_col`
and `mast_col`. Table blocking is unaffected (the corner posts still stop the cart).
Verified: dropped over a slot the cup enters and seats on the basket floor
(z = 0.709, 0.2 mm drift); dropped on a rib it rests on top (z = 0.779).

**The cart and mast are the real upstream CAD.** They come from
Vector-Wangel/XLeRobot's URDF release — `simulation/xlerobot_urdf.zip`,
`meshes/xlerobot/assets/` — vendored into `assets/` next to `robot.xml`:

| mesh | what | tris |
|---|---|---|
| `raskogbody` + `raskogwheel1/2` | the actual IKEA RÅSKOG trolley + casters | 16.7k |
| `topbase1` / `topbase2` | arm mounting plate + head mast, servo | 5.9k |
| `tophead1/4/5/6` | pan yoke, tilt gimbal, camera bar | 12.4k |

Two traps if you edit them. **The vertices are baked in absolute mm at assembly
pose**, so each geom carries upstream's own visual origin — the numbers look
arbitrary but are the CAD assembly; don't "tidy" them. And the `raskog*` meshes
are **Y-up** (`euler="1.5708 0 0"`) with upstream's anisotropic `0.9/1.0/0.9`
scale.

Note the earlier `simulation/mujoco/xlerobot.xml` is a red herring: *it* draws the
cart as one translucent box (`size="0.2 0.2 0.38"`), which is why the cart looked
wrong here for so long. The geometry only exists in the URDF zip.

**Heights.** The RÅSKOG is genuinely 0.78 m, so it is **not** stretched to our
deck: its rim lands at z = 0.775 and the 41 mm up to the guarded arm deck (0.816)
is the mounting plate, which is exactly what `topbase1` models. The whole
mast/head assembly is shifted by that same +0.041 m. Only the arms are unaffected
— they keep the calibrated SO-101 meshes.

The user's `ikea-raskog-pink-utility-trolley.zip` (IKEA's own textured GLB, 24.5k
tris) is kept **untracked** as an alternative: it is pink, needs a GLB→OBJ
conversion step, and covers only the cart — the mast would still come from
upstream.

**Cart collision.** The cart is a hollow *shell*, not a solid block: lower basket
box (z 0.02–0.41) + a plate at each upper deck + the four corner posts. This
matters — the collider used to stop at z = 0.60, so the arm deck at z = 0.816 had
no collider at all and the whole cart drove straight through 0.8 m tabletops.
Basket interiors stay open so objects can rest **on** a deck. Contacts with the
arms are filtered by MuJoCo's parent-child rule, so per-arm
`tau_ext_contact` is still exactly 0 in free space (guarded by the GT tests).

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

| frame | x [m] | y [m] | z [m] | source |
|---|---|---|---|---|
| **right arm base** | −0.0911 | **−0.137** | 0.8215 | mounting-pad circle fit |
| **left arm base** | −0.0911 | **+0.137** | 0.8215 | mounting-pad circle fit |
| head pan axis | −0.103 | 0 | 1.094 | upstream URDF assembly |
| head tilt | −0.102 | +0.002 | 1.192 | upstream URDF assembly |
| head camera | ≈ −0.110 | 0 | ≈ 1.219 | on the `tophead6` bar |
| IMU (2nd tray layer) | 0 | 0 | 0.420 | estimate — measure |

The **head frames changed** when the mast meshes went in: they are now upstream's
own CAD frames (+0.041 m, the mounting-plate shift), replacing an earlier estimate
that put the pan axis at z = 0.945. The head carries no force estimation, so no
dynamics result moves with it — but `head_cam` now sits ~9 cm higher, which
changes the framing of any previously recorded head-camera data.

**→ The two arm bases are 274 mm apart (0.274 m), purely lateral.** Both sit
**91.1 mm BEHIND** the cart centre — they bolt to the two circular pads on the top
plate (`topbase1`), which straddle the head mast at the rear of the cart, not out
over the front tray. Height 821.5 mm puts the SO-101 base mesh flat on the pad
surface (z = 0.819). Both face **straight forward with zero yaw** (no toe-in/out).

These x/y came from fitting circles to the plate mesh's upper surface (centres
(−0.0911, ±0.137), radius 56 mm, sub-mm residual) and are corroborated by
upstream's own URDF arm mount at x = −0.09. They replaced an earlier estimate of
(+0.1352, ±0.150) that floated the arms 23 cm forward of the plate, off the
mounting hardware entirely — visible immediately once the real cart mesh went in.

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

**Ports on the sim rig.** With no follower buses plugged in, the two leader
arms enumerate as the *first* two devices. On this machine:

| arm | port | USB serial (`ID_SERIAL_SHORT`) |
|---|---|---|
| **left leader** | `/dev/ttyACM1` | `5AAF218741` |
| **right leader** | `/dev/ttyACM0` | `5AE6054679` |

(The `ttyACM2`/`ttyACM3` defaults baked into `XLerobotLeaderKeyboardConfig` are
for the **real** rig, where the two follower buses occupy `ACM0`/`ACM1`. Always
pass the ports explicitly — `/dev/ttyACM*` numbering re-enumerates on replug.)

Preflight — both ports free, each holding one 6-motor SO-101:

```bash
fuser -v /dev/ttyACM0 /dev/ttyACM1        # expect: no output
conda run -n lerobot python -c "
from lerobot.motors.feetech import FeetechMotorsBus
for p in ('/dev/ttyACM0','/dev/ttyACM1'): print(p, FeetechMotorsBus.scan_port(p))"
# expect each: {1000000: [1, 2, 3, 4, 5, 6]}
```

```bash
export PYTHONPATH=~/Documents/dev/lerobot-1-coffee/src:~/Documents/dev/robosuite-xlerobot

lerobot-teleoperate \
  --robot.type=xlerobot_sim \
  --robot.has_renderer=true \
  --robot.control_freq=60 \
  --robot.camera_names='[robot0_head_cam]' \
  --teleop.type=xlerobot_leader_keyboard \
  --teleop.left_arm_port=/dev/ttyACM1 \
  --teleop.right_arm_port=/dev/ttyACM0 \
  --teleop.id=xlerobot_leaders
```

**The viewer must not eat your teleop keys.** The native MuJoCo viewer binds a
single-letter shortcut to *all twelve* teleop keys — `w`=wireframe, `s`=shadow,
`a`=auto-connect, `d`=static-body, `q`=camera, `e`=equality, `n`=island,
`m`=centre-of-mass, `i`=inertia, `j`=joint, `k`=skybox, `l`=additive (they come
from `mjVISSTRING` / `mjRNDSTRING`). Those live in the viewer's internal
`mjvScene`/`mjvOption` and **cannot be unbound from Python**. So the follower
defaults to `viewer_backend="opencv"`: robosuite's `OpenCVViewer` draws into a
cv2 window whose `waitKey` swallows keypresses without acting on them, and it
tiles several cameras side by side. Set `--robot.viewer_backend=mjviewer` only
when you want mouse orbit and are *not* driving from the keyboard.

| flag | default | what it does |
|---|---|---|
| `--robot.viewer_backend` | `opencv` | `opencv` (keys safe, fixed cams) or `mjviewer` (orbit, steals keys) |
| `--robot.render_camera` | `[agentview, robot0_head_cam]` | camera tiles in the viewer window |
| `--robot.viewer_fps` | `20` | display refresh cap, decoupled from `control_freq` |
| `--robot.viewer_height/width` | `480` / `640` | per-tile display size |
| `--robot.camera_names` | `[]` | cameras recorded **into the observation** |

The only on-robot camera is `robot0_head_cam`, aimed by the head servos. **There
are deliberately no wrist cameras** — the real build has none, and a sim-only
camera would produce observations no policy could ever get on hardware. (Upstream
ships `XLeRobot_camera1/2` meshes; they were tried here and removed.) Scene
cameras: `frontview`, `birdview`, `agentview`, `sideview`. Any of them can go in
`camera_names` (into the observation dict, so `lerobot-record` captures them) or
`render_camera` (just displayed).

**Placing `head_cam` is fiddly** — the lens must clear the `tophead6` bar, whose
front face is at x = +0.0326 in the tilt frame. Sitting it flush at x = 0.008 put
it *inside* the `tophead5` gimbal (x −0.012…+0.013) and every frame rendered black.
It now sits at x = 0.036. If head frames ever move again, re-check with a raycast
rather than by eye: `mujoco.mj_ray` from `cam_xpos` along `-cam_xmat[:,2]` should
report the wall/table metres away, not a `robot0_head_*` geom at ~0.0001 m.

`viewer_fps` matters: `cv2.imshow` on this box costs ~10–25 ms per draw (Qt
backend), far more than a physics step, so drawing every control step would peg
teleop at single-digit Hz. Throttling the *display* leaves the control loop free.

`--teleop.id` names the calibration profile: the two leaders are stored as
`~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/xlerobot_leaders_arms_{left,right}.json`
(same id the real-rig recording script uses, so one calibration serves both).
If those files are absent the first `connect()` runs the interactive
range-of-motion calibration per arm — move each joint through its full travel,
then press Enter.

**⚠ CLI registration gotcha (recurs on every upstream lerobot merge).** If you
get `--robot.type: invalid choice: 'xlerobot_sim'`, nothing is wrong with the
sim — draccus only offers a `--robot.type` / `--teleop.type` whose config module
has actually been *imported*, and each CLI script carries its own hardcoded
import manifest. `lerobot_record.py` lists `xlerobot` + `xlerobot_leader_keyboard`;
`lerobot_teleoperate.py` originally listed neither, so recording worked while
teleop was unreachable. Fixed in `lerobot/scripts/lerobot_teleoperate.py` by
adding `xlerobot_sim` (guarded, next to `robosuite_sim` — both pull robosuite)
and `xlerobot_leader_keyboard` to the two import blocks. Re-check after any
upstream rebase:

```bash
conda run -n lerobot python -c "
from lerobot.scripts import lerobot_teleoperate
from lerobot.robots import RobotConfig; from lerobot.teleoperators import TeleoperatorConfig
assert 'xlerobot_sim' in RobotConfig.get_known_choices()
assert 'xlerobot_leader_keyboard' in TeleoperatorConfig.get_known_choices()
print('CLI registration OK')"
```

**Percent signs in config comments.** draccus turns field comments into argparse
help, which is `%`-expanded — a bare `%` makes `--help` crash with a
`TypeError`. Write `%%` in config field comments (see the note at the top of
`teleoperators/so_leader/config_so_leader.py`).

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

**Step-rate fix (was ~12 Hz).** robosuite's controller probed `mj_fullM`'s
signature with `try/except TypeError` on *every* controller update. On mujoco 3.9
the first signature always fails, and pybind11 builds the error message by
`repr()`-ing the whole mass matrix — milliseconds per update, ~99% of the step
budget. Physics itself was never the problem (`mj_step` is 0.033 ms; 17 substeps
= 0.56 ms). The signature is now resolved once and cached
(`robosuite/controllers/parts/controller.py`), verified bit-identical:

| | before | after |
|---|---|---|
| `control_freq=30` | 76.5 ms/step (13 Hz) | **7.4 ms/step (136 Hz)** |
| `control_freq=60` | 36.2 ms/step (28 Hz) | **3.9 ms/step (257 Hz)** |

Both rates now run far inside budget (33.3 ms and 16.7 ms). With the on-screen
viewer at its default 20 fps cap, end-to-end teleop measures ~67 Hz at
`control_freq=30` and ~90–106 Hz at 60. Note raising `control_freq` alone never
fixed this: it shortens each step but needs proportionally more of them, so
real-time factor was unchanged — that inverse scaling is what identified the
per-update overhead as the culprit.

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
- `cupPnP_task1` now accepts XLeRobot (see §8). `cupPnP_task2/3/5` still use the
  single desk arm — re-parenting those is the rest of M4 (journal E10).

---

## 8. The cupPnP tasks on XLeRobot

The task was built around the single desk arm bolted to a support table beside
the main table. XLeRobot arrives on its own RÅSKOG cart, so that support table is
**not built at all** — the robot is parked on the floor where it used to stand:

```bash
cd ~/Documents/dev/robosuite-xlerobot
conda run -n lerobot python -c "
import numpy as np, robosuite as suite
env = suite.make('cupPnP_task1', robots=['XLeRobot'], has_renderer=True,
                 has_offscreen_renderer=False, use_camera_obs=False,
                 ignore_done=True, control_freq=30)
env.reset()
for _ in range(3000):
    env.step(np.zeros(env.action_spec[0].shape)); env.render()
env.close()"
```

Two things the env now does automatically when the requested robot is mobile
(`_robots_bring_their_own_base`), and why:

1. **`base_types` switches to the real mobile base.** `SOARM101Lift` hardcodes
   `base_types="NullMount"`, which is right for a desk arm but leaves a mobile
   robot with a `fixed_mount0` that has no `center` site — the env dies at
   `ValueError: Current sensor for observable robot0_base_pos is invalid`.
2. **It parks by ARM BASE, not by chassis — then clamps to the table.** Matching
   chassis-to-chassis puts the arms in the wrong place entirely, so the cart is
   positioned from the arm bases. But the arms bolt to the pads at the *rear* of
   the cart (x = −0.0911), so hitting the desk arm's x = −0.655 exactly would
   drive the chassis 3 cm past the table edge. The standoff is therefore clamped:
   cart centre **x = −0.606** (front edge −0.410 vs table edge −0.400, 1 cm gap),
   putting the arm bases at **x = −0.697**. Verified: **0 robot-scene contacts**
   at reset and while holding.

**Reach is tight, and that is physical, not a bug.** The arms sit behind their own
cart, so they must reach over its top tray before touching the table. Distance from
the left arm base to the cup start is **0.350 m** — right at the SO-101 limit; the
right arm is 0.493 m away, out of reach. Add the 0.865 m tabletop against arm bases
at 0.8215 and the working volume is marginal. Moving the cup/machine toward the near
edge (or giving the task a base-motion phase) is a task-design decision, not a
mounting one, so it is deliberately left alone here.

The desk arm is untouched — `robots=['SOARM101']` still builds its support table
and puts the base at (−0.655, 0.10, 0.77) exactly as before.

### Teleoperating the tasks

The sim follower takes any robosuite env by name, so the leader arms drive a task
scene instead of `Lift` — same 17 keys, nothing else changes. **Only
`--robot.env` differs between tasks:**

| task | `--robot.env` | real dataset | notes |
|---|---|---|---|
| 1 | `cupPnP_task1` | `t1_place_cup` | |
| 2 | `cupPnP_task2` | `t2_push_button` | presses with the **right** hand |
| 3 | `cupPnP_task3` | `t3_cup_to_tray` | |
| 4 | — | `t4_navigate` | **not built** — needs `Room128` (§10) |
| 5 | `cupPnP_task5` | `t5_tray_to_table` | cart spawns rotated 90° |

All four verified through the follower at 120–180 Hz with the two-tile viewer.

```bash
export PYTHONPATH=~/Documents/dev/lerobot-1-coffee/src:~/Documents/dev/robosuite-xlerobot

lerobot-teleoperate \
  --robot.type=xlerobot_sim \
  --robot.env=cupPnP_task1 \
  --robot.has_renderer=true \
  --robot.control_freq=60 \
  --robot.render_camera='[sideview, robot0_head_cam]' \
  --robot.camera_names='[robot0_head_cam]' \
  --teleop.type=xlerobot_leader_keyboard \
  --teleop.left_arm_port=/dev/ttyACM1 \
  --teleop.right_arm_port=/dev/ttyACM0 \
  --teleop.id=xlerobot_leaders
```

Two viewer tiles: **`sideview`** (whole scene — cart, both arms, head, table,
machine, cup, target) and **`robot0_head_cam`** (what the head is actually looking
at). `sideview` is a *fixed world* camera, so if you drive far the cart leaves the
frame — swap it for `birdview` (top-down, keeps everything in view while driving)
if that bothers you. Avoid `agentview`: it looks at the table from the far side and
never shows the robot. Measured **121 Hz** at `control_freq=60`, so there is ample
headroom. Swap `lerobot-teleoperate` for `lerobot-record` with the same flags to
capture; the head camera in `camera_names` lands in the dataset.

Both the cart and the head are live under the standard bindings — `w`/`a`/`s`/`d`
drive, `q`/`e` rotate, `n`/`m` change speed, `i`/`k` tilt and `j`/`l` pan the head.
Verified: 0.2 m/s commanded gives 0.37 m in 2 s, and the head reaches commanded
angles to within ~1 deg.

**Head aiming was broken until now** and is worth knowing about if you see it
resurface. The head actuators were `<position>` servos while every arm actuator is
`<motor>`. robosuite's `JOINT_POSITION` controller writes a *torque* into `ctrl`,
and a position actuator reads that number as an *angle setpoint* — so the head
quietly settled at ~0.72x every commanded angle (a 40 deg command reached 28.8
deg). Both head actuators are now `<motor>`, matching the arms, and the joint
ranges still bound the travel.

---

## 9. Dataset parity with the real robot

The sim's recording schema is matched, channel for channel, against
[`IntelligentDecisionLab/xlerobot-coffee-real`](https://huggingface.co/datasets/IntelligentDecisionLab/xlerobot-coffee-real)
(`room-128/t1_place_cup/meta/info.json`). Verified programmatically:

| | real | sim | status |
|---|---|---|---|
| `action` | 17 | 17 | same names, **same order** |
| `observation.state` (telemetry on) | 84 | 84 | same names, **same order** |
| `observation.images.head` | 480×640×3 | 480×640×3 | ✅ |
| `observation.images.head_depth` | 480×640×1 | 480×640×1 | ✅ uint16 mm |
| fps | 30 | `control_freq=30` | ✅ |

**`record_motor_telemetry` mirrors `XLerobotConfig` exactly, default and all
(`False`).** Off, both sides emit 17 state channels; on, both emit 84. One flag,
same name, same default on real and sim — so the two can never silently diverge.
**Set it on both when recording.**

```bash
--robot.record_motor_telemetry=true --robot.camera_names='[robot0_head_cam]'
```

Three things are worth knowing about how the last channels are filled, because
they are *not* free simulation:

- **Wheel telemetry is derived, not simulated.** The base is three virtual planar
  joints, so `base_*_wheel.vel_hw` is recovered by running the real driver's own
  kiwi mixing (wheels at 240/0/120° minus 90°, r = 0.05 m, base radius 0.125 m) on
  the commanded body velocity. `pos`, `current_raw`, `load_raw` have no sim
  counterpart and are **0**.
- **Head `current_raw`/`load_raw`/`vel_hw` are 0** — the head is not force-estimated.
  Arm telemetry is real, from `read_motor_signals`.
- **Un-simulable IMU channels carry the real dataset's measured means**, not zeros:
  `mpu_temp` 28.19, `pressure` 1011.91, `bmp_temp` 29.75, `altitude` 11.18. Emitting
  zeros would inject a large synthetic distribution shift. `mag_x/y/z` are **0**,
  which is exactly right — the real recordings show the magnetometer is never read.

**IMU sign convention.** The real GY-91 reports `accel_z ≈ −10.39` at rest (board
z points down); MuJoCo's accelerometer reports **+9.81**. `imu_axis_signs`
defaults to `(1, 1, −1)` to match the recorded data. The real data also shows a
small mounting bias (`accel_x ≈ −0.53`, `accel_y ≈ +0.54`) that is **not** modelled.
Verify the sign against the physical board before trusting IMU-driven work.

Depth is validated against ground truth, not eyeballed: a raycast down the camera
axis and the centre depth pixel agree to **0.5 mm** (0.3435 m vs 0.3430 m).

Re-run the check any time with:

```bash
PYTHONPATH=~/Documents/dev/lerobot-1-coffee/src:~/Documents/dev/robosuite-xlerobot \
conda run -n lerobot python -c "
import json,pathlib
from huggingface_hub import hf_hub_download
from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.xlerobot_sim import XLerobotSimConfig
t=(pathlib.Path.home()/'.cache/huggingface/token').read_text().strip()
info=json.loads(pathlib.Path(hf_hub_download('IntelligentDecisionLab/xlerobot-coffee-real',
    'room-128/t1_place_cup/meta/info.json',repo_type='dataset',token=t)).read_text())
real=info['features']['observation.state']['names']
r=make_robot_from_config(XLerobotSimConfig(record_motor_telemetry=True,camera_names=['robot0_head_cam']))
sim=[k for k in r.observation_features if k not in r._cameras_ft]
print('same order:', real==sim, '| cameras:', list(r._cameras_ft))"
```

---

## 10. Room 128 (the real recording room)

`Room128Arena` is the classroom the real dataset was recorded in — every episode
in the HF dataset lives under `room-128/`. Geometry is a **1:1 copy** of
`robosuite/environments/Room_128/classroom128_v2.usd`: that USD is Z-up in metres
(the same conventions as MuJoCo) and built entirely from `Cube` prims, so it needs
no mesh conversion at all. Inner floor **3.665 × 4.935 m**, walls 2.8 m, plus
`door`, `cabinet`, `storage_cabinet`, `desk`, `desk2`, `pillar1`, `pillar2`.

**`Room128` is a registered env**, so placing the robot in the room is one line —
no manual arena/task composition:

```python
import robosuite as suite
env = suite.make("Room128", robots=["XLeRobot"], has_renderer=True,
                 has_offscreen_renderer=False, use_camera_obs=False,
                 ignore_done=True, control_freq=30)
env.reset()
```

Park it wherever with `robot_start_pos` / `robot_start_yaw` (room coordinates:
x ±1.83, y ±2.47, floor z=0):

```python
env = suite.make("Room128", robots=["XLeRobot"],
                 robot_start_pos=(-0.8, 1.2, 0.0), robot_start_yaw=1.57, ...)
```

And drive it with the leader arms + keyboard, exactly like a task:

```bash
lerobot-teleoperate \
  --robot.type=xlerobot_sim --robot.env=Room128 \
  --robot.has_renderer=true --robot.control_freq=60 \
  --robot.render_camera='[birdview, robot0_head_cam]' \
  --teleop.type=xlerobot_leader_keyboard \
  --teleop.left_arm_port=/dev/ttyACM1 --teleop.right_arm_port=/dev/ttyACM0 \
  --teleop.id=xlerobot_leaders
```

`birdview` rather than `sideview` here: the cart drives around, and a fixed
3/4 camera loses it quickly. Reward is always 0 — this env exists to place and
move the robot, not to score it.

Verified: driving `x.vel=0.3` for 2 s moves the base 0.552 m, and the cart is
stopped by the furniture (`room128_cabinet`) at x=1.203. **Measure motion on
`mobilebase0_support`, not `robot0_base`** — the latter is a static shell and will
make a moving robot look frozen.

The lower-level pieces, if you need them:

```python
from robosuite.models.arenas import Room128Arena
```

Two deliberate differences from robosuite's stock arenas: the room is **re-centred
on the origin** (the USD keeps its origin in a corner, while the tasks build their
furniture around 0,0), and the walls and furniture **collide** — stock arena walls
are visual-only, but this room exists so the cart can be driven around obstacles.

The XML is generated, not hand-written. Regenerate after any USD change with
`robosuite/scripts/usd_to_room128_arena.py` (needs `usd-core`, which is *not* a
project dependency — use a throwaway venv).

**Task 4 is why this room exists.** Mapping the real dataset onto the sim tasks:

| real dataset | sim env |
|---|---|
| `t1_place_cup` | `cupPnP_task1` |
| `t2_push_button` | `cupPnP_task2` |
| `t3_cup_to_tray` | `cupPnP_task3` |
| **`t4_navigate`** | **not built yet — needs this arena** |
| `t5_tray_to_table` | `cupPnP_task5` |

Patricia's set has no task 4 because it is a *navigation* task: it cannot be
expressed on a tabletop arena. The `Room128` env above is the missing piece — the
room, the robot, collidable furniture and a driveable base. What `t4_navigate`
still needs on top is a goal pose, a success condition and a reward; the scene
itself is done.

---

## 11. Current manipulation-task integration

The current `xlerobot-17dof` manipulation setup includes the following local
integration work:

- `LockedNullMobileBase` preserves the three base action/state channels while
  constraining the chassis to its reset pose. Contact from either arm therefore
  cannot move the cart during Tasks 1, 2, 3, or 5.
- The foam cup holder and blind holder floor are raised by 17 mm so the holder
  top is flush with the cart tray rim. Task 3 and Task 5 use the same holder
  center and support-plane constants from `soarm101_lift.py`.
- The block-built coffee machine contains a physical 10 cm white cup pad. Task
  1 grades the cup center against this round pad and requires the cup to be
  upright, released, and at least 8 cm clear of the active gripper.
- Tasks 1, 3, and 5 apply small reset-time cup position, yaw, and visual-color
  randomization. Deterministic reset mode disables this randomization.
- Task 2 attaches its stylus to the XLeRobot right fixed gripper and retains the
  seven-button contact classification used by the desk-arm version.
- The right fixed gripper carries `assets/version0.stl` plus an optional
  `right_gripper_cam`. Tune the bracket or child camera independently with
  `tools/tune_xlerobot_gripper_camera_mount.py`.
- Viewer-only alpha/RGB overrides are implemented in `mjviewer_renderer.py`.
  They operate on a copied render model, so recorded camera observations keep
  the original opaque materials. The OpenCV viewer now opens at the rendered
  image resolution and remains resizable.

The task implementations share gripper-tip and jaw-contact helpers so success
checks work for both the single SOARM101 and the dual-arm XLeRobot. XLeRobot cup
tasks use the left arm as the manipulation arm; Task 2's stylus uses the right
arm.

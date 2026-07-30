# SOARM101 Sim-to-Real — Experiment Journal

> Living document. Newest dated entries at the bottom of **§8 Running Log**.
> Everything above §8 is the current, maintained state of the system — update it
> in place when the architecture changes; append a dated note in §8 explaining why.

---

## 1. Objective

Build a robosuite simulation of the **SO-ARM101** (5-DOF arm + 1-DOF gripper) that
emits **ground-truth dynamics** — per-joint torque decomposition and wrist
force/torque — at every physics step, drive it by **teleoperation** from the
physical leader arm, and use those signals to **close the sim-to-real gap**.

The thesis: a policy that has access to (or is trained against) the *dynamic
model* and *force-feedback* signals — not just kinematics — transfers to the real
SO-ARM101 with higher success rate. The simulator is the ground-truth oracle that
the real-robot estimators in **lerobot-private** are validated against.

**Roles**
- **robosuite (this repo)** — the physics oracle. Produces ground-truth `τ` and
  wrench. Knows nothing about LeRobot (enforced; see §3).
- **lerobot-private** — owns the real robot, the teleop loop, calibration files,
  the force-feedback safety stack, and the estimation algorithms whose output is
  compared against robosuite's ground truth.

---

## 2. System Architecture

```
                       same conda env  (robosuite, py3.10)
┌───────────────────────────┐   import   ┌──────────────────────────────────┐
│   robosuite_private/       │ ◀───────── │   lerobot-private                 │
│   (framework-agnostic)     │            │   (teleop, estimators, FF stack)  │
│                            │            │                                   │
│  SOARM101Sim  (sim_api)    │            │   • drives sim with joint targets │
│   ├ read_motor_signals     │ ─ signals→ │   • runs estimators on signals    │
│   ├ get_ground_truth_dyn.  │ ─ truth →  │   • compares estimate vs truth    │
│   ├ apply_noise            │            │   • renders force feedback on the │
│   ├ get_camera_frames      │            │     physical leader arm           │
│   └ set_friction_params    │            │                                   │
└───────────────────────────┘            └──────────────────────────────────┘
        │ wraps
        ▼
   robosuite SOARM101Lift  (MuJoCo, 20 Hz control / N substeps)
```

**Key design rule:** the boundary is a *data contract*, not a framework
dependency. `robosuite_private` must never import `lerobot*`
(test_no_lerobot_imports.py enforces this). lerobot-private imports
`robosuite_private`, never the reverse. This keeps the physics oracle
reusable and independently testable.

**Substep visibility.** robosuite sets actuator targets once per *control* step
(20 Hz) but MuJoCo integrates at the *model* timestep (N substeps per control
step). `SOARM101Sim.substep_hook()` + `_step_with_hooks()` let estimators sample
ground truth at the full physics rate, which matters for force/contact transients
that are invisible at 20 Hz.

---

## 3. The Ground-Truth Data Contract

This is the scientific core of the project. Two dicts, both callable per substep.

### 3.1 `read_motor_signals(robot, kt) -> dict`  (motor_signals/reader.py)

What a real STS3215 servo bus could plausibly report — the *input* to estimators.

| key              | shape | units    | source / formula |
|------------------|-------|----------|------------------|
| `pos`            | (6,)  | rad      | `qpos[arm(5)+grip(1)]` |
| `vel_hw`         | (6,)  | rad/s    | `qvel[…]` |
| `current`        | (6,)  | A        | `qfrc_actuator / kt` |
| `current_signed` | (6,)  | A        | `current · sign(vel)` |
| `load`           | (6,)  | unitless | `qfrc_actuator / SERVO_FORCE_RANGE[1]` (∈ roughly [-1,1]) |
| `dt`             | scalar| s        | `model.opt.timestep` |

`kt` (motor torque constant) defaults to `KT_DEFAULT = 0.5 N·m/A` but should be
**calibrated per robot_id** in production — pass the calibrated value.

### 3.2 `get_ground_truth_dynamics(model, data, robot, *, grouped=False, load_body=None) -> dict`  (dynamics_gt/ground_truth.py)

The torque decomposition estimators are graded against. All `(6,) float64` in DOF
order (5 arm joints then gripper, via `robot._ref_*` indices — **R8**). As of
2026-06-22 the output is split into two named groups; `grouped=True` →
`{"model": {...}, "ext": {...}}`. The flat default view is kept for backward
compatibility (see below).

**`ground_truth.model.*`** — rigid-body decomposition (matches lerobot's
`TorqueEstimate` fields 1:1):

| key            | meaning | how it is computed |
|----------------|---------|--------------------|
| `tau_motor`    | commanded actuator torque | `qfrc_actuator[dof]` |
| `tau_gravity`  | gravity torque G(q)       | **zero-velocity trick**: scratch `MjData`, copy `qpos`, `qvel=0`, `mj_forward`, read `qfrc_bias` |
| `tau_coriolis` | Coriolis + centrifugal    | `qfrc_bias − tau_gravity` |
| `tau_inertial` | M(q)·q̈                    | `mj_fullM` → `M @ qacc` |
| `tau_friction` | Coulomb + viscous (APPLIED/negative sign) | **reconstructed**: `−(sign(v)·frictionloss + damping·v)` |
| `tau_model`    | model-sum (NEW)           | `tau_gravity + tau_coriolis + tau_inertial + tau_friction` |

**`ground_truth.ext.*`** — external-force truth (what the estimators try to recover):

| key                | meaning | how it is computed |
|--------------------|---------|--------------------|
| `tau_ext`          | external joint torque (**PRIMARY**) | `−Jᵀ·w` when a load/contact wrench is known on `load_body` (or an auto-detected nonzero `xfrc_applied` body), else falls back to the residual |
| `tau_ext_residual` | analytic EoM residual — what the *estimator* can compute from motor signals | `tau_motor − tau_gravity − tau_coriolis − tau_inertial + tau_friction` (**R1**: trailing **+**, friction stored negative — never write `tau_motor − tau_model`) |
| `tau_ext_jac`      | `−Jᵀ·w` when a wrench is known, else zeros | COM Jacobian (`mj_jacBody`); for an eef-body load `tau_ext_jac[5] == 0` (gripper is a sibling DOF — **R4**) |
| `tau_ext_contact`  | external joint torque from the sim's ACTUAL contacts (added 2026-06-22) | `mj_contactForce` over `data.contact`, robot↔WORLD pairs only (self-collision skipped), each wrench (force+torsional torque) projected via the point Jacobian `mj_jac` at `c.pos`. **Exactly 0 in free space**, immune to friction/saturation/limits/self-collision, real reaction on table/cube/grasp. Verified `== −(contact-only efc force)` to 0.0 on all DOFs. The clean FF signal (`--ff_source=contact`); also exposes `read_external_contacts()` → contacting geom pairs |
| `tcp_wrench_ext`   | purely-external wrench at the TCP, **world frame** `[fx,fy,fz,tx,ty,tz]` | transported from `p_app` (`xipos[body]`) to `p_tcp` (`site_xpos[eef]`): `f_tcp=f`, `t_tcp = t + cross(p_app − p_tcp, f)` (**R2**) |
| `tcp_wrench_ft`    | raw wrist FT sensor, gripper-body frame, **total transmitted** wrench (distal gravity+inertia+external) | `robot0_wrist_ft_force`/`_torque` — diagnostic only, never into `tau_ext` (**R5**) |

**Flat default view** = the model + ext keys, plus legacy `tcp_wrench` =
`tcp_wrench_ft` (preserves the pre-2026-06-22 meaning).

**Anchor identity (proven, T8):** with `frictionloss=0`, after one `mj_forward` at a
contact-free pose, `tau_ext_residual == tau_ext_jac == −Jᵀ_com·w` exactly (viscous
damping cancels `qfrc_passive`). This is the test that grounds the whole decomposition.

**Methodological decisions (do not "simplify" these):**
- **Friction is reconstructed from `dof_frictionloss`/`dof_damping`, NOT read from
  `qfrc_constraint`.** `qfrc_constraint` mixes contact forces with friction;
  reconstructing with the *same functional form the estimator uses* means
  residuals reflect algorithm quality, not a parameter mismatch.
- **Gravity uses a scratch `MjData`** because MuJoCo has no standalone G(q) call,
  and `qfrc_bias` at the live state already contains Coriolis terms.
- **`tau_ext`/`tau_ext_jac` use the COM Jacobian** (`mj_jacBody`) to match where
  `xfrc_applied` acts (**R7**); a body-origin Jacobian would mis-transport the moment.
- **The FT sensor (`tcp_wrench_ft`) is the *total transmitted* wrench**, kept for
  diagnostics, never summed into `tau_ext` (**R5**). `cfrc_ext` layout is
  `[torque, force]` about COM — reversed vs `xfrc_applied`'s `[force, torque]` (**R3**).
- Friction params are live-mutable via `set_friction_params(model, …)`; the next
  `get_ground_truth_dynamics` call reflects the change with no re-instantiation,
  enabling per-robot friction identification sweeps.
- Load helpers `apply_external_load(model, data, body_name, wrench)` /
  `clear_external_loads(data)` mutate `xfrc_applied` (world, at COM) for
  tests/harness — they are NOT part of the read path.

### 3.3 `apply_noise(signals, profile, rng) -> dict`  (noise/)

Pure function (no global state). Degrade clean sim signals toward sensor realism
before feeding estimators. Models: `GaussianNoise`, `QuantizationNoise`
(encoder bins), `DropoutNoise` (bus packet loss), `BiasNoise` (calibration
offset). Per-signal-key config via `NoiseProfile`; pass a seeded `rng` for
reproducibility. Unmatched keys pass through unmodified.

### 3.4 Canonical spec  (robot_spec/canonical.py)

Single source of truth, derived from the MJCF. STS3215: `kp=998.22`,
`kv=2.731`, `forcerange=±2.94 N·m`, `damping=0.60`, `frictionloss=0.052`,
`armature=0.028`. `validate_against_mjcf(model)` asserts the loaded model still
matches these numbers; `export_robot_spec()` returns the spec as a plain dict
(**version 2**) for lerobot-private CI to cross-check against its Pinocchio URDF.
v2 also carries the **STS3215 register constants**: `current_raw→mA = 6.5`,
`torque_const = 31.8 mN·m/A`, `gear_ratio = 1/345`, `stall_current = 3.0 A`.

### 3.5 `to_servo_raw(signals, *, spec=canonical) -> dict`  (motor_signals/servo_raw.py)

Parity-tier exporter: converts the SI `read_motor_signals` dict into the raw
register arrays a real STS3215 bus reports, so estimators consume an identical
schema in sim and on hardware. Pure/numeric, no lerobot import.

| key           | units | formula |
|---------------|-------|---------|
| `pos_deg`     | deg   | `pos · 180/π` |
| `vel_deg_s`   | deg/s | `vel_hw · 180/π` |
| `current_raw` | tick  | `current_A · 1000 / 6.5` (**unsigned**) |
| `load_raw`    | tick  | sign `−sign(load)` (fallback `sign(vel)`), magnitude `|load|·1000` (**signed**) |

**R6**: the servo-raw path uses the STS torque↔current constants
(`current_A = τ_out · 345 / 0.0318`), NOT the SI reader's generic `kt=0.5` —
mixing them is a ~10⁴ scale error. Round-trip identity (`to_servo_raw` → STS
forward) recovers `load · forcerange[1]`; guarded by `test_servo_raw.py`.

---

## 4. Repository Changes Log

Grouped by area. "tracked" = modifies upstream robosuite (shows in `git diff`);
"new" = added file (untracked / private).

### 4.1 New environment — `SOARM101Lift`  (new: environments/manipulation/soarm101_lift.py)
- Subclass of `Lift`. `base_types="NullMount"` (arm is fixed-base, no mobile base).
- Cube: `cube_mass=0.03 kg`, `cube_size=(0.018)³ m`, density derived so mass is
  exact; `cube_offset=(-0.2,-0.2,0)` to place it in the arm's reachable zone.
- Three cameras (tune `pos`/`quat` here for recording — see §6):
  - `operator_view` — front-facing, mirrors the human operator's viewpoint.
  - `top` — overhead bird's-eye (raise `pos[2]` to zoom out).
  - `corner` — right-front table corner, ~4 cm above the surface.

### 4.2 Robot model  (tracked: models/robots/manipulators/soarm101_robot.py)
- XML swapped to **`soarm_with_sensor.xml`** (instrumented variant with wrist FT).
- `default_controller_config`: `osc_pose → osc_position`.
- `init_qpos`: now **5 values** `[0, 0.65, 0, 1.5, 0]` (arm only — gripper DOF
  moved into the gripper model).

### 4.3 Gripper — 5+1 DOF split  (tracked: models/assets/grippers/so101_gripper.xml)
- The `moving_jaw` body + `gripper` hinge joint + collision/visual geoms + the
  position actuator now live in the **gripper** XML (previously a null gripper).
- Actuator is a **position** actuator (`kp=998.22`, `forcerange=±2.94`), giving
  proportional jaw control (GRIP action [-1,1] → joint range linearly), not
  bang-bang torque.
- Result: arm = 5 DOF, gripper = 1 DOF, `action_dim = 6`. (Earlier the jaw lived
  in the arm XML and pushed `action_dim` to 7 — bug, now fixed by this split.)

### 4.4 Instrumented arm XML  (new: …/SOARM101/SO101/soarm_with_sensor.xml)
- Copy of `so101_new_calib.xml` with a wrist FT **site** at the gripper body
  origin and `force`/`torque` sensors → `robot0_wrist_ft_force` /
  `robot0_wrist_ft_torque` (read via `robot.get_sensor_measurement`).
- `wrist_roll_follower` mesh collision removed (caused premature contact ~16 mm
  short of the grip site); replaced by a small `static_finger_tip` box at the
  true grip zone. Jaw body/joint/actuator removed (now owned by the gripper XML).

### 4.5 Upstream robosuite patches (tracked)
- `__init__.py` — register `SOARM101Lift`.
- `lift.py` — forward `base_types` param (was hard-coded `"default"`), so
  `SOARM101Lift` can request `NullMount`.
- `devices/device.py` — accept **`OSC_POSITION`** controllers (3-DOF arm delta),
  not just `OSC_POSE`/`JOINT_POSITION`; size `norm_delta` accordingly.
- `demos/demo_device_control.py` — minor support for the above.
- `setup.py` — include the `robosuite_private` package in `find_packages`.
- `so101_new_calib.xml` — same jaw/follower fixes as the sensor XML.

### 4.6 The `robosuite_private/` package (new)
| module | exports | purpose |
|--------|---------|---------|
| `sim_api.py` | `SOARM101Sim` | facade: reset/step, substep hooks, camera frames |
| `motor_signals/reader.py` | `read_motor_signals` | servo-bus signal schema (§3.1) |
| `motor_signals/servo_raw.py` | `to_servo_raw` | SI → STS raw register arrays (§3.5) |
| `dynamics_gt/ground_truth.py` | `get_ground_truth_dynamics` | torque decomposition composer (§3.2) |
| `dynamics_gt/model_terms.py` | `compute_model_terms` | `ground_truth.model.*` rigid-body decomposition |
| `dynamics_gt/external.py` | `compute_external_terms`, `apply_external_load`, `clear_external_loads` | `ground_truth.ext.*` + load helpers |
| `dynamics_gt/jacobian.py` | `body_jacobian` | 6×6 COM spatial Jacobian (`mj_jacBody`) |
| `dynamics_gt/friction.py` | `set_friction_params` | live-mutate friction coeffs |
| `noise/` | `apply_noise`, `NoiseProfile`, … | sensor-realism noise (§3.3) |
| `rendering/cameras.py` | `get_camera_frames` | offscreen RGB frames |
| `robot_spec/` | `export_robot_spec`, `validate_against_mjcf`, canonical constants | single source of truth (§3.4) |

### 4.7 Tests
- `tests/test_robosuite_private/` — **39 tests, all passing** (was 32): schema
  shape/dtype checks for both groups, spec round-trip, substep-hook firing + `dt`,
  the no-LeRobot-import guard, plus `test_external_gt.py` (the T8 anchor identity,
  R1/R2/R4/R5 guards), `test_servo_raw.py` (R6 current↔torque round-trip), and
  `test_soarm101_upstream_inertials.py` (the upstream drift guard — mass/COM/inertia
  vs `UPSTREAM.json`, massless `gripper_base`, total mass, mesh md5; mutation-tested).
- `robosuite_private/tools/gen_upstream_manifest.py` — regenerates `UPSTREAM.json` from a
  SO-ARM100 clone (the only path that should ever change those frozen numbers).
- `tests/test_robots/test_soarm101.py` (tracked) — extended for the 5+1 DOF model.
- (lerobot side) `tests/robots/test_robosuite_sim.py` — **8 passed**, skips where
  robosuite is absent.

---

## 5. Architecture Evolution — superseded prototypes

The first working approach (documented in `~/.claude/plans/now-help-me-plan-squishy-honey.md`)
was a set of **standalone demo scripts** plus a **subprocess bridge** to cross the
Python 3.10 (robosuite) ↔ 3.12 (lerobot) version gap:

- `demos/demo_soarm101_teleop.py`, `demos/demo_soarm101_record.py`
- `devices/so_leader_device.py` + `devices/so_leader_server.py` (subprocess in the
  lerobot py3.12 env, streaming JSON over stdout)
- `wrappers/lerobot_dataset_wrapper.py` (custom LeRobot v3.0 parquet/MP4 writer)

**These files are no longer in the working tree.** They were superseded by the
`robosuite_private/` package + **direct same-conda-env import** (sim_api.py:10:
"lerobot-private imports this module directly (same conda environment)"), which
removes the subprocess bridge and the dual-Python split entirely. Stale `.pyc`
caches may remain under `__pycache__/`.

> If a self-contained teleop/record demo is still wanted, regenerate it on top of
> the `SOARM101Sim` facade rather than resurrecting the subprocess scripts. Ask
> and it can be rebuilt against the current API.

---

## 6. Camera Tuning (recording)

All camera placement lives in **one place**:
`environments/manipulation/soarm101_lift.py` → the `set_camera()` calls.
`pos=[x,y,z]` metres, `quat=[w,x,y,z]` MuJoCo convention. Recompute a `corner`
quaternion after moving it with the cross-product helper noted in that file.
Frames are pulled via `SOARM101Sim.get_camera_frames(["top","corner"], 640, 480)`
to match the real rig's 640×480 resolution.

---

## 7. LeRobot Bridge & Validation Harness  (lives in lerobot-private)

The contract in §3 is consumed by a drop-in LeRobot `Robot` and validated by a
ground-truth harness. Both live in the lerobot repo (allowed import direction:
lerobot → robosuite_private, never the reverse).

### 7.1 `robosuite_sim` drop-in Robot  (lerobot/robots/robosuite_sim/)
`@RobotConfig.register_subclass("robosuite_sim")` → `RobosuiteSimFollower(Robot)`,
schema-identical to `so_follower` so the EKF / TCP-wrench / friction estimators
consume it unchanged. Real↔sim becomes a single `--robot.type` flag.
- **parity tier** (always): `{motor}.pos`, `.current_raw`, `.load_raw`, `.vel_hw`
  (from `read_motor_signals` → `to_servo_raw`; optional `apply_noise`).
- **privileged tier** (`emit_ground_truth=True`): flattened `ground_truth.model.{m}.*`,
  `ground_truth.ext.{m}.*`, `ground_truth.ext.tcp_wrench_ext/_ft` — labels, never
  required at real deploy time.
- `send_action`: 5 arm `{m}.pos` (deg→rad, absolute JOINT_POSITION goals) + GRIP.
- controller config is PRE-FLATTENED (`body_parts` keyed by `"right"`) because a
  dict passed via `controller_configs` bypasses robosuite's arms→per-arm file-loader.
- **Config knobs added for teleop (2026-06-22):** `has_renderer` (calls `SOARM101Sim.render()`
  each `send_action` → on-screen MuJoCo window); `kp`/`kv` (JOINT_POSITION stiffness — default 100
  SAGS under gravity, use 300–400 for teleop); `max_relative_target` (per-step clamp);
  `zero_coulomb_friction` (debug). GRIP uses `use_action_scaling=True` (was False → dead-zone +
  never-closing jaw). New methods: `read_ground_truth(grouped, load_body)`, `read_external_contacts()`.

### 7.2 `scripts/sim_validation/estimator_vs_truth.py`
Drives the follower on the parity tier, runs the estimators, compares to
`ground_truth.ext.*`. Phases: (0) model-agreement spec↔URDF; (1) known-load static
hold 0/1/50/100/200 g; (2) motion sweep (friction observable → L0/L2 ablation);
(3) Coulomb-free known-load (isolates frame/sign + the model gap).

**Findings (2026-06-22 first run):**
1. **τ_ext recovery is correct.** Under known load the LPF estimator tracks the
   physical `−Jᵀw`; the residual bias is load-**independent** (shoulder_lift
   ≈ −0.49 N·m flat across 0→200 g), proving the recovery math/sign is right.
2. **✅ RESOLVED — model-fidelity gap was a robosuite phantom mass.** That
   load-independent bias *was* `G_mujoco − G_urdf`: shoulder_lift −0.49, elbow_flex
   −0.33 N·m (others ≈ 0). `gravity_model_compare.py --dump-inertials` localized it to
   **one body**: robosuite's `gripper_base`. Every kinematic-chain link (shoulder→jaw) is
   **byte-identical** between MJCF and URDF (same CAD source). `gripper_base` is a
   robosuite-ONLY mounting body the arm/gripper split inserted; it has **no counterpart
   in upstream** (TheRobotStudio/SO-ARM100 MJCF *or* URDF, nor SO100) and was given a
   fabricated `mass=3e-1` placeholder ([so101_gripper.xml:10]). The real gripper mass
   (6th servo + jaw) is already fully carried by `gripper`(0.087) + `moving_jaw`(0.012).
   **Conclusion inverted vs the first read:** the URDF/Pinocchio model the *real-robot
   estimator uses is correct*; only this sim copy was wrong (it injected a phantom
   +0.3 kg, +47.5%, at the wrist). **Fix = make `gripper_base` massless** (`1e-9`, done
   2026-06-22). Verified: 226-pose `G_mujoco−G_urdf` RMSE → **0.0000 N·m** on every joint;
   total moving mass 0.932→**0.632 kg**; `estimator_vs_truth` phase-3 0 g gravity gap ~0.
   Locked in by `UPSTREAM.json` + the drift-guard test (§4.7). See §8 resolution entry.
3. **TCP-wrench sign convention.** The estimator's `_dls_wrench` solves
   `tau_ext = +Jᵀw`, so its published wrench is the REACTION (`−F_ext`); the sim's
   `tcp_wrench_ext` is the APPLIED force. The Fz slope d(est)/d(imposed) ≈ −1
   confirms it. Either document the reaction convention or negate in `_dls_wrench`.
   The absolute wrench magnitude is further degraded by finding 2 + the DLS λ, so
   trust the joint-space τ_ext comparison over the Cartesian wrench until 2 is fixed.

### 7.3 Force-feedback teleoperation  (lerobot `lerobot_teleoperate.py` + `so_leader`)
Physical SO-101 **leader** teleoperates the **`robosuite_sim` follower**; the follower's external
joint torque is rendered back on the leader's servos so the operator feels contact. Drop-in via
`lerobot-teleoperate --robot.type=robosuite_sim --teleop.type=so101_leader --force_feedback=true`.

**RUN-FROM-SOURCE GOTCHA:** the `lerobot-teleoperate` console entry resolves to the base env / the
`lerobot-1-hpi_act` git WORKTREE (different branch, no `robosuite_sim`). Always run the maintained
tree explicitly: `cd lerobot-1 && PYTHONPATH=src python -m lerobot.scripts.lerobot_teleoperate …`.
`robosuite_sim` is registered for the CLI by a guarded import in the teleop script's robots block.

**`--ff_source` (ABSOLUTE per-source — applies to ALL joints incl. gripper, no per-joint hybrids):**
- `estimator` — `{m}.estimated/tau_ext` from the physics-model estimator (real-arm path; needs the
  torque pipeline). Faithful at rest; LAGS in motion (LPF/FD q̈) → phantom τ_ext, worst where the
  arm sags. The gripper feels HEAVY here (its τ_ext is its own armature·q̈ with a blown-up q̈ from the
  stiff force-saturating jaw — up to ±30 N·m while nothing touches it).
- `ground_truth` — the sim's exact EoM residual `tau_ext`. No estimator lag; still carries the
  at-rest friction band + saturation/self-collision (it's a residual).
- `contact` — `tau_ext_contact` (§3.2): the sim's real contact reaction. **0 in free space**, real on
  grasp/table. Cleanest haptics for the whole arm AND gripper, but SIM-ONLY (no contact oracle on the
  real arm). `--ff_debug` appends `touching: geomA~geomB` so you can SEE which collision geom hits
  (e.g. the visual-less `static_finger_tip` box).
- The estimator pipeline works on the sim because `make_so_torque_pipeline_steps` maps robot_type
  `robosuite_sim → so101_follower` (else SOArmDynamicsModel has no URDF/gear ratios → silent no-FF).

**Leader rendering (`SOLeader.send_torque_feedback`, unchanged here):** per-joint POSITION-mode
saturated-goal — deadband → gain·sign → velocity damping → saturate (cap ∧ physical stall) → slew →
Torque_Limit + goal lead. Engage/release hysteresis (`ff_engage_Nm`/`ff_release_Nm`) and a velocity
gate (free while the joint moves > `ff_max_render_speed_rad_s`). Raising `ff_engage_Nm` (0.4→0.7)
gates the residual friction band on the estimator/ground_truth sources.

**Estimator-side fixes that landed this session (lerobot `so_follower/dynamics/`):**
- **Reflected rotor inertia** added to the Pinocchio `M` (`model.armature` = `STS3215_REFLECTED_ROTOR_INERTIA_kg_m2`
  = 0.028, in `SOArmDynamicsModel`, set before the CasADi copy → both `crba` paths; codegen cache key
  now hashes the armature). Removes the phantom `τ_ext = armature·q̈` ("moment of inertia", worst on
  wrist_roll, link inertia ≈ 0). Adversarially verified (6 lenses): M matches MuJoCo to 8.6e-8 over
  408 poses, G/C/friction/current unchanged.
- **`SOArmSelfCollisionChecker`** (`self_collision.py`, Pinocchio+coal from the URDF collision meshes,
  adjacent pairs filtered) — `is_colliding(q)`, `collisions(q)`, `filter_free(q_rows)` to drop
  self-collision-contaminated samples from calibration (self-collision corrupts the MEASURED τ_motor,
  not the model — the Pinocchio dynamics are collision-INDEPENDENT, verified). Not yet wired into the
  calibration loop (user to integrate).

**Dynamics components verified (Pinocchio == MuJoCo at clean AND self-colliding poses):** gravity ✓,
Coriolis ✓, inertia ✓ (after the armature fix). Collision does NOT leak into G/C/M.

---

## 8. Running Log

### 2026-06-22 — Journal created; architecture consolidated
- Created this journal. Audited the full working-tree state.
- Confirmed `robosuite_private/` is the maintained architecture (797 LoC across 6
  modules). `tests/test_robosuite_private/` → **16 passed**.
- Confirmed the prototype demo/device/wrapper scripts (§5) are gone from the tree
  (superseded by the same-env direct-import design); only stale `.pyc` remain.
- Current tracked diff touches 9 upstream files (§4.5); `robosuite_private/` and
  the new env/sensor XML are untracked.
- **Open questions / next experiments:**
  1. Calibrate per-`robot_id` `kt` (currently `KT_DEFAULT=0.5`) against real
     STS3215 current draw — `load`/`current` realism depends on it.
  2. Friction identification: sweep `set_friction_params` to match real
     break-away + viscous behaviour; validate via `tau_friction` residuals.
  3. Decide `tcp_wrench` frame transform convention (wrist_roll body → TCP) on the
     lerobot-private side and document it here once fixed.
  4. Define the `NoiseProfile` that best matches the real servo bus (encoder
     resolution → `QuantizationNoise`; bus dropout rate → `DropoutNoise`).
  5. Wire a force-feedback closed-loop test: render `tcp_wrench`-derived torque on
     the leader and measure teleop success-rate delta vs. no feedback.

### 2026-06-22 — Ground-truth groups, LeRobot bridge, validation harness
- Split `ground_truth` into **`model.*`** (added `tau_model`) and **`ext.*`** (new
  `tau_ext`, `tau_ext_residual`, `tau_ext_jac`, `tcp_wrench_ext`, `tcp_wrench_ft`).
  Flat view kept; legacy `tcp_wrench` = `tcp_wrench_ft`. New modules
  `dynamics_gt/{model_terms,external,jacobian}.py`, `motor_signals/servo_raw.py`.
- Built the LeRobot side (§7): `robosuite_sim` drop-in Robot + `estimator_vs_truth`
  harness. robosuite_private suite 16→**32 passed**; adapter **8 passed**.
- **Resolves open Q3** (tcp_wrench frame): ground truth now provides the
  world-frame TCP-transported `tcp_wrench_ext` (R2) AND keeps the raw FT
  (`tcp_wrench_ft`) as diagnostic. The remaining convention question moved to the
  estimator side (reaction-sign — §7.2 finding 3).

### 2026-06-22 — ⚠ Model-fidelity gap is the priority: objective XML-vs-URDF test
The harness measured `G_mujoco − G_urdf` ≈ **−0.49 N·m (shoulder_lift), −0.33
(elbow_flex)**, load-independent → a pure inertial-model mismatch. Gravity torque
depends only on the **mass first-moments** (per-link `mass` and `mass·com`), not the
full inertia tensor, so the gap is fully explained by mass/COM differences between
the MJCF and the URDF. Hypothesis: the MJCF (`so101_new_calib`) is closer to the
real arm. **Objective test to decide it (no circular sim self-grading):**

1. **Real static gravity-ID (gold standard).** Command a grid of static poses
   spanning each joint's range (≥2 non-trivial configs per joint, e.g. shoulder_lift
   and elbow_flex at ±30°/±60°). Hold each pose with **zero velocity and no payload**.
   Approach every pose **from both directions** and average the two steady-state
   motor torques → the Coulomb friction band cancels, leaving the friction-free
   `τ_grav_real(q) = G_real(q)`. (Torque from current via the STS constants, §3.5.)
2. **Score both models** at the same `q`: `RMSE(G_mujoco(q) − τ_grav_real)` vs
   `RMSE(G_urdf(q) − τ_grav_real)`. Lower wins, per joint. This is objective and
   needs only real current logs + joint angles.
3. **Localize** with a parameter diff: extract per-link `mass`, `mass·com`, and
   inertia from MJCF `<inertial>` and the URDF, tabulate side-by-side. Gravity gap
   ⇒ look at the first-moments of the distal links (wrist/gripper) about each axis.
4. **Fix** = make the loser match the truth. If real confirms the MJCF, re-derive
   the URDF `<inertial>` blocks from the MJCF (mass, COM, inertia) so the estimator's
   Pinocchio model equals the sim — closing the gap for BOTH sim grading and real
   estimation. Re-run `estimator_vs_truth.py` phase 3: the gravity-gap row must go to
   ~0 and the τ_ext bias must vanish.

Tooling: `scripts/sim_validation/gravity_model_compare.py` — (a) analytic G(q)
sweep MJCF-vs-URDF across the workspace (works today, quantifies the gap
everywhere); (b) `--real-csv` ingest of static-hold logs to score both models
against measured gravity torque (step 2); (c) `--dump-inertials` side-by-side
parameter diff (step 3). See §7.2 finding 2.

### 2026-06-22 — ✅ Model gap ROOT-CAUSED: the `gripper_base` placeholder
Built `gravity_model_compare.py` and ran the analytic sweep + `--dump-inertials`.
**Decisive result:** the entire MJCF↔URDF gravity mismatch is *one body*. Per-link
mass AND COM match to 5 decimals across the whole chain (shoulder 0.10001, upper_arm
0.103, lower_arm 0.104, wrist 0.079, gripper 0.087, jaw 0.012) — same CAD source. The
sole discrepancy: robosuite's gripper attachment body `gripper_base`, given a
hand-entered placeholder in [so101_gripper.xml:10] (`mass="3e-1"`, `pos="0 0 0"`,
`diaginertia="1e-2 1e-2 1e-2"` — clean round numbers, unlike every CAD-derived link),
and **absent from the URDF**. Moving-mass Δ = exactly 0.300 kg.

Sweep confirms the kinematic signature: the missing distal mass loads the *proximal*
joints via long levers — RMSE gap shoulder_lift 0.486, elbow_flex 0.381, wrist_flex
0.128, and ≈0 on shoulder_pan/wrist_roll/gripper (axes that see no gravity torque from
a near-axis load). Exactly the `estimator_vs_truth` finding-2 pattern.

> **⚠ SUPERSEDED PRESCRIPTION (corrected in the resolution entry below).** This entry's
> conclusion — "neither model trustworthy, *measure* the real gripper_base and write it
> into both" — was wrong. Checking upstream proved `gripper_base` has no real counterpart
> at all: it is a robosuite-only mounting frame. The real gripper mass is already in
> `gripper`(0.087)+`moving_jaw`(0.012). The URDF (estimator) was right; the sim was wrong.
> Fix = make `gripper_base` massless, NOT measure-and-copy. Kept for the audit trail.

**Reframed conclusion (later corrected — see above):** the user's "MJCF is closer to
real" is *half* right — MJCF at least models a mass there; URDF ignores it. But 0.3 kg is
a guess (~6× a real STS3215, COM at origin). So the fix is NOT copy-MJCF→URDF; it is
**measure the real gripper_base first moment** and write it into both.

**Objective measurement (two independent routes):**
1. **Scale (fastest).** The gripper base + 6th servo + bracket is a removable
   subassembly — weigh it. Gives the real mass directly.
2. **Bidirectional static gravity-ID + linear identifier.** Hold ≥3 poses that load
   shoulder_lift/elbow_flex, each approached from both directions (cancels Coulomb),
   log current→τ (STS constants §3.5). Feed to `gravity_model_compare.py
   --solve-gripper-base --real-csv holds.csv`: gravity is *exactly linear* in the
   first-moment vector φ=[m, m·cx, m·cy, m·cz], so a per-pose (6×4) regressor + least
   squares solves m and COM for that one body. **Self-test passed** (synthetic MJCF
   truth → recovers m=0.300, COM=origin, RMSE 0.00000), so the math/regressor are
   validated; real data in → real parameter out.

**Fix + verify:** write the measured mass+COM into [so101_gripper.xml:10] `<inertial>`
AND add the matching link/inertial to the URDF gripper, then re-run
`estimator_vs_truth.py` phase 3 — the gravity-gap row and τ_ext bias must go to ~0.
(Note: the placeholder `diaginertia=1e-2` is also wrong — ~0.01 kg·m² is huge for this
part — and corrupts `tau_inertial`/Coriolis during motion; fix it from CAD at the same
time, though gravity/static estimation only needs the first moment.)

CSV format for `--real-csv`/`--solve-gripper-base`: columns `pose_id, approach,
{motor}.pos_deg ×6, {motor}.tau_nm ×6`; rows sharing `pose_id` are averaged across
approach directions.

### 2026-06-22 — ✅✅ RESOLVED + LOCKED: gripper_base was a robosuite phantom; URDF was right
Cloned the authoritative upstream **TheRobotStudio/SO-ARM100** (`fda892cb`) and checked
all three models against it. **Decisive, conclusion-inverting result:**

- `gripper_base` exists in **no real source**. Upstream `Simulation/SO101/so101_new_calib.xml`
  goes `gripper`(0.087)→`moving_jaw`(0.012) directly; the upstream **URDF**, lerobot's URDF
  (md5 `308645ed…`, byte-identical mirror of upstream), and even **SO100's URDF** all lack
  any such body. It is purely a robosuite artifact of the arm/gripper split, which gave it a
  fabricated `mass="3e-1"` placeholder.
- So the earlier "measure it and write into both models" prescription was wrong. The real
  gripper mass (6th servo + jaw) is **already** captured by `gripper`+`moving_jaw` (both
  CAD-derived, identical across MJCF/URDF). **The URDF/Pinocchio model the real-robot
  estimator uses was correct all along; only the robosuite sim copy was inflated** by a
  phantom +0.300 kg (+47.5% of the 0.632 kg arm) sitting at the wrist.

**Fix applied:** [so101_gripper.xml] `gripper_base` `mass 3e-1→1e-9`, `diaginertia 1e-2→1e-9`
(massless mounting frame; the `1e-9` mirrors the URDF dummy-link convention — not a MuJoCo
stability requirement, a workflow agent verified `mass=0`/`1e-9`/no-inertial all compile+step).

**Empirical verification (all green):**
- `gravity_model_compare.py --sweep` (226 poses): `G_mujoco − G_urdf` whole-arm RMSE
  **0.0000 N·m**, every joint `−0.0000` (was shoulder_lift −0.49 / elbow_flex −0.33).
- `--dump-inertials`: `gripper_base` mass 0.000; total moving mass 0.932→**0.632 kg**
  (matches URDF 0.485 + fixed base 0.147).
- `estimator_vs_truth.py` phase-3 frictionless 0 g gravity gap ~0 on all joints; the
  load-independent τ_ext bias is gone. Sim is now a faithful ground-truth oracle.

**Locked to upstream as single source of truth (the "feed both from the real repo" goal).**
robosuite cannot import the monolithic upstream MJCF (it needs the actuator/sensor/eef
split), so upstream owns the *invariant physics* and we pin to it instead of vendoring code:
- `robosuite/models/assets/robots/SOARM101/UPSTREAM.json` — provenance manifest: repo + SHA
  `fda892cb` + per-body mass/COM/fullinertia + per-mesh md5 (13 STLs, all byte-identical to
  upstream). Regenerate via `python -m robosuite_private.tools.gen_upstream_manifest <clone>`.
- `tests/test_robosuite_private/test_soarm101_upstream_inertials.py` — drift guard: reads the
  *compiled* assembled model, asserts every body's mass/COM/inertia == manifest, that
  `gripper_base` stays massless (<1e-6), total mass == 0.632006, and mesh md5s match.
  **Mutation-tested**: restoring `mass=3e-1` turns it red with an actionable message.
- lerobot needs no equivalent — its SO101 dir is already the verbatim upstream mirror.
- robosuite_private suite **39 passed** (was 32); boundary test still green (no lerobot import).

**Also fixed (user-approved):** the gripper actuator was missing the upstream sts3215 class's
`kv="2.731"` (undamped jaw) and used the class-default `forcerange ±2.94` instead of the
per-actuator override `±3.35` (real STS3215 stall torque). Both now match upstream exactly
([so101_gripper.xml] actuator; `kp`/`ctrlrange` already matched). Confirmed in the compiled
model (`biasprm kv=-2.731`, `forcerange ±3.35`); steps stably.

**Decisions:** user chose **vendor + pin** over a git submodule (a submodule only dedups meshes,
is pip-unfriendly, and can't guard the hand-authored split's inertials — the manifest+test do).
The `…/SOARM101/SO101/so101_new_calib.xml` copy is **not** deletable as first thought — `scene.xml`
`<include>`s it for a standalone MuJoCo viewer scene, a path separate from the robosuite env load
(`soarm_with_sensor.xml`+`so101_gripper.xml`) and not covered by the drift guard. Left in place;
reconcile it to upstream later only if the viewer scene matters.

### 2026-06-22 — Force-feedback chain validated headless; rest-friction band is the next lever
New `scripts/sim_validation/ff_sim_preview.py` (lerobot side) — the safe first test of the
sim→leader force-feedback path with NO physical arm: drives `robosuite_sim` into known TCP
loads, reads ground truth, and runs each joint's torque through the leader's *pure* FF safety
stack (`so_leader.compute_ff_torque_limit`) to print the exact `Torque_Limit` the SO-101 would
command. Two signals reported per joint:
- **`tau_phys`** = `tau_ext_jac` = Jᵀw, the exact physical external torque (drives the render).
  **0 g → 0.0000 N·m on every joint** ⇒ the phantom is gone (pre-fix this was a large
  pose-dependent gravity torque the operator would have felt as a constant pull). Loaded rows
  scale cleanly (500 g → shoulder_lift −0.80 / elbow −0.43 → 2 joints engage, ~0.24/0.13 N·m
  rendered). Contact rendering is correct.
- **`tau_resid`** = `τ_motor − τ_model`, what the DEPLOYED estimator computes (no Jᵀw oracle on
  the real arm). At rest it carries a static-friction band (shoulder_lift ~0.63 > the 0.4
  engage threshold). `--coulomb-free` collapses it to `tau_phys` (~0), proving it is friction,
  NOT model error. **Implication:** the model fix removed the gravity phantom; the remaining
  at-rest FF artifact is a friction-calibration / engage-gate concern, not a physics gap.

**Teleop force-feedback source — implemented `--ff_source` (lerobot side).** `lerobot_teleoperate.py`
now selects where the rendered torque comes from:
- `--ff_source=estimator` (default): `{motor}.estimated/tau_ext` — the physics-model estimate from
  motor signals. The real-arm path; LPF-smoothed, carries any estimator model/calib error.
- `--ff_source=ground_truth`: `ground_truth.ext.{motor}.tau_ext` straight off the sim obs. NOTE the
  correct key is the **primary `tau_ext`**, NOT `tau_ext_jac`: `tau_ext_jac`/`tcp_wrench_ext` are
  computed ONLY from `data.xfrc_applied` (programmatic loads), so they read **zero on real MuJoCo
  contacts** — useless for teleop. The primary `tau_ext` falls back to the EoM residual when no
  xfrc is present (the teleop case), which DOES capture real contacts (PD fights contact → τ_motor
  rises → residual rises). So the GT source = the *exact* residual: zero estimator model error
  (isolates the haptic renderer), but unfiltered (spikes harder than the LPF on transients) and
  still carrying the at-rest friction band. Falls back to estimator (with a warning) if the
  follower exposes no `ground_truth.ext.*` keys. GT source also lets `--enable_torque_estimator=false`
  (no Pinocchio/CasADi needed).
- On-screen viewer: added `SOARM101Sim.render()` + the adapter calls it each `send_action` when
  `has_renderer=True`, so `--robot.has_renderer=true` shows the MuJoCo window during teleop.
- The remaining at-rest artifact (friction band, shoulder_lift ~0.6 > 0.4 engage) is the same for
  both sources; the proper clean-at-rest fix is a future contact-force-derived GT term
  (`mj_contactForce` projection), not in scope yet.

### 2026-06-22 — Teleop FF debug: gripper mapping fixed; "stiction" is the home pose clipping the table
First hardware-in-the-loop attempt surfaced three issues; root-caused all via headless diagnostics
(no leader needed). Note the CLI runs the WRONG code by default: the `lerobot-teleoperate` console
script resolves to the base env / the `lerobot-1-hpi_act` git worktree (different branch, no
robosuite_sim). Run from source: `cd lerobot-1 && PYTHONPATH=src python -m lerobot.scripts.lerobot_teleoperate …`.
Also registered `robosuite_sim` for the CLI (guarded import in the teleop script's robots block).

1. **Gripper didn't follow — FIXED.** The GRIP controller had `use_action_scaling=False`
   ([robosuite_sim.py] `_controller_configs`), so the command [-1,1] passed straight to the
   position actuator clamped to ctrlrange: the lower half ([-1,-0.17]) collapsed to the open stop
   (dead zone) and +1 reached only 1.0 of the 1.745 close. Set `use_action_scaling=True` → linear
   full-range mapping; verified jaw sweeps open→closed with no dead zone (the flat 75→100%→~73° is
   the jaws meeting, physically correct).
2. **"Stiction" on shoulder_lift/elbow/wrist — NOT friction.** Added `--robot.zero_coulomb_friction`
   (zeros frictionloss on all joints in `configure()`); zeroing it left the at-rest `tau_ext`
   essentially unchanged (shoulder_lift 0.49→0.54), exonerating the friction model. Decomposed the
   residual: at the home pose `tau_motor≈0` while gravity needs −0.53, i.e. a CONTACT holds the arm.
   `d.ncon` showed `static_finger_tip` ↔ `table_collision` (|F|≈4.7 N); `qfrc_constraint` on the arm
   joints (sl −0.523, wf +0.424) matched the residuals exactly. FF renders that table-contact
   reaction → feels like stiction. (The 4 extra contacts in ncon are the cube resting on the table,
   a red herring.)
3. **Root cause = home pose clips the table.** `init_qpos=[0,0.65,0,1.5,0]`
   ([soarm101_robot.py:34]) is **penetrating the table by ~2.9 cm** (kinematic finger↔table gap at
   sl=0.65 is −0.0285 at every wrist angle). Kinematic sweep: `shoulder_lift≈0.30` clears the table
   at all wrist_flex; sl=0.65 always penetrates. **Fix = lower shoulder_lift in init_qpos to ~0.30.**
   Secondary: the JOINT_POSITION controller (kp=100) sags ~15–30° under gravity and `kp`↑ alone
   makes the table-press worse (it drives the clipped pose harder), so kp is not the lever here.
4. **Real-robot stiction is a DIFFERENT cause** (no table under the real arm at home): most likely
   the unobservable-at-rest Coulomb band the estimator can't subtract → friction calibration /
   compensation is the real-arm fix, not the sim's pose bug.

### 2026-06-22 — In-air stiction RE-ROOT-CAUSED: the residual is the wrong FF signal (actuator saturation)
Follow-up test: the user reports the shoulder_lift/elbow stiction PERSISTS with the arm lifted off
the table, so the home-pose clip (prior entry) is only part of it. Headless test at an in-air pose
(no `static_finger_tip↔table` contact) shows **free-space residuals of −2.6 (elbow) / −2.8 (wrist_flex)
N·m with nothing touching** — these are not external torque. Root cause: the arm uses `<motor>`
torque actuators clamped to ±3.35 N·m ([soarm_with_sensor.xml]) driven by the JOINT_POSITION
computed-torque controller (gravity comp IS on). When the commanded move needs more than ±3.35
(soft `kp=100` lets the arm sag to a joint LIMIT; stiff `kp=1500` saturates on fast moves),
`tau_motor` is clamped and the residual `tau_motor − tau_model` reports the shortfall + the limit
reaction as fake `tau_ext`. FF renders it → stiction that survives lifting the arm.
- `kp` sweep at one in-air pose: kp≤800 sags to the wf limit (residual 3.36); kp=1500 holds THAT
  pose (residual 0.05) but other poses + fast motion still saturate (residual up to 3.9). So `kp`
  alone can't fix it — soft sags, stiff saturates. The residual is fundamentally fragile for FF.
- wrist_roll residual stays small (<0.15, below the 0.4 engage gate) → FF barely touches it; its
  "huge moment of inertia" feel is most likely the physical leader's own wrist_roll rotor inertia
  (hardware), not the sim.
- **Robust fix (recommended): drive FF from the TRUE contact force, not the EoM residual.** Compute
  external joint torque from MuJoCo contacts directly (`mj_contactForce` over `data.contact`,
  projected through the body Jacobian) → a `tau_ext_contact` GT term that is 0 in free space
  regardless of controller saturation/sag/limits, and equals the real reaction on contact. Wire it
  as a new `--ff_source`. This is the clean-at-rest term flagged earlier; the diagnosis now makes it
  necessary. [PROPOSED — pending build.]

### 2026-06-22 — Dynamics-component verification (Pinocchio vs MuJoCo): gravity✓ coriolis✓ INERTIA✗
User redirected: not friction; "model" = URDF contact geometry; hypothesis that self-collision
LEAKS into the Pinocchio-derived dynamics and corrupts τ_ext estimation + friction calibration.
**Tested every component** (sim MuJoCo vs the estimator's own Pinocchio model, both clean and
self-colliding poses):
- **Collision-leak REFUTED, two ways.** (1) Code: the estimator builds with
  `pin.buildModelFromUrdf` ([urdf_loader.py:93]) — no `buildGeomFromUrdf`, so `<collision>` geometry
  is never in the dynamics model; `crba`/`rnea` are inertial-only. (2) Numerical: G/C/M are
  IDENTICAL Pinocchio↔MuJoCo to ~1e-6/1e-8/1e-8 at clean poses AND at poses with up to 32 active
  self-collisions — the dynamics are completely collision-independent. So collision cannot corrupt
  G/C/M; it corrupts the **measured τ_motor** (contact reaction in motor current), which then leaks
  into `τ_ext = τ_motor − τ_model` and into any calibration sample taken at a self-colliding pose.
- **Gravity ✓ and Coriolis ✓ verified correct** (match MuJoCo to numerical noise — gravity already
  fixed via the gripper_base phantom; Coriolis confirmed here for the first time).
- **INERTIA ✗ — concrete model bug.** The estimator's Pinocchio `M(q)` has `rotorInertia=[0…0]`,
  `rotorGearRatio=[1…1]`: it OMITS the reflected rotor inertia. The sim carries `armature=0.028`
  per joint; M only matched after subtracting that diagonal. A 345:1 STS3215 has large reflected
  rotor inertia (J_rotor·N²), so zero is physically wrong. Consequence: during acceleration the
  estimator's `τ_model` is short by `armature·q̈`, so `τ_ext` reads a **phantom torque ∝ q̈** — i.e.
  the operator feels a spurious "moment of inertia," WORST on wrist_roll (smallest link inertia, so
  the missing rotor term dominates). Matches the user's reported wrist_roll feel exactly.
- **Fixes (both friction-free):** (a) add the reflected rotor inertia to the estimator's Pinocchio
  dynamics (`model.armature`/`rotorInertia`·`rotorGearRatio²`; start at the sim's 0.028, then
  verify against real acceleration data per the user's "verify with real load"); (b) fix/verify the
  self-collision geometry (shoulder geoms g0/g1/g2 vs lower_arm/wrist/gripper, §prior entry) so
  τ_motor is clean, and exclude any self-colliding poses from calibration datasets.

### 2026-06-22 — Fix (1) DONE+VERIFIED: rotor inertia added to estimator M (lerobot side)
Added `STS3215_REFLECTED_ROTOR_INERTIA_kg_m2=0.028` (motor_parameters.py; distinct from the
motor-side 0.00012 backlash constant) and a `rotor_inertia` param on `SOArmDynamicsModel` that
writes `model.armature` BEFORE the CasADi compiler copies the model — one assignment reaching both
`pin.crba` (numerical) and `cpin.crba` (symbolic). Wrist_roll's link inertia ≈0, so M[4,4] went
0.000→0.028 (the missing term was its WHOLE effective inertia). Adversarial workflow (6 lenses,
197k tok) verdict:
- ✅ numerical M == MuJoCo mj_fullM over 408 poses (incl. self-colliding) to 8.6e-8; ✅ CasADi M to
  7e-18; ✅ G/C/friction/current bitwise-unchanged; ✅ 47 dynamics tests pass; ✅ 0.028 plausible
  (implied J_rotor 2.35e-7, uncalibrated — flagged for real-data refinement).
- ❌→FIXED a real silent hazard the completeness lens caught: the **codegen `.so` cache key was
  `urdf_hash+casadi_version` only**; since armature isn't in the URDF, the pre-fix `M.so` stayed
  "valid" and `use_codegen=True` (production EKF fast-path) silently loaded the armature-free M.
  Fixed by hashing `model.armature` into the cache key; verified across separate processes
  (0.028→M[4,4]=0.028, 0.05→0.05 with cache regeneration). Pre-existing stale cache auto-invalidates.
- Non-blocking notes: armature applied UNIFORMLY (matches the sim's sts3215 class today, but
  per-joint sourcing from the model would be more robust if a joint ever differs); value is
  upstream-inherited, not yet measured — calibrate against real acceleration data.

### 2026-06-22 — Contact-force FF source built (the clean signal): `tau_ext_contact` + `--ff_source=contact`
Built the robust FF signal that ends the recurring stiction. `compute_contact_ext_torque` (external.py):
reads the sim's ACTUAL contacts via `mj_contactForce` over `data.contact`, keeps only robot↔world
pairs (skips robot↔robot self-collision), and projects each contact wrench (force AND torsional torque
via the point Jacobian `mj_jac` at `c.pos`, jacp+jacr) to joint space. Sign matches `tau_ext_residual`.
- **Verified EXACT**: `tau_ext_contact == −(contact-only efc constraint force)` to 0.0 on all 6 joints at
  multiple poses (isolated CONTACT efc rows from LIMIT/FRICTION rows — the earlier wrist_flex "mismatch"
  was the joint-limit force, which this term correctly excludes). **0 in free space**, immune to
  friction/saturation/self-collision; equals the real reaction on table/cube/grasp contact.
- Wired: added `tau_ext_contact` to `_EXT_JOINT_TERMS`/`_EXT_KEYS`/flat aggregator (+schema tests);
  adapter emits `ground_truth.ext.{m}.tau_ext_contact` and a `read_external_contacts()` debug method;
  teleop `--ff_source=contact` reads it; `--ff_debug` appends a `touching: geomA~geomB` readout so the
  user can SEE which collision geom hits (e.g. the collision-only `static_finger_tip` box, size
  24×4×24 mm at gripper pos (-0.008,-0.004,-0.092), which has no visual — the "invisible geometry").
- FF source is ABSOLUTE per-source (user directive): `--ff_source` ∈ {estimator, ground_truth, contact}
  applies uniformly to ALL joints incl. the gripper — no per-joint exclude/override hybrids (those were
  prototyped then removed). So the gripper feels clean ONLY under `contact` (its estimator/residual
  τ_ext is dominated by its own mis-estimated inertia: armature·q̈ with q̈ blown up by the stiff,
  force-saturating jaw actuator → up to ±30 N·m phantom while nothing touches it; the contact signal is
  0 in free space, real on grasp). NOTE: contact source is SIM-ONLY (no contact oracle on the real arm);
  on hardware the gripper FF should come from the gripper's grip current/load, gated to engage only when
  stalled against an object — a follow-up.
- robosuite_private suite 39 passed.

### 2026-06-22 — Teleop test (estimator FF source): gripper FF loop fixed; estimator faithful at rest, lags in motion
Wired the estimator FF source to work on the sim (map robot_type robosuite_sim→so101_follower in
make_so_torque_pipeline_steps; previously the estimator silently failed to init on the sim → no τ_ext).
Estimator now carries the rotor-inertia fix (armature 0.028). Two issues surfaced from the live run:
- **Gripper oscillation — FIXED.** The estimator's gripper τ_ext swung ±3.5 N·m: the gripper is a stiff
  position actuator (kp=998.22), so its τ_ext is the actuator's own saturating torque, and rendering it
  on the leader gripper closed a feedback loop. Added `ff_exclude_motors` (default `("gripper",)`) to the
  teleop loop — the gripper gets zero FF. Proper grip-force FF would need a contact-force signal, not τ_ext.
- **"Worse on estimator" = dynamic-estimation lag, NOT a calibration bias.** Localized with live
  est-vs-GT: the current round-trip is FAITHFUL (current_raw 23.2 → 1.654 N·m == sim qfrc_actuator 1.65,
  STS constants both ways). At STATIC/off-table poses est ≈ GT (|est−GT| ≤ 0.16). But in MOTION (e.g. the
  weak controller can't hold elbow=−50°, arm drifts) est diverges to −3.7…−5.9 while GT ≈ 0 — the
  LPF/finite-difference acceleration LAGS the true q̈, so τ_inertial/τ_coriolis are wrong → phantom τ_ext.
  Amplified on the sim because the soft position controller (kp) lets the arm sag/move far more than the
  rigid real servos. The elbow ~+0.40 the user felt is this motion lag, not a friction/gravity error.
- **Calibration implication:** friction is already tracked well at rest (|est−GT|~0.1 = the unobservable
  band), so re-calibration is low value. The "worse" feel was the gripper loop (fixed) + acceleration lag
  (sim-sag-amplified, smaller on the rigid real arm; EKF backend handles q̈ better than LPF). Re-test with
  the gripper fix; for a fairer sim estimator test bump --robot.kp and/or --estimation_method=ekf.

### 2026-06-22 — Fix (2): self-collision detector built; sim collision-geometry refinement NOT warranted
User scope (multiselect): build a self-collision detector + refine the sim collision geometry (NOT
the calibration/teleop wiring — to integrate later). Key reconciliations first: lerobot loads the
URDF collision geometry NOWHERE (no buildGeomFromUrdf/computeCollisions/coal in estimation or
calibration), and the geoms are the actual CAD meshes (accurate) — so literally editing the URDF
collision does nothing to the real pipeline, and there's no geometry bug. The self-collision is
PHYSICAL; it corrupts the MEASURED τ_motor at folded poses (not the model).
- **Detector DONE:** `so_follower/dynamics/self_collision.py::SOArmSelfCollisionChecker` (Pinocchio +
  coal/hppfcl from the URDF `<collision>` meshes; drops |Δjoint|≤1 adjacent pairs → 85 meaningful
  pairs). `is_colliding(q)`, `collisions(q)`→link pairs, `filter_free(q_rows)`→mask for dropping
  contaminated calibration samples. Exported from the dynamics package. Validated: agrees with the
  MuJoCo sweep (folded pose → shoulder↔lower_arm/wrist; clean pose → none).
- **Sim refinement NOT warranted (premise didn't hold).** Workspace sweep coal(accurate)↔MuJoCo
  (convex-hull) over 216 poses: REAL pairs match (shoulder↔lower_arm 42, shoulder↔wrist 32,
  shoulder↔gripper 15); convex-hull FALSE positives are negligible (shoulder↔moving_jaw:3,
  shoulder↔wrist:3) and sit on otherwise-real pairs so they can't be excluded without losing real
  detections. The only real gap is the sim UNDER-collides on base↔distal (base body has VISUAL geoms
  only, no collision — the deliberate base↔shoulder_pan fix from git history). So no safe/clean
  surgery: removing the few FPs loses real detections; adding base collision regresses the base↔
  shoulder fix. Recommendation: leave the sim geometry as-is; use the coal detector as the accurate
  reference. The detector (not geometry edits) is the deliverable that actually de-contaminates
  estimation/calibration when the user wires `filter_free` into their calibration loop.

<!-- Append new dated entries below this line. -->

### 2026-07-30 — Foundation repair (red GT tests root-caused) + XLeRobot 17-DoF model (M1)

**Branch `xlerobot-17dof`** (worktree `~/Documents/dev/robosuite-xlerobot`, copy of `fork/patricia-xlerobot`)
— start of the XLeRobot sim port for the coffee project. Env note: `lerobot` conda env now has **mujoco 3.9.0**.

**test_external_gt was red (2 tests) — root causes found, both PHYSICAL constraint forces the
residual was honestly reporting, not pipeline bugs:**
1. **Init pose outside joint range** (since 7421b91f, 2026-07-03): scene-match `init_qpos`
   `shoulder_lift = -1.8174` vs model range ±1.7453 → `LIMIT_JOINT` row fires at reset.
   Fix: clamped to −1.7353 (0.01 margin) in `soarm101_robot.py`; deviation documented.
2. **Phantom self-contact** (since 9859a1cc, 2026-07-30): at the folded pose the fixed-finger
   collision pads penetrate the *shoulder* mesh-collision convex hulls (dist to −3.2 mm; the hulls
   of the non-convex motor-holder brackets bulge past the real part; the real arm holds this pose
   freely). Contact wrench on a distal body → constraint torque spread across joints 1–4, residual
   to −7.4 N·m (> motor clamp — impossible as a real force, which was the tell).
   Fix: `<contact><exclude gripper↔shoulder/>` in `soarm_with_sensor.xml` (same philosophy as the
   historical base-collision removal). 750a7484's `mj_fullM` shim was **exonerated**: dense M ==
   `mj_mulM` to 1e-16 under mujoco 3.9.0.
   **Implication:** any July force recording at/near the scene-match pose carries these phantom
   torques at episode start — prime suspect for the reported SOARM101 GT-vs-estimate mismatch.
   Re-grade after this fix before hunting estimator bugs.
3. Hygiene: GT identity fixture now `initialization_noise=None` (σ=0.02 noise could re-enter the
   limit); regression guard `test_soarm101_init_qpos_within_joint_ranges`.
4. Restored kwargs orphaned by the cup rework: `cube_mass` (payload knob for estimator sweeps —
   now applied to the compiled cup body in `_setup_references`, default `None` = keep XML mass, so
   task3/5's 0.215 kg water cup is untouched) and `cube_offset` (wired into the default sampler;
   task envs unaffected — they place the cup via `cup_start_offset`).

**M1 — XLeRobot 17-DoF robosuite model** (kinematics-first; force-oracle generalization is M2):
- `models/assets/robots/XLeRobot/robot.xml`: two **verbatim calibrated SO-101 chains**
  (drift-guarded inertials; right chain first — robosuite splits arm joints positionally) with
  `right_`/`left_` prefixes, per-arm wrist FT sites+sensors, per-arm gripper↔shoulder excludes;
  2-DoF head (upstream has joints but no actuators/geoms — added position actuators, estimate
  masses, RealSense stand-in camera `head_cam` fovy 42.5); chassis = RÅSKOG-footprint primitive
  (collision box only to z=0.60 so folded arms can never phantom-contact it; placeholder inertial
  9 kg — Tier-2 TODO).
- **Mount transforms** from upstream Vector-Wangel/XLeRobot `simulation/mujoco/xlerobot.xml`
  @ 3d14695e, remapped Rz(−90°) to robosuite base frame: arms (0.1352, ∓0.15, 0.8215) yaw 0,
  head pan (−0.125, 0, 0.945), tilt +(0.05, 0, 0.18). Registry `XLEROBOT_FRAME_TREE` in
  `xlerobot_robot.py`, drift-guarded by `test_xlerobot_frame_tree_matches_mjcf`.
  **Must be hardware-verified** (Project_definition §3.3 Validations A/C) before platform force
  data is trusted. Upstream MJCF is a geometry donor ONLY — its arm inertials are SO-100
  (+58% upper-arm mass vs our calib), chassis 121.6 kg placeholder, zero joint friction/armature.
- Base = `NullMobileBase` virtual planar joints (forward/side/yaw velocity) — the robosuite
  convention AND the real robot's base command interface (body-frame x/y/θ vel; kiwi mixing stays
  on the host, as on hardware). Physical wheels not simulated.
- `XLeRobotGripper` (so101_gripper.py): arm-unique namespace `robot0_{right,left}_` so the jaw
  reads as each arm's 6th motor (`robot0_right_gripper`), matching the real per-arm motor layout.
- Default composite `default_xlerobot.json`: arms JOINT_POSITION(kp=100)+GRIP, head JOINT_POSITION,
  base JOINT_VELOCITY → **17-dim action = [R arm 5, R grip 1, L arm 5, L grip 1, head 2, base 3]**,
  semantically identical to the real 17-motor interface.
- Tests `tests/test_robots/test_xlerobot.py` (5): registration, part classification + positional
  split, frame-tree guard, init-vs-range guard, 17-dim env smoke + FT sensors. Full suite green
  (61 private/robot/task + composite matrix; SO101Gripper×{Baxter,Tiago,GR1} pre-existing failures
  resolved by skipping robot-specific adapter grippers in the combo matrix, with rationale).

**Next (M2→M5):** per-arm GT oracle (dynamics_gt takes per-arm joint sets; per-arm robot_spec /
motor_signals namespaces) → EKF-vs-GT free-motion grading (the never-run E04), single desk arm
FIRST, then per mounted arm with the estimator gravity-rotation hook (set `model.gravity.linear`
after `urdf_loader.py:93`, before the CasADi copy; codegen cache key must gain the mount rotation)
→ payload slope on-platform (cube_mass knob) → coffee tasks re-parented onto XLeRobot → sim E-FC
(drive virtual base, arms free, GT contact = 0 by construction).

#!/usr/bin/env python
"""Probe: what limits stability when dof_armature is zeroed (real-EKF parity)?

Checks actuator wiring (motor vs position, ctrlrange, gear), the armature-free mass
matrix (diag + eigenvalues), how to mutate the JOINT_POSITION kp/kd at runtime, and a
short stability screen for a few gain pairs.  Simulation only.

  cd /home/ecappiell/Documents/dev/robosuite
  PYTHONPATH=$PWD:/home/ecappiell/Documents/dev/lerobot-1-tg/src \
    conda run -n lerobot --no-capture-output python -u scripts_tmp/sim_armature0_probe.py
"""

from __future__ import annotations

import time

import mujoco
import numpy as np

from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
DT = 0.001


def raw_model(m):
    while not isinstance(m, mujoco.MjModel):
        m = m._model
    return m


def raw_data(d):
    while not isinstance(d, mujoco.MjData):
        d = d._data
    return d


def main() -> int:
    ov = {m: {"armature": 0.0} for m in MOTORS}
    cfg = RobosuiteSimConfig(id="sim", control_freq=500, dynamics_robot_id="FT-02",
                             emit_ground_truth=True, has_renderer=False,
                             friction_overrides=ov, kp=100.0, kv=1.0)
    t0 = time.perf_counter()
    bot = RobosuiteSimFollower(cfg)
    bot.connect()
    print(f"env connect time = {time.perf_counter() - t0:.2f} s")
    env = bot.sim.env
    model, data = raw_model(env.sim.model), raw_data(env.sim.data)

    print(f"timestep={model.opt.timestep} integrator={model.opt.integrator} "
          f"iterations={model.opt.iterations} cone={model.opt.cone}")
    print(f"enableflags={model.opt.enableflags:#x} "
          f"(mjENBL_INTEG_DAMPING bit = {1 << 5}, mjENBL_INTEG_VEL={1 << 4}, "
          f"mjENBL_INTEG_POS={1 << 3}, filterparent={1 << 0}, multibgndamp={1 << 6})")

    print(f"\n{'actuator':28} {'trntype':>8} {'dyntype':>8} {'gear':>7} {'ctrlrange':>18} "
          f"{'forcerange':>18} {'ctrllimited':>12} {'forcerange_lim':>15}")
    for i in range(model.nu):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "?"
        print(f"{nm:28} {model.actuator_trntype[i]:>8} {model.actuator_dyntype[i]:>8} "
              f"{model.actuator_gear[i, 0]:>7.3f} "
              f"{model.actuator_ctrlrange[i, 0]:>8.3f}..{model.actuator_ctrlrange[i, 1]:<8.3f} "
              f"{model.actuator_forcerange[i, 0]:>8.3f}..{model.actuator_forcerange[i, 1]:<8.3f} "
              f"{model.actuator_ctrllimited[i]:>12} {model.actuator_forcelimited[i]:>15}")

    dof = {}
    for m in MOTORS:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_{m}")
        dof[m] = int(model.jnt_dofadr[j])
    print("\narmature after override =", np.array2string(
        np.array([model.dof_armature[dof[m]] for m in MOTORS])))
    print("dof_damping =", np.array2string(np.array([model.dof_damping[dof[m]] for m in MOTORS])))
    print("dof_frictionloss =", np.array2string(np.array([model.dof_frictionloss[dof[m]] for m in MOTORS])))

    mujoco.mj_forward(model, data)
    nv = int(model.nv)
    M = np.zeros((nv, nv))
    mujoco.mj_fullM(model, M, data.qM)
    arm_idx = [dof[m] for m in MOTORS]
    Ma = M[np.ix_(arm_idx, arm_idx)]
    print(f"\narm-block mass matrix (armature 0), N.m·s^2/rad:\n{np.array2string(Ma, precision=6)}")
    print(f"arm-block eigenvalues = {np.array2string(np.linalg.eigvalsh(Ma), precision=7)}")
    print(f"full-model eigenvalues = {np.array2string(np.linalg.eigvalsh(M), precision=7)}")
    print(f"arm-block diag = {np.array2string(np.diag(Ma), precision=6)}")
    print("\nper-joint explicit-Euler limits at dt=1e-3: kv_max = 2M/dt, kp_max = 4M/dt^2")
    for i, m in enumerate(MOTORS):
        Mi = float(Ma[i, i])
        print(f"   {m:14} M={Mi:.6f}  kv<{2 * Mi / DT:9.3f}  kp<{4 * Mi / DT ** 2:12.1f}")
    ev = np.linalg.eigvalsh(Ma)
    print(f"eigenvalue limits: kv<{2 * ev.min() / DT:.4f}  kp<{4 * ev.min() / DT ** 2:.1f}  "
          f"(worst); kv<{2 * ev.max() / DT:.3f} kp<{4 * ev.max() / DT ** 2:.1f} (best)")

    # --- controller handle for runtime kp/kd mutation -------------------------
    rob = env.robots[0]
    cc = rob.composite_controller
    print(f"\ncomposite controller: {type(cc).__name__}")
    parts = getattr(cc, "part_controllers", None)
    print(f"  part_controllers keys: {sorted(parts) if parts else None}")
    if parts:
        for k, v in parts.items():
            print(f"   {k}: {type(v).__name__} kp={getattr(v, 'kp', None)} kd={getattr(v, 'kd', None)}")
    print(f"  robot._joint_actuator_indexes = {rob._ref_arm_joint_actuator_indexes}")
    print(f"  robot.action_dim = {rob.action_dim}")

    # --- short stability screen ---------------------------------------------
    obs = bot.get_observation()
    init_deg = {m: float(obs[f"{m}.pos"]) for m in MOTORS}
    print(f"\ninit pose deg = {init_deg}")

    per_dof = [0.0] * 12
    qpos0, qvel0 = data.qpos.copy(), data.qvel.copy()

    def restore():
        data.qpos[:] = qpos0
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        mujoco.mj_forward(model, data)

    def run(kp, kv, n=500):
        restore()
        parts["right"].kp = np.full(5, float(kp))
        parts["right"].kd = np.full(5, float(kv))
        qacc_max = qvel_max = 0.0
        nans = 0
        pd = [0.0] * 12
        for i in range(n):
            bot.send_action({f"{m}.pos": init_deg[m] for m in MOTORS})
            d = env.sim.data
            if not (np.isfinite(d.qacc).all() and np.isfinite(d.qvel).all()):
                nans += 1
                break
            qacc_max = max(qacc_max, float(np.max(np.abs(d.qacc))))
            qvel_max = max(qvel_max, float(np.max(np.abs(d.qvel))))
            for k in range(12):
                pd[k] = max(pd[k], float(abs(d.qacc[k])))
        per_dof.clear()
        per_dof.extend(pd)
        return qacc_max, qvel_max, nans

    print("\nactuator gainprm / biasprm (position actuators have biasprm=[0,-kv,-kp]):")
    for i in range(model.nu):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "?"
        print(f"   {nm:24} gainprm={np.array2string(model.actuator_gainprm[i], precision=3)} "
              f"biasprm={np.array2string(model.actuator_biasprm[i], precision=3)}")

    print(f"\n{'kp':>6} {'kv':>6} {'max|qacc|':>12} {'max|qvel|':>12} nan  per-dof max|qacc|")
    for kp in (100, 200, 400, 800, 1500):
        for kv in (0.5, 1, 2, 4):
            qa, qv, nn = run(kp, kv, 300)
            print(f"{kp:>6} {kv:>6} {qa:>12.1f} {qv:>12.3f} {nn:>3}  "
                  + np.array2string(np.array(per_dof), precision=1))
    bot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

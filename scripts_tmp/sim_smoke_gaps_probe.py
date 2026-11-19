#!/usr/bin/env python
"""Quantify the sim-vs-real-FT-02 dynamics gaps (mass matrix with/without MuJoCo armature,
gravity torque scale, Coulomb that the identified calibration did NOT map).

Companion to sim_smoke_1000hz.py.  Simulation only — never touches a serial port.

  cd /home/ecappiell/Documents/dev/robosuite
  PYTHONPATH=$PWD:/home/ecappiell/Documents/dev/lerobot-1-tg/src \
    conda run -n lerobot --no-capture-output python -u scripts_tmp/sim_smoke_gaps_probe.py
"""

from __future__ import annotations

import mujoco
import numpy as np

from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
G = 9.80665


def raw_model(m):
    while not isinstance(m, mujoco.MjModel):
        m = m._model
    return m


def raw_data(d):
    while not isinstance(d, mujoco.MjData):
        d = d._data
    return d


def main() -> int:
    cfg = RobosuiteSimConfig(id="sim", control_freq=500, dynamics_robot_id="FT-02",
                             emit_ground_truth=True, has_renderer=False)
    bot = RobosuiteSimFollower(cfg)
    bot.connect()
    model, data = raw_model(bot.sim.env.sim.model), raw_data(bot.sim.env.sim.data)

    obs = bot.get_observation()
    init_deg = {m: float(obs[f"{m}.pos"]) for m in MOTORS}
    for _ in range(250):  # settle
        bot.send_action({f"{m}.pos": init_deg[m] for m in MOTORS})
    for _ in range(1750):  # long hold, as in T4
        bot.send_action({f"{m}.pos": init_deg[m] for m in MOTORS})
    obs = bot.get_observation()

    dof = {}
    for m in MOTORS:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_{m}")
        dof[m] = int(model.jnt_dofadr[j])

    def qM_diag():
        mujoco.mj_forward(model, data)          # fills qM (link inertia + armature)
        nv = int(model.nv)
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(model, M, data.qM)
        return np.diag(M).copy()

    diag_with = qM_diag()
    arm = np.array([model.dof_armature[dof[m]] for m in MOTORS])
    saved = model.dof_armature.copy()
    model.dof_armature[list(dof.values())] = 0.0
    diag_without = qM_diag()
    model.dof_armature[:] = saved

    gt = {k: np.array([obs[f"ground_truth.model.{m}.{k}"] for m in MOTORS])
          for k in ("tau_gravity", "tau_motor", "tau_friction")}
    tau_ext = np.array([obs[f"ground_truth.ext.{m}.tau_ext"] for m in MOTORS])

    print(f"\n{'joint':14} {'armature':>9} {'M(q)diag arm':>13} {'M(q)diag noarm':>15} "
          f"{'arm share':>10} {'fricloss':>9} {'damping':>8} {'tau_grav':>9} {'tau_fric':>9}")
    for i, m in enumerate(MOTORS):
        print(f"{m:14} {arm[i]:9.4f} {diag_with[i]:13.5f} {diag_without[i]:15.5f} "
              f"{100 * (diag_with[i] - diag_without[i]) / max(diag_with[i], 1e-12):9.1f}% "
              f"{model.dof_frictionloss[dof[m]]:9.4f} {model.dof_damping[dof[m]]:8.4f} "
              f"{gt['tau_gravity'][i]:9.4f} {gt['tau_friction'][i]:9.4f}")

    print(f"\narmature vector = {np.array2string(arm)}")
    print(f"mass-matrix diag WITH armature    = {np.array2string(diag_with, precision=5)}")
    print(f"mass-matrix diag WITHOUT armature = {np.array2string(diag_without, precision=5)}")
    print(f"tau_gravity (sim, N.m) = {np.array2string(gt['tau_gravity'], precision=4)}")
    print(f"tau_motor   (sim, N.m) = {np.array2string(gt['tau_motor'], precision=4)}")
    print(f"tau_friction(sim, N.m) = {np.array2string(gt['tau_friction'], precision=4)}")
    print(f"tau_ext     (sim, N.m) = {np.array2string(tau_ext, precision=4)}  "
          "(free-space hold: should be ~0; non-zero = unmodelled hold torque)")
    print("static-friction band the sim enforces (frictionloss, N.m) = "
          f"{np.array2string(np.array([model.dof_frictionloss[dof[m]] for m in MOTORS]))}")
    print("FT-02 identified coulomb_Fc: pan 0.0247, lift 0.0, elbow 0.0, wrist_flex 0.0, "
          "wrist_roll 0.0134, gripper 0.0425 -> joints with Fc=0 keep the XML 0.052 in sim.")

    # --- is the arm actually holding the commanded pose? -------------------
    print("\n-- hold tracking / contact audit after 2000 steps --")
    err = {m: init_deg[m] - float(obs[f"{m}.pos"]) for m in MOTORS}
    print("pos error (deg)   = " + np.array2string(
        np.array([err[m] for m in MOTORS]), precision=3))
    print("qvel (rad/s)      = " + np.array2string(
        np.array([float(obs[f"{m}.vel_hw"]) for m in MOTORS]), precision=4))
    print("qfrc_actuator(N.m)= " + np.array2string(
        np.array([data.qfrc_actuator[dof[m]] for m in MOTORS]), precision=4))
    print("qfrc_bias  (N.m)  = " + np.array2string(
        np.array([data.qfrc_bias[dof[m]] for m in MOTORS]), precision=4))
    print("qacc       (rad/s2)= " + np.array2string(np.array(data.qacc), precision=4))
    print(f"ncon (total contacts) = {data.ncon}")
    rb = set()
    for b in range(model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if nm.startswith("robot0_"):
            rb.add(b)
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = int(model.geom_bodyid[c.geom1])
        b2 = int(model.geom_bodyid[c.geom2])
        if (b1 in rb) != (b2 in rb):
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            print(f"   external contact #{i}: {g1} <-> {g2} dist={c.dist:.5f}")
    print("body names: " + ", ".join(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "?" for b in range(model.nbody)))

    # --- gripper jaw: position-actuator (kp 998.22 kv 2.731, forcerange +-3.35) droop ------
    gj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot0_gripper")
    gadr = int(model.jnt_qposadr[gj])
    jaw_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_moving_jaw")
    print(f"\ngripper joint qpos = {np.rad2deg(data.qpos[gadr]):+.3f} deg "
          f"(commanded {init_deg['gripper']:+.3f}); "
          f"actuator force = {data.qfrc_actuator[dof['gripper']]:+.4f} N.m; "
          f"jaw z = {data.xpos[jaw_body][2]:+.4f} m")
    print("effective sim joint stiffness = M(q)*kp = "
          f"{np.array2string(diag_with[:6] * 100.0, precision=2)} N.m/rad "
          "(JOINT_POSITION with use_torque_compensation=True pre-multiplies the PD output by M)")
    bot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

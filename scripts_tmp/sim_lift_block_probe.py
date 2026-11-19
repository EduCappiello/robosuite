#!/usr/bin/env python
"""Why does shoulder_lift barely track a 10 deg sine?  Audit joint ranges, contacts and
actuator torques.  Simulation only.

  cd /home/ecappiell/Documents/dev/robosuite
  PYTHONPATH=$PWD:/home/ecappiell/Documents/dev/lerobot-1-tg/src \
    conda run -n lerobot --no-capture-output python -u scripts_tmp/sim_lift_block_probe.py
"""

from __future__ import annotations

import mujoco
import numpy as np

from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


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
                             emit_ground_truth=True, has_renderer=False,
                             friction_overrides={m: {"armature": 0.0} for m in MOTORS[:5]},
                             kp=400.0, kv=2.0)
    bot = RobosuiteSimFollower(cfg)
    bot.connect()
    env = bot.sim.env
    model, data = raw_model(env.sim.model), raw_data(env.sim.data)

    print(f"{'joint':22} {'type':>8} {'range(rad)':>22} {'range(deg)':>22} {'qpos':>10}")
    for m in MOTORS:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_{m}")
        r = model.jnt_range[j]
        print(f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j):22} "
              f"{model.jnt_type[j]:>8} [{r[0]:+.4f},{r[1]:+.4f}] "
              f"[{np.rad2deg(r[0]):+.2f},{np.rad2deg(r[1]):+.2f}] "
              f"{np.rad2deg(data.qpos[model.jnt_qposadr[j]]):>+10.3f}")
    print("\nkeyframe / init qpos (deg):")
    for k in range(model.nkey):
        print(f"   key {k} ({mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, k)}): "
              f"{np.array2string(np.rad2deg(model.key_qpos[k][:6]), precision=3)}")

    obs = bot.get_observation()
    init_deg = {m: float(obs[f"{m}.pos"]) for m in MOTORS}
    print(f"\nobs init deg = {init_deg}")

    def audit(tag):
        print(f"\n-- {tag} --")
        print("ncon =", data.ncon)
        rb = {b for b in range(model.nbody)
              if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("robot0_")}
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = int(model.geom_bodyid[c.geom1])
            b2 = int(model.geom_bodyid[c.geom2])
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            bn1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1)
            bn2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2)
            print(f"   #{i} {n1}({bn1}) <-> {n2}({bn2}) dist={c.dist:+.6f} "
                  f"frame={np.array2string(c.frame[:3], precision=4)} ext={(b1 in rb) != (b2 in rb)}")
        dof = {m: int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_{m}")])
               for m in MOTORS}
        print("   qpos(deg)   = " + np.array2string(
            np.array([np.rad2deg(data.qpos[model.jnt_qposadr[dof[m]]]) for m in MOTORS]), precision=3))
        print("   ctrl(N.m)   = " + np.array2string(np.array(data.ctrl[:6]), precision=4))
        print("   qfrc_act    = " + np.array2string(np.array([data.qfrc_actuator[dof[m]] for m in MOTORS]), precision=4))
        print("   qfrc_bias   = " + np.array2string(np.array([data.qfrc_bias[dof[m]] for m in MOTORS]), precision=4))
        print("   qfrc_constraint = " + np.array2string(np.array([data.qfrc_constraint[dof[m]] for m in MOTORS]), precision=4))
        print("   qacc        = " + np.array2string(np.array([data.qacc[dof[m]] for m in MOTORS]), precision=4))

    audit("after connect")
    for _ in range(2000):
        bot.send_action({f"{m}.pos": init_deg[m] for m in MOTORS})
    audit("after 2000-step hold at init")

    # sine on lift
    for i in range(2000):
        t = dict(init_deg)
        t["shoulder_lift"] = init_deg["shoulder_lift"] + 10.0 * np.sin(2 * np.pi * 0.5 * i / 500.0)
        bot.send_action({f"{m}.pos": t[m] for m in MOTORS})
    audit("after 2000-step 0.5Hz +/-10deg sine on shoulder_lift")
    print(f"\ncontroller goal_qpos(deg) = "
          f"{np.array2string(np.rad2deg(env.robots[0].composite_controller.part_controllers['right'].goal_qpos), precision=3)}")
    print(f"controller joint_pos(deg)  = "
          f"{np.array2string(np.rad2deg(env.robots[0].composite_controller.part_controllers['right'].joint_pos), precision=3)}")
    print(f"controller torques(N.m)    = "
          f"{np.array2string(env.robots[0].composite_controller.part_controllers['right'].torques, precision=4)}")
    print(f"controller mass_matrix diag= "
          f"{np.array2string(np.diag(env.robots[0].composite_controller.part_controllers['right'].mass_matrix), precision=6)}")
    print(f"controller torque_compensation = "
          f"{np.array2string(env.robots[0].composite_controller.part_controllers['right'].torque_compensation, precision=4)}")
    bot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

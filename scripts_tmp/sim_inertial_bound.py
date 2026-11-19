#!/usr/bin/env python
"""Can (b) work? Re-derive link inertials so M(q) WITHOUT armature is large enough to be
stable at 1000 Hz physics / 500 Hz control with the current PD gains?

For each joint report:
  M_now      mass-matrix diagonal with armature removed (the number we must raise)
  M_needed   smallest M that satisfies explicit-Euler limits at dt=1e-3 for (kp, kv)
  M_maxphys  hard upper bound: ALL distal subtree mass placed at the farthest reachable
             point from the joint axis. No inertial re-derivation can exceed this.
If M_needed > M_maxphys the joint cannot be stabilised by any physically valid inertia
re-derivation, and (b) is off the table for it.

Run:
  cd ~/Documents/dev/robosuite
  PYTHONPATH=$PWD:~/Documents/dev/lerobot-1-tg/src \
    conda run -n lerobot --no-capture-output python -u scripts_tmp/sim_inertial_bound.py
"""

import numpy as np
import mujoco

from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
DT = 0.001
GAINS = [(100.0, 20.0), (100.0, 1.0), (50.0, 0.5)]


def main() -> int:
    cfg = RobosuiteSimConfig(id="sim", control_freq=500, dynamics_robot_id="FT-02",
                             emit_ground_truth=False)
    bot = RobosuiteSimFollower(cfg)
    # armature zeroing is guarded; bypass the guard by applying it after connect
    bot.connect()
    m, d = bot._model, bot._data
    dof = {k: int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_{k}")])
           for k in MOTORS}
    jnt = {k: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_{k}") for k in MOTORS}

    for k in MOTORS:
        m.dof_armature[dof[k]] = 0.0
    mujoco.mj_forward(m, d)

    nq = m.nq
    Mfull = np.zeros((m.nv, m.nv))
    mujoco.mj_crb(m, d)
    try:
        mujoco.mj_fullM(m, Mfull, d.qM)
    except TypeError:
        mujoco.mj_fullM(m, Mfull.ravel(), d.qM)

    print(f"{'joint':14} {'M_now':>10} {'M_needed':>10} {'M_maxphys':>11}  verdict")
    print("-" * 78)
    for k in MOTORS:
        Mi = float(Mfull[dof[k], dof[k]])
        needed = max(kv * DT / 2.0 for _kp, kv in GAINS)
        needed = max(needed, max(kp * DT ** 2 / 4.0 for kp, _kv in GAINS))

        # distal subtree: bodies whose kinematic path passes through this joint's body
        jbody = int(m.jnt_bodyid[jnt[k]])
        axis = np.asarray(m.jnt_axis[jnt[k]], dtype=float)
        # walk weldid/parent chain for every body
        distal, total_mass = [], 0.0
        for b in range(m.nbody):
            anc, ok = b, False
            while anc > 0:
                if anc == jbody:
                    ok = True
                    break
                anc = int(m.body_parentid[anc])
            if ok:
                pos = np.asarray(d.xpos[b], dtype=float)
                r = pos - np.asarray(d.xpos[jbody], dtype=float)
                perp = float(np.linalg.norm(r - np.dot(r, axis) * axis))
                bm = float(m.body_mass[b])
                distal.append((bm, perp))
                total_mass += bm
        rmax = max((p for _bm, p in distal), default=0.0)
        # upper bound: point masses at their actual farthest radius (sum m*r^2) is the
        # honest bound; the absolute bound is total_mass * rmax^2.
        sum_mr2 = sum(bm * p * p for bm, p in distal)
        m_maxphys = total_mass * rmax * rmax

        verdict = "FEASIBLE" if needed <= m_maxphys else "IMPOSSIBLE"
        print(f"{k:14} {Mi:10.3e} {needed:10.3e} {m_maxphys:11.3e}  {verdict}"
              f"   (sum m*r^2={sum_mr2:.3e}, subtree mass={total_mass:.4f} kg, rmax={rmax:.3f} m)")
    bot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

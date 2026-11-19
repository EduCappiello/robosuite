#!/usr/bin/env python
"""Armature-zero gain screen: which (kp, kv) keep the SOARM101 sim stable at
1000 Hz physics / 500 Hz control when dof_armature is zeroed?

Simulation only — never opens a serial port.

  cd /home/ecappiell/Documents/dev/robosuite
  PYTHONPATH=$PWD:/home/ecappiell/Documents/dev/lerobot-1-tg/src \
    conda run -n lerobot --no-capture-output python -u scripts_tmp/sim_armature0_gains.py [--variant A|B|C]

Variants (which armatures are zeroed):
  A  all 6 joints armature=0            (literal task spec)
  B  5 arm joints armature=0, gripper keeps its XML 0.028
  C  all 6 joints armature=0 + gripper <position> actuator gains rescaled to the
     armature-free inertia (kv_g, kp_g chosen from the same explicit-Euler limits)
"""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM = MOTORS[:5]
DT = 0.001
CTRL_HZ = 500
N_HOLD = 2000
N_SINE = 2000
SINE_HZ = 0.5
SINE_AMP_DEG = 10.0
SINE_JOINTS = ("shoulder_lift", "elbow_flex")
QACC_UNSTABLE = 5000.0

KPS = (100.0, 200.0, 400.0, 800.0, 1500.0)
KVS = (0.5, 1.0, 2.0, 4.0)


def raw_model(m):
    while not isinstance(m, mujoco.MjModel):
        m = m._model
    return m


def raw_data(d):
    while not isinstance(d, mujoco.MjData):
        d = d._data
    return d


def dof_of(model, name):
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_{name}")
    assert j >= 0, f"robot0_{name} not found"
    return int(model.jnt_dofadr[j]), int(model.jnt_qposadr[j])


def arm_mass_diag(model, data):
    mujoco.mj_forward(model, data)
    nv = int(model.nv)
    M = np.zeros((nv, nv))
    mujoco.mj_fullM(model, M, data.qM)
    return M


class Runner:
    """One env, gain pairs swapped in place, state restored from a snapshot."""

    def __init__(self, variant: str):
        if variant == "A":
            ov = {m: {"armature": 0.0} for m in MOTORS}
        elif variant == "B":
            ov = {m: {"armature": 0.0} for m in ARM}
        elif variant == "C":
            ov = {m: {"armature": 0.0} for m in MOTORS}
        elif variant == "BASE":
            ov = None                     # XML/identified armature 0.028 kept
        else:
            raise ValueError(variant)
        cfg = RobosuiteSimConfig(id="sim", control_freq=CTRL_HZ, dynamics_robot_id="FT-02",
                                 emit_ground_truth=True, has_renderer=False,
                                 friction_overrides=ov, kp=100.0, kv=1.0)
        self.bot = RobosuiteSimFollower(cfg)
        self.bot.connect()
        self.env = self.bot.sim.env
        self.model = raw_model(self.env.sim.model)
        self.data = raw_data(self.env.sim.data)
        self.ctrl = self.env.robots[0].composite_controller.part_controllers["right"]

        self.dof, self.qadr = {}, {}
        for m in MOTORS:
            self.dof[m], self.qadr[m] = dof_of(self.model, m)

        if variant == "C":
            # Rescale the gripper <position> actuator to the armature-free inertia:
            # kv_g = 1.6*M/dt (kv*dt/M = 1.6), kp_g = (2/dt)^2 * 0.25 -> kv*dt/M=1.6,
            # kp*dt^2/M = 1.0 (critically-ish damped, 2x margin on both limits).
            M = arm_mass_diag(self.model, self.data)
            Mg = float(M[self.dof["gripper"], self.dof["gripper"]])
            gu = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot0_gripper"))
            self.grip_kp0 = float(self.model.actuator_biasprm[gu, 1])
            self.grip_kv0 = float(self.model.actuator_biasprm[gu, 2])
            new_kv = 1.6 * Mg / DT
            new_kp = 1.0 * Mg / DT ** 2
            # <position> actuator: f = gainprm[0]*ctrl + biasprm[1]*qpos + biasprm[2]*qvel
            self.model.actuator_biasprm[gu, 1] = -new_kp
            self.model.actuator_biasprm[gu, 2] = -new_kv
            self.grip_kp_new, self.grip_kv_new = new_kp, new_kv
            print(f"[variant C] gripper actuator {self.grip_kp0:.2f}/{self.grip_kv0:.3f} -> "
                  f"{self.grip_kp_new:.3f}/{self.grip_kv_new:.4f} (Mg={Mg:.3e})")

        obs = self.bot.get_observation()
        self.init_deg = {m: float(obs[f"{m}.pos"]) for m in MOTORS}
        # settle with the construction gains, then snapshot
        self._drive(self.init_deg, 250)
        self.snap = (self.data.qpos.copy(), self.data.qvel.copy(), float(self.data.time))

    def _drive(self, targets_deg, n):
        for _ in range(n):
            self.bot.send_action({f"{m}.pos": float(targets_deg[m]) for m in MOTORS})

    def restore(self):
        self.data.qpos[:] = self.snap[0]
        self.data.qvel[:] = self.snap[1]
        self.data.qacc[:] = 0.0
        self.data.time = self.snap[2]
        mujoco.mj_forward(self.model, self.data)

    def set_gains(self, kp, kv):
        self.ctrl.kp = np.full(self.ctrl.control_dim, float(kp))
        self.ctrl.kd = np.full(self.ctrl.control_dim, float(kv))

    def measure(self, kp, kv):
        self.restore()
        self.set_gains(kp, kv)
        st = dict(qacc=0.0, qvel=0.0, nan=0, sat=0, dof_acc={})
        dof_max = {}

        def step(targets):
            self.bot.send_action({f"{m}.pos": float(targets[m]) for m in MOTORS})
            d = self.env.sim.data
            if not (np.isfinite(d.qacc).all() and np.isfinite(d.qvel).all()
                    and np.isfinite(d.qfrc_bias).all()):
                st["nan"] += 1
                return
            st["qacc"] = max(st["qacc"], float(np.max(np.abs(d.qacc))))
            st["qvel"] = max(st["qvel"], float(np.max(np.abs(d.qvel))))
            c = np.asarray(d.ctrl[:5], dtype=float)
            st["sat"] += int(np.sum(np.abs(c) >= 3.349))
            for m in MOTORS:
                k = self.dof[m]
                dof_max[m] = max(dof_max.get(m, 0.0), float(abs(d.qacc[k])))

        t0 = time.perf_counter()
        # ---- phase 1: gravity sag, hold init pose
        for _ in range(N_HOLD):
            step(self.init_deg)
        sag = {m: float(np.rad2deg(self.data.qpos[self.qadr[m]]) - self.init_deg[m])
               for m in MOTORS}

        # ---- phase 2: 0.5 Hz +/-10 deg sine on lift + elbow, centered on the init pose
        #      (as specified).  NB shoulder_lift init (-104.13 deg) is OUTSIDE the XML
        #      joint range [-100, +100], so half of this command is into the hard stop.
        qrec = {m: np.zeros(N_SINE) for m in SINE_JOINTS}
        cmd = {m: np.zeros(N_SINE) for m in SINE_JOINTS}
        for i in range(N_SINE):
            tgt = dict(self.init_deg)
            for m in SINE_JOINTS:
                tgt[m] = self.init_deg[m] + SINE_AMP_DEG * np.sin(2 * np.pi * SINE_HZ * i / CTRL_HZ)
                cmd[m][i] = tgt[m]
            step(tgt)
            for m in SINE_JOINTS:
                qrec[m][i] = np.rad2deg(float(self.data.qpos[self.qadr[m]]))
        track = self._sine_stats(qrec, cmd)

        # ---- phase 3 (diagnostic): same sine, re-centred inside the joint ranges so the
        #      realised excursion reflects the gains and not the hard stops.
        centre = {"shoulder_lift": -90.0, "elbow_flex": 85.0}
        qrec3 = {m: np.zeros(N_SINE) for m in SINE_JOINTS}
        cmd3 = {m: np.zeros(N_SINE) for m in SINE_JOINTS}
        for i in range(N_SINE):
            tgt = dict(self.init_deg)
            for m in SINE_JOINTS:
                tgt[m] = centre[m] + SINE_AMP_DEG * np.sin(2 * np.pi * SINE_HZ * i / CTRL_HZ)
                cmd3[m][i] = tgt[m]
            step(tgt)
            for m in SINE_JOINTS:
                qrec3[m][i] = np.rad2deg(float(self.data.qpos[self.qadr[m]]))
        wall = time.perf_counter() - t0
        st["dof_acc"] = dof_max
        track3 = self._sine_stats(qrec3, cmd3)
        return dict(st=st, sag=sag, track=track, track3=track3, wall=wall,
                    sim_s=(N_HOLD + 2 * N_SINE) / CTRL_HZ)

    def _sine_stats(self, qrec, cmd):
        track = {}
        cycle = int(CTRL_HZ / SINE_HZ)          # 1000 control steps per 0.5 Hz cycle
        ss = slice(N_SINE - cycle, N_SINE)      # last full cycle = steady state
        for m in SINE_JOINTS:
            q, c = qrec[m], cmd[m]
            dc = float(np.mean(q[ss]))
            exc = float(np.max(np.abs(q - dc)))
            err = q - c
            track[m] = dict(exc=exc, dc=dc, rms=float(np.sqrt(np.mean(err[ss] ** 2))),
                            mx=float(np.max(np.abs(err[ss]))),
                            span_pp=float(np.max(q[ss]) - np.min(q[ss])))
        return track


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="A", choices=["A", "B", "C", "BASE"])
    ap.add_argument("--kp", default="", help="comma-separated kp grid (overrides default)")
    ap.add_argument("--kv", default="", help="comma-separated kv grid (overrides default)")
    args = ap.parse_args()
    kps = tuple(float(x) for x in args.kp.split(",")) if args.kp else (
        (100.0,) if args.variant == "BASE" else KPS)
    kvs = tuple(float(x) for x in args.kv.split(",")) if args.kv else (
        (20.0,) if args.variant == "BASE" else KVS)

    print("=" * 96)
    print(f"armature-zero gain sweep — variant {args.variant}")
    print("=" * 96)
    r = Runner(args.variant)
    M = arm_mass_diag(r.model, r.data)
    print(f"\ninit pose deg = {r.init_deg}")
    print(f"{'joint':14} {'armature':>9} {'M diag':>11} {'kv<2M/dt':>10} {'kp<4M/dt^2':>12}")
    for m in MOTORS:
        Mi = float(M[r.dof[m], r.dof[m]])
        print(f"{m:14} {r.model.dof_armature[r.dof[m]]:9.4f} {Mi:11.6f} "
              f"{2 * Mi / DT:10.3f} {4 * Mi / DT ** 2:12.1f}")

    rows = []
    for kp in kps:
        for kv in kvs:
            res = r.measure(kp, kv)
            st, sag, tr = res["st"], res["sag"], res["track"]
            unstable = st["nan"] > 0 or st["qacc"] > QACC_UNSTABLE
            pred = []
            for m in MOTORS:
                Mi = float(M[r.dof[m], r.dof[m]])
                if kv * DT / Mi > 2 or kp * DT ** 2 / Mi > 4:
                    pred.append(m)
            rows.append((kp, kv, unstable, st, sag, tr, pred, res))
            print(f"\nkp={kp:<7g} kv={kv:<5g} {'UNSTABLE' if unstable else 'stable  '} "
                  f"max|qacc|={st['qacc']:.1f} max|qvel|={st['qvel']:.2f} nan={st['nan']} "
                  f"sat_steps={st['sat']} realtime={res['sim_s'] / res['wall']:.2f}x")
            print(f"   sag deg      : " + "  ".join(f"{m}={sag[m]:+.3f}" for m in MOTORS))
            for m in SINE_JOINTS:
                t = tr[m]
                print(f"   {m:13}: realised peak {t['exc']:.3f} deg (cmd {SINE_AMP_DEG}), "
                      f"span_pp={t['span_pp']:.3f}, DC={t['dc']:+.3f}, "
                      f"ss err rms={t['rms']:.3f} max={t['mx']:.3f} deg")
            t3 = res["track3"]
            for m in SINE_JOINTS:
                t = t3[m]
                print(f"   {m:13} [in-range centre]: realised peak {t['exc']:.3f} deg, "
                      f"span_pp={t['span_pp']:.3f}, DC={t['dc']:+.3f}, "
                      f"ss err rms={t['rms']:.3f} max={t['mx']:.3f} deg")
            if pred:
                print(f"   predicted-unstable by kv*dt/M / kp*dt^2/M on: {pred}")
            print("   per-dof max|qacc| : " + "  ".join(
                f"{m}={st['dof_acc'].get(m, 0.0):.3g}" for m in MOTORS))

    print("\n" + "=" * 110)
    print(f"{'kp':>7} {'kv':>5} {'stable':>7} {'max|qacc|':>11} {'max|qvel|':>10} "
          f"{'exc_lift':>9} {'exc_elb':>8} {'err_lift':>9} {'err_elb':>8} "
          f"{'exc3_lift':>10} {'exc3_elb':>9} {'err3_lift':>10} {'err3_elb':>9} "
          f"{'sag_lift':>9} {'sag_elb':>8} {'rt':>6}")
    print("-" * 110)
    for kp, kv, un, st, sag, tr, pred, res in rows:
        t3 = res["track3"]
        print(f"{kp:>7g} {kv:>5g} {'NO' if un else 'yes':>7} {st['qacc']:>11.1f} "
              f"{st['qvel']:>10.2f} {tr['shoulder_lift']['exc']:>9.3f} "
              f"{tr['elbow_flex']['exc']:>8.3f} {tr['shoulder_lift']['rms']:>9.3f} "
              f"{tr['elbow_flex']['rms']:>9.3f} {t3['shoulder_lift']['exc']:>10.3f} "
              f"{t3['elbow_flex']['exc']:>9.3f} {t3['shoulder_lift']['rms']:>10.3f} "
              f"{t3['elbow_flex']['rms']:>9.3f} {sag['shoulder_lift']:>9.3f} "
              f"{sag['elbow_flex']:>8.3f} {res['sim_s'] / res['wall']:>6.2f}")
    r.bot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

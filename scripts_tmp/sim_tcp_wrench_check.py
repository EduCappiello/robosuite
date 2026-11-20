"""Problem 6: does the sim publish the wrist wrench in the frame the HARDWARE publishes it in?

Hardware (TCPWrenchProcessorStep) has no force sensor. It estimates the wrench from tau_ext
through the Pinocchio frame Jacobian at gripper_frame_link with LOCAL_WORLD_ALIGNED, which
means: components WORLD-ALIGNED, moment taken about the TCP (fingertip).

The raw MuJoCo <force site=wrist_ft_site> reading is neither: it is in the SITE frame
(body-fixed wrist_roll axes) with the moment taken at the wrist_roll origin.

Payload sweep: a known mass must read 9.810 N/kg in the world z component of a correctly
framed wrench.
"""
import numpy as np
import mujoco
from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower

G = 9.80665
M = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    (sl, ic), *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1.0 - float(np.sum((y - (sl * x + ic)) ** 2)) / max(float(np.sum((y - y.mean()) ** 2)), 1e-12)
    return float(sl), float(r2)


bot = RobosuiteSimFollower(RobosuiteSimConfig(
    id="p6", control_freq=500, dynamics_robot_id=None, cameras=()))
bot.connect()
m, d = bot._model, bot._data

jaw = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot0_moving_jaw")
ft_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "robot0_wrist_ft_site")
tcp_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "robot0_grip_site")
base_mass = float(m.body_mass[jaw])
home = {x: float(bot.get_observation()[f"{x}.pos"]) for x in M}
home["shoulder_lift"], home["elbow_flex"] = -60.0, -40.0


def hold(n=250):
    for _ in range(n):
        bot.send_action({f"{x}.pos": home[x] for x in M})


hold(600)

rows = []
for grams in (0, 50, 100, 200):
    m.body_mass[jaw] = base_mass + grams / 1000.0
    mujoco.mj_forward(m, d)
    hold(700)
    o = bot.get_observation()
    f_site = np.array([o["ground_truth.ext.tcp_wrench_ft." + a] for a in ("fx", "fy", "fz")])
    t_site = np.array([o["ground_truth.ext.tcp_wrench_ft." + a] for a in ("tx", "ty", "tz")])
    R = np.asarray(d.site_xmat, float).reshape(-1, 3, 3)[ft_site]
    f_w = R @ f_site
    t_w = R @ t_site
    p = np.asarray(d.site_xpos, float).reshape(-1, 3)
    r = p[ft_site] - p[tcp_site]
    t_tcp_expected = t_w + np.cross(r, f_w)
    rows.append({
        "kg": grams / 1000.0,
        "fz_site": float(f_site[2]),
        "fz_world": float(f_w[2]),
        "fz_pub": float(o["tcp/wrench/fz"]),
        "tau_site_norm": float(np.linalg.norm(t_site)),
        "tau_tcp_norm": float(np.linalg.norm([o["tcp/wrench/" + a] for a in ("tx", "ty", "tz")])),
        "tau_tcp_expected": t_tcp_expected,
        "tau_tcp_pub": np.array([o["tcp/wrench/" + a] for a in ("tx", "ty", "tz")]),
        "lever_m": float(np.linalg.norm(r)),
    })
m.body_mass[jaw] = base_mass

kg = [r["kg"] for r in rows]
print(f"{'quantity':34} {'slope N/kg':>11} {'R^2':>8}   expected")
for key, exp in (("fz_site", "NOT 9.81 -- body-fixed axis"),
                 ("fz_world", "9.807"),
                 ("fz_pub", "9.807 (published key)")):
    sl, r2 = slope(kg, [r[key] for r in rows])
    print(f"{key:34} {sl:+11.4f} {r2:8.4f}   {exp}")

print(f"\nlever arm FT site -> TCP: {rows[0]['lever_m'] * 1000:.1f} mm")
print(f"{'kg':>5} {'|tau| at site':>14} {'|tau| at TCP':>13} {'transport err':>14}")
for r in rows:
    err = float(np.max(np.abs(r["tau_tcp_expected"] - r["tau_tcp_pub"])))
    print(f"{r['kg']:5.2f} {r['tau_site_norm']:14.4f} {r['tau_tcp_norm']:13.4f} {err:14.2e}")

g_sim = float(-m.opt.gravity[2])
sl_fz, r2_fz = slope(kg, [r["fz_pub"] for r in rows])
intercept = float(np.mean([r["fz_pub"] for r in rows]) - sl_fz * np.mean(kg))
ok_fz = abs(sl_fz - g_sim) < 0.02
ok_tr = max(float(np.max(np.abs(r["tau_tcp_expected"] - r["tau_tcp_pub"]))) for r in rows) < 1e-9
ok_grow = rows[-1]["tau_tcp_norm"] - rows[0]["tau_tcp_norm"] > 0.02 * 9.8 * 0.098
print("\nCHECKS")
print(f"  published fz slope == g ({g_sim:.4f}) : {'PASS' if ok_fz else 'FAIL'}  slope={sl_fz:+.4f} R2={r2_fz:.5f} intercept={intercept:+.4f} N")
print(f"  moment transport identity exact  : {'PASS' if ok_tr else 'FAIL'}")
print(f"  |tau| at TCP grows by m*g*lever        : {'PASS' if ok_grow else 'FAIL'}  d|tau|={rows[-1]["tau_tcp_norm"] - rows[0]["tau_tcp_norm"]:.4f} N*m, m*g*lever={0.2*9.80665*0.0984:.4f} (jaw CoM is NOT at the TCP, so no cancellation is expected)")
bot.disconnect()
raise SystemExit(0 if (ok_fz and ok_tr and ok_grow) else 1)

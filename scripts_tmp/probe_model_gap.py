"""Are the sim's dynamics and the real fitted model materially different?
Both publish their own gravity/friction terms on the same sample, so compare them directly:
  sim : ground_truth.model.{m}.tau_gravity / .tau_friction
  EKF : {m}.model/tau_gravity            / .model/tau_friction   (fitted on the real arm)
Same q, same qdot, one sample, two models."""
import numpy as np, mujoco
from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower
from lerobot.scripts.lerobot_teleoperate import make_default_processors
from lerobot.robots.so_follower.dynamics.torque_estimator_processor import make_so_torque_pipeline_steps
M=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]; ARM=M[:5]
b=RobosuiteSimFollower(RobosuiteSimConfig(id="s",control_freq=500,dynamics_robot_id="FT-02",
    emit_ground_truth=True))
b.connect()
pipe=make_default_processors()[2]
pipe.steps=list(pipe.steps)+make_so_torque_pipeline_steps(motor_names=M,robot_type="so101_follower",
    robot_id="FT-02",estimation_method="ekf")
o=b.get_observation(); home={x:float(o[f"{x}.pos"]) for x in M}
POSES=[]
for lift in (-30.,-60.,-90.):
    for elbow in (-20.,-50.,-80.):
        p=dict(home); p["shoulder_lift"]=lift; p["elbow_flex"]=elbow; POSES.append(p)
SG={m:[] for m in ARM}; EG={m:[] for m in ARM}; SF={m:[] for m in ARM}; EF={m:[] for m in ARM}
for p in POSES:
    for _ in range(300): b.send_action({f"{x}.pos":p[x] for x in M})
    for _ in range(60):
        o=b.get_observation()
        for m in ARM:
            SG[m].append(float(o[f"ground_truth.model.{m}.tau_gravity"]))
            SF[m].append(float(o[f"ground_truth.model.{m}.tau_friction"]))
        obs={k:v for k,v in o.items() if not k.startswith("ground_truth")}
        eo=pipe(dict(obs))
        for m in ARM:
            EG[m].append(float(eo[f"{m}.model/tau_gravity"]))
            EF[m].append(float(eo[f"{m}.model/tau_friction"]))
b.disconnect()
print(f"{'joint':15} {'G_sim':>8} {'G_ekf':>8} {'dG':>8} {'rel%':>7} | {'F_sim':>8} {'F_ekf':>8} {'dF':>8}")
for m in ARM:
    gs,ge=np.mean(SG[m]),np.mean(EG[m]); fs,fe=np.mean(SF[m]),np.mean(EF[m])
    rel=100*(gs-ge)/max(abs(ge),1e-9)
    print(f"{m:15} {gs:+8.3f} {ge:+8.3f} {gs-ge:+8.3f} {rel:+7.1f} | {fs:+8.3f} {fe:+8.3f} {fs-fe:+8.3f}")

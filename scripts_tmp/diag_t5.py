import numpy as np, mujoco
from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower
M=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
b=RobosuiteSimFollower(RobosuiteSimConfig(id="s",control_freq=500,dynamics_robot_id="FT-02",emit_ground_truth=False))
b.connect(); m,d=b._model,b._data
o=b.get_observation(); init={k:float(o[f"{k}.pos"]) for k in M}
print("init deg", {k:round(v,1) for k,v in init.items()})
print("joint ranges rad:", {k:tuple(np.round(m.jnt_range[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_{k}")],3)) for k in M})
for i in range(200): b.send_action({f"{k}.pos":init[k] for k in M})
print("after settle ncon",d.ncon,"qvelmax",float(np.max(np.abs(d.qvel))))
for i in range(1000):
    b.send_action({f"{k}.pos":init[k]+(10*np.sin(2*np.pi*0.5*i/500) if k in("shoulder_lift","elbow_flex") else 0) for k in M})
qa=int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,"robot0_shoulder_lift")])
print("lift q deg now", round(float(np.rad2deg(d.qpos[qa])),2), "ncon",d.ncon)
seen={}
for c in range(d.ncon):
    g1,g2=int(d.contact[c].geom1),int(d.contact[c].geom2)
    n1=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g1) or str(g1); n2=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g2) or str(g2)
    seen[(n1,n2)]=round(float(d.contact[c].dist),4)
for k,v in list(seen.items())[:20]: print("  contact",k,"dist",v)
b.disconnect()

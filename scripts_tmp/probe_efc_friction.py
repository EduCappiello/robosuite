"""Where does MuJoCo put the friction force it actually applied?
The gt term -(sign(v)*fl + damp*v) is 0 at rest, so it hides static friction the solver IS
applying. Inspect the constraint array to find the rows that carry it."""
import numpy as np, mujoco
from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower
b=RobosuiteSimFollower(RobosuiteSimConfig(id="s",control_freq=500,dynamics_robot_id="FT-02"))
b.connect(); m,d=b._model,b._data
lift=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,"robot0_shoulder_lift")
qa,da=int(m.jnt_qposadr[lift]),int(m.jnt_dofadr[lift])
names=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
for _ in range(1000):
    b.send_action({f"{x}.pos":v for x,v in zip(names,[0.,-60.,-40.,20.,0.,0.])})
q_ref=d.qpos.copy()
m.dof_frictionloss[da]=1.0; m.dof_damping[da]=0.0
d.qpos[:]=q_ref; d.qvel[:]=0.; d.ctrl[:]=0.; d.qfrc_applied[:]=0.
for _ in range(200): mujoco.mj_step(m,d)
print("nefc", int(d.nefc))
types=np.array([d.efc_type[i] for i in range(int(d.nefc))])
tn={getattr(mujoco,n):n for n in dir(mujoco) if n.startswith("mjCON_")}
for t in sorted(set(types.tolist())):
    idx=[i for i in range(int(d.nefc)) if d.efc_type[i]==t]
    print(f"  type {tn.get(t,t):16} n={len(idx):3d} "
          f"force[:3]={np.round([float(d.efc_force[i]) for i in idx[:3]],4)}")
idx1=[i for i in range(int(d.nefc)) if d.efc_type[i]==1]
print("type-1 rows: efc_id =", [int(d.efc_id[i]) for i in idx1],
      " arm dof addrs =", [int(m.jnt_dofadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'robot0_{x}')]) for x in names])
print("type-1 efc_frictionloss =", [round(float(d.efc_frictionloss[i]),4) for i in idx1])
print("\nqfrc_constraint[lift] =", round(float(d.qfrc_constraint[da]),4),
      " gravity =", round(float(-d.qfrc_bias[da]),4),
      " analytic gt friction =", round(float(-(np.sign(d.qvel[da])*1.0)),4))
b.disconnect()

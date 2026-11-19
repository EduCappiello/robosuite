"""Adopt armature=0 on the 5 arm joints (gripper keeps 0.028). Compare against baseline on
stability, sine tracking, gravity sag and the 0/50/100/200 g payload slope."""
import numpy as np, mujoco
from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower
MOTORS=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
ARM=MOTORS[:5]
def run(zero_arm):
    cfg=RobosuiteSimConfig(id="sim",control_freq=500,kp=100,kv=20.0,dynamics_robot_id="FT-02",emit_ground_truth=True)
    b=RobosuiteSimFollower(cfg); b.connect(); m,d=b._model,b._data
    dof={k:int(m.jnt_dofadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_{k}")]) for k in MOTORS}
    if zero_arm:
        for k in ARM: m.dof_armature[dof[k]]=0.0
    mujoco.mj_forward(m,d)
    o=b.get_observation(); init={k:float(o[f"{k}.pos"]) for k in MOTORS}
    def step(tgt):
        b.send_action({f"{k}.pos":tgt[k] for k in MOTORS})
    mx=0.;nan=0
    for i in range(2000): step(init)
    for i in range(2000):
        step({k:init[k]+(10*np.sin(2*np.pi*0.5*i/500) if k in("shoulder_lift","elbow_flex") else 0) for k in MOTORS})
        if not np.isfinite(d.qacc).all(): nan+=1
        else: mx=max(mx,float(np.max(np.abs(d.qacc))))
    exc={}
    for k in("shoulder_lift","elbow_flex"):
        qa=int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_{k}")])
        hist=[]
        for i in range(1000):
            step({j:init[j]+(10*np.sin(2*np.pi*0.5*i/500) if j in("shoulder_lift","elbow_flex") else 0) for j in MOTORS})
            hist.append(np.rad2deg(d.qpos[qa]))
        exc[k]=float(np.max(hist)-np.min(hist))/2
    jaw=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"robot0_moving_jaw"); base=float(m.body_mass[jaw])
    fz=[]
    for g in (0,50,100,200):
        m.body_mass[jaw]=base+g/1000.; mujoco.mj_forward(m,d)
        for _ in range(250): step(init)
        o=b.get_observation()
        v=np.array([o[f"ground_truth.ext.tcp_wrench_ft.{a}"] for a in ("fx","fy","fz")])
        smat=np.asarray(d.site_xmat).reshape(-1,3,3)[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,"robot0_wrist_ft_site")]
        fz.append(float(np.linalg.norm(smat@v)))
    fz=np.array(fz); ms=np.array([0,.05,.1,.2])
    sl=float(np.polyfit(ms[1:],fz[1:]-fz[0],1)[0])
    print(f"arm5zero={zero_arm}: max|qacc|={mx:9.1f} nan={nan} halfspan_deg={ {k:round(v,2) for k,v in exc.items()} } "
          f"payload slope={sl:.3f} N/kg (g=9.807) empty|F|={fz[0]:.3f} N")
    b.disconnect()
run(False); run(True)

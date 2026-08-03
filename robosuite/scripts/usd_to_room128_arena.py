"""Generate robosuite's room128_arena.xml from classroom128_v2.usd.

The USD is Z-up in metres, exactly like MuJoCo, so geometry maps 1:1; the only
transform is a translation that re-centres the room on the world origin (the USD
puts its origin in a corner, while robosuite tasks build their furniture around
0,0). Run with the scratchpad usdvenv, which has usd-core.
"""
from pxr import Gf, Usd, UsdGeom

USD = "/home/ecappiell/Documents/dev/robosuite-xlerobot/robosuite/environments/Room_128/classroom128_v2.usd"
OUT = "/home/ecappiell/Documents/dev/robosuite-xlerobot/robosuite/models/assets/arenas/room128_arena.xml"

stage = Usd.Stage.Open(USD)
mpu = UsdGeom.GetStageMetersPerUnit(stage)
xc = UsdGeom.XformCache(Usd.TimeCode.Default())

boxes = []
for pr in stage.TraverseAll():
    if pr.GetTypeName() != "Cube":
        continue
    size = UsdGeom.Cube(pr).GetSizeAttr().Get()
    M = xc.GetLocalToWorldTransform(pr)
    t = M.ExtractTranslation()
    s = [Gf.Vec3d(M[i][0], M[i][1], M[i][2]).GetLength() for i in range(3)]
    name = str(pr.GetPath()).split("/")[3].lower()   # Floor / WallFront / door / ...
    boxes.append({
        "name": name,
        "pos": [t[i] * mpu for i in range(3)],
        "half": [size / 2.0 * s[i] * mpu for i in range(3)],
    })

floor = next(b for b in boxes if b["name"] == "floor")
cx, cy = floor["pos"][0], floor["pos"][1]          # re-centre on the floor slab
inner_x, inner_y = floor["half"][0], floor["half"][1]

FURNITURE = {"door", "cabinet", "storage_cabinet", "desk", "desk2", "pillar1", "pillar2"}
lines = []
for b in boxes:
    if b["name"] == "floor":
        continue
    p = [b["pos"][0] - cx, b["pos"][1] - cy, b["pos"][2]]
    h = b["half"]
    mat = "walls_mat" if b["name"].startswith("wall") else "furniture_mat"
    lines.append(
        f'    <geom name="room128_{b["name"]}" type="box" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" '
        f'size="{h[0]:.4f} {h[1]:.4f} {h[2]:.4f}" group="1" material="{mat}" '
        f'condim="3" friction="1 0.005 0.0001"/>'
    )

xml = f"""<mujoco model="room128_arena">
  <!-- Room 128 (classroom128_v2.usd), the room the real coffee dataset was recorded
       in: IntelligentDecisionLab/xlerobot-coffee-real, all episodes under room-128/.

       GENERATED - do not hand-edit. Source of truth is
       robosuite/environments/Room_128/classroom128_v2.usd; the USD is Z-up in metres
       (same conventions as MuJoCo) and built entirely from Cube prims, so every box
       below is a 1:1 copy. The only change is a translation of ({-cx:.4f}, {-cy:.4f})
       that re-centres the room on the origin, because robosuite tasks build their
       furniture around (0,0) while the USD keeps its origin in a corner.

       Inner floor is {2*inner_x:.3f} x {2*inner_y:.3f} m, walls {boxes[1]['half'][2]*2:.2f} m tall.
       Walls and furniture COLLIDE (this arena exists for the t4_navigate task, where
       the cart has to drive around them) - unlike robosuite's stock arenas, whose
       walls are visual-only. -->
  <asset>
    <texture builtin="gradient" height="256" rgb1=".9 .9 1." rgb2=".2 .3 .4" type="skybox" width="256"/>
    <texture file="../textures/light-gray-floor-tile.png" type="2d" name="texplane"/>
    <material name="floorplane" reflectance="0.01" shininess="0.0" specular="0.0" texrepeat="8 8" texture="texplane" texuniform="true"/>
    <texture file="../textures/light-gray-plaster.png" type="2d" name="tex-light-gray-plaster"/>
    <material name="walls_mat" reflectance="0.0" shininess="0.1" specular="0.1" texrepeat="6 6" texture="tex-light-gray-plaster" texuniform="true"/>
    <material name="furniture_mat" reflectance="0.0" shininess="0.2" specular="0.2" rgba="0.55 0.45 0.35 1"/>
  </asset>
  <worldbody>
    <geom condim="3" group="1" material="floorplane" name="floor" pos="0 0 0" size="{inner_x:.4f} {inner_y:.4f} .125" type="plane"/>
{chr(10).join(lines)}
    <light pos="0 0 2.6" dir="0 0 -1" specular="0.2 0.2 0.2" directional="true" castshadow="false"/>
    <light pos="1.2 1.2 2.4" dir="-0.3 -0.3 -1" specular="0.1 0.1 0.1" directional="true" castshadow="false"/>
    <camera mode="fixed" name="frontview" pos="1.6 0 1.45" quat="0.56 0.43 0.43 0.56"/>
    <camera mode="fixed" name="birdview" pos="0 0 6.2" quat="0.7071 0 0 0.7071"/>
    <camera mode="fixed" name="agentview" pos="0.5 0 1.35" quat="0.653 0.271 0.271 0.653"/>
    <camera mode="fixed" name="sideview" pos="-0.0565 1.2761 1.4880" quat="0.0099 0.0069 0.5912 0.8064"/>
    <camera mode="fixed" name="roomview" pos="-1.55 -2.15 2.35" quat="0.780 0.386 0.190 0.455"/>
  </worldbody>
</mujoco>
"""
with open(OUT, "w") as f:
    f.write(xml)
print(f"wrote {OUT}")
print(f"re-centre offset: ({-cx:.4f}, {-cy:.4f})   inner floor {2*inner_x:.3f} x {2*inner_y:.3f} m")
for b in boxes:
    if b["name"] in FURNITURE:
        print(f"   {b['name']:18s} centred pos=({b['pos'][0]-cx:+.3f},{b['pos'][1]-cy:+.3f},{b['pos'][2]:.3f})")

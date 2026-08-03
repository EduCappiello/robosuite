"""Emit the MJCF for the printed foam cup holder, from the measured dimensions.

Prints geoms to stdout; paste them into the cup-holder block in
robosuite/models/assets/robots/XLeRobot/robot.xml. Edit the constants here rather
than the XML, so the block stays traceable to the measurements.

Measured (cm):  L 29.5  W 18  thickness 7   hole diameter 7
                between holes 2, hole-to-short-end 2.25, gaps 3 (right) / 6.5 (left)

MuJoCo has no CSG, so the slab is built as a lattice of boxes around square
apertures, then each aperture's four corners are chamfered at 45 deg. The result is
a regular octagon inscribing the 7 cm circle - round enough at this scale to read
correctly and to retain a cup, without 6x16 ring segments.
"""
import numpy as np

CM = 0.01
L, W, T = 29.5 * CM, 18 * CM, 7 * CM        # length (y), width (x), thickness (z)
D = 7 * CM                                   # hole diameter
R = D / 2
PITCH = 9 * CM                               # hole centre spacing (7 + 2 gap)

# Placement in the cart's base frame -------------------------------------------
# Top basket floor, found by raycasting down inside the basket: 0.688 (the three
# floors are 0.688 / 0.408 / 0.138). The block rests on it, top at 0.758, ~1.7 cm
# below the cart rim -- exactly as in the reference photo.
FLOOR_Z = 0.688
# Interior at this height (raycast outward from the basket centre):
#   x -0.1529..+0.1496 (30.25 cm),  y -0.1958..+0.1966 (39.24 cm)
# The measured gaps 3 + 29.5 + 6.5 = 39 cm match that y interior to 2 mm, which
# confirms they were taken at the foam's own height.
# x: the arm mounting plate (topbase1) ends at x=-0.03, leaving 17.96 cm to the
# front wall -- which is why the block is 18 cm wide. It fills that gap.
CX = (-0.030 + 0.1496) / 2
# y: anchor the measured 3 cm gap on the right (-y) wall.
CY = -0.1958 + 3.0 * CM + L / 2
CZ = FLOOR_Z + T / 2

# One slot is a CUP PLACE: blind, only 3 cm deep, so a cup sits proud of the foam
# instead of dropping to the basket floor. Indices into (hole_x, hole_y) below.
# (1, 0) = forward (+x) and right (-y) -> the "top right" slot in the head-cam view,
# whose image right is world -y and image up is world +x.
CUP_PLACE = (1, 0)
CUP_PLACE_DEPTH = 3 * CM

hx, hy, hz = W / 2, L / 2, T / 2
hole_x = [-PITCH / 2, PITCH / 2]                 # 2 across the width
hole_y = [-PITCH, 0.0, PITCH]                    # 3 along the length

geoms = []


def box(name, cx, cy, cz, sx, sy, sz, quat=None):
    q = f' quat="{quat}"' if quat else ""
    geoms.append(
        f'      <geom name="{name}" type="box" pos="{cx:.5f} {cy:.5f} {cz:.5f}" '
        f'size="{sx:.5f} {sy:.5f} {sz:.5f}"{q} group="1" material="cup_foam" '
        f'friction="1.2 0.02 0.001"/>'
    )


# --- lattice: material strips between/around the apertures ---------------------
# apertures span +/-R about each hole centre
x_edges = [(-hx, hole_x[0] - R), (hole_x[0] + R, hole_x[1] - R), (hole_x[1] + R, hx)]
y_edges = [(-hy, hole_y[0] - R), (hole_y[0] + R, hole_y[1] - R),
           (hole_y[1] + R, hole_y[2] - R), (hole_y[2] + R, hy)]

for i, (a, b) in enumerate(x_edges):                     # full-length strips
    box(f"cup_foam_xrib{i}", CX + (a + b) / 2, CY, CZ, (b - a) / 2, hy, hz)
for j, (a, b) in enumerate(y_edges):                     # cross strips, between ribs
    for i in (0, 1):
        lo, hi = hole_x[i] - R, hole_x[i] + R
        box(f"cup_foam_yrib{j}_{i}", CX + (lo + hi) / 2, CY + (a + b) / 2, CZ,
            (hi - lo) / 2, (b - a) / 2, hz)

# --- chamfers: octagonalise each square aperture -------------------------------
# Chamfer plane sits tangent to the circle at 45 deg; it spans from where that plane
# meets one square side to the other, i.e. half-length (R - (sqrt(2)-1)*R)... derived
# numerically below so the octagon inscribes the circle exactly.
diag = R * np.sqrt(2)                       # corner distance
t_half = (diag - R) / 2                     # chamfer thickness
c_off = R + t_half                          # centre distance along the diagonal
meet = R * (np.sqrt(2) - 1)                 # where the chamfer plane meets a side
w_half = np.hypot(R - meet, R - meet) / 2   # chamfer half-width

for hi_, hxc in enumerate(hole_x):
    for hj, hyc in enumerate(hole_y):
        for k, ang in enumerate((45, 135, 225, 315)):
            a = np.radians(ang)
            px = CX + hxc + c_off * np.cos(a)
            py = CY + hyc + c_off * np.sin(a)
            qw, qz = np.cos(a / 2), np.sin(a / 2)
            box(f"cup_foam_ch{hi_}{hj}{k}", px, py, CZ, t_half, w_half, hz,
                quat=f"{qw:.6f} 0 0 {qz:.6f}")

# --- blind bottom for the cup-place slot ---------------------------------------
# Fills the aperture from the block's underside up to (top - 3 cm), leaving a 3 cm
# recess. Square footprint: it only needs to plug the octagon, and the overlap with
# the chamfers is interior material either way.
_px, _py = hole_x[CUP_PLACE[0]], hole_y[CUP_PLACE[1]]
_plug_t = T - CUP_PLACE_DEPTH                     # 4 cm of foam beneath the recess
box("cup_foam_place_bottom", CX + _px, CY + _py, CZ - hz + _plug_t / 2,
    R, R, _plug_t / 2)

print(f"<!-- cup-place slot at ({CX+_px:.4f}, {CY+_py:.4f}), recess {CUP_PLACE_DEPTH*100:.0f} cm, "
      f"floor z={FLOOR_Z + _plug_t:.4f} -->")
print(f"<!-- block {L*100:.1f} x {W*100:.1f} x {T*100:.1f} cm, 6 holes d={D*100:.0f} cm -->")
print(f"<!-- centre ({CX:.4f}, {CY:.4f}, {CZ:.4f})  z {FLOOR_Z:.3f}..{FLOOR_Z+T:.3f} -->")
print("\n".join(geoms))
print(f"\n<!-- {len(geoms)} geoms -->", )

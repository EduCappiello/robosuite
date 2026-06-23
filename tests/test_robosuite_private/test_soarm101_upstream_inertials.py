"""
Drift guard: the SOARM101 model robosuite actually loads must stay faithful to the
authoritative upstream physics (TheRobotStudio/SO-ARM100, pinned in UPSTREAM.json).

WHY THIS EXISTS
---------------
robosuite cannot consume the upstream monolithic MJCF directly — it hand-authors an
arm/gripper split (it needs <motor> actuators, eef sites, a wrist FT sensor, and a
separate add_gripper body). That split is the exact place a physics regression slips in
unnoticed: it once inserted a placeholder ``gripper_base`` with a fabricated mass="3e-1"
(+0.3 kg, +47.5% of the real 0.632 kg arm) at the wrist, silently corrupting every
gravity / inertial ground-truth label the sim emits. Pinocchio (the real-robot estimator's
URDF) was correct the whole time; only this sim copy drifted.

This test reads the *compiled* MuJoCo model (what physics actually integrates, post-merge,
robot0_-prefixed) and asserts each body's mass / COM / inertia equals the upstream value
frozen in UPSTREAM.json. Regenerate that manifest with
``python -m robosuite_private.tools.gen_upstream_manifest /path/to/SO-ARM100`` when
intentionally bumping upstream — a physics change then shows up as a manifest diff with a
human in the loop, instead of as a phantom torque months later.

Reads only upstream-derived data + robosuite assets; imports no lerobot* (boundary rule).
"""

import hashlib
import json
import os

import numpy as np
import pytest

import mujoco
import robosuite as suite

_ASSET_ROOT = os.path.join(
    os.path.dirname(suite.__file__), "models", "assets", "robots", "SOARM101"
)
_MANIFEST_PATH = os.path.join(_ASSET_ROOT, "UPSTREAM.json")
_MESH_DIR = os.path.join(_ASSET_ROOT, "SO101", "assets")

# Tolerances. Masses/COMs are specified to ~6 significant figures upstream; inertias are
# O(1e-5) so a relative tol is the meaningful one, with a tiny absolute floor.
_MASS_ATOL = 1e-6      # kg
_COM_ATOL = 1e-6       # m
_INERTIA_RTOL = 1e-3
_INERTIA_ATOL = 1e-9   # kg·m²


@pytest.fixture(scope="module")
def manifest():
    if not os.path.exists(_MANIFEST_PATH):
        pytest.skip(f"provenance manifest missing: {_MANIFEST_PATH}")
    with open(_MANIFEST_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def compiled():
    """The assembled, compiled MuJoCo model + the robot's body-name prefix."""
    env = suite.make(
        env_name="Lift",
        robots=["SOARM101"],
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    env.reset()
    robot = env.robots[0]
    prefix = getattr(robot.robot_model, "naming_prefix", "robot0_")
    model = env.sim.model._model
    yield model, prefix
    env.close()


def _body(model, prefix, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + name)
    assert bid != -1, f"compiled model has no body {prefix + name!r}"
    return bid


def _full_inertia_body_frame(model, bid):
    """Reconstruct the inertia tensor (about COM, body-frame axes) from the compiled
    model's principal moments + principal-frame quaternion."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, model.body_iquat[bid])
    R = R.reshape(3, 3)
    Ip = np.diag(model.body_inertia[bid])
    return R @ Ip @ R.T


def test_manifest_internally_consistent(manifest):
    """total_robot_mass_kg equals the sum of per-body masses it lists."""
    s = sum(b["mass"] for b in manifest["bodies"].values())
    assert s == pytest.approx(manifest["total_robot_mass_kg"], abs=1e-9)


def test_body_masses_match_upstream(manifest, compiled):
    model, prefix = compiled
    bad = []
    for name, ref in manifest["bodies"].items():
        bid = _body(model, prefix, name)
        got = float(model.body_mass[bid])
        if abs(got - ref["mass"]) > _MASS_ATOL:
            bad.append(f"{name}: got {got:.6g} kg, upstream {ref['mass']:.6g} kg")
    assert not bad, "mass drift vs upstream:\n  " + "\n  ".join(bad)


def test_body_coms_match_upstream(manifest, compiled):
    model, prefix = compiled
    bad = []
    for name, ref in manifest["bodies"].items():
        bid = _body(model, prefix, name)
        got = np.asarray(model.body_ipos[bid])
        ref_com = np.asarray(ref["com"])
        if not np.allclose(got, ref_com, atol=_COM_ATOL):
            bad.append(f"{name}: got {got}, upstream {ref_com}")
    assert not bad, "COM drift vs upstream:\n  " + "\n  ".join(bad)


def test_body_inertias_match_upstream(manifest, compiled):
    model, prefix = compiled
    bad = []
    for name, ref in manifest["bodies"].items():
        bid = _body(model, prefix, name)
        I = _full_inertia_body_frame(model, bid)
        # manifest order: Ixx Iyy Izz Ixy Ixz Iyz
        got = np.array([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]])
        want = np.asarray(ref["fullinertia"])
        if not np.allclose(got, want, rtol=_INERTIA_RTOL, atol=_INERTIA_ATOL):
            bad.append(f"{name}:\n    got  {got}\n    want {want}")
    assert not bad, "inertia drift vs upstream:\n  " + "\n  ".join(bad)


def test_gripper_base_is_massless_mounting_frame(compiled):
    """gripper_base is a robosuite-only attachment frame with NO upstream counterpart;
    it must carry effectively zero mass so it injects no gravity/inertia. Guards against
    re-introducing the phantom-mass placeholder."""
    model, prefix = compiled
    bid = _body(model, prefix, "gripper_base")
    m = float(model.body_mass[bid])
    assert m < 1e-6, (
        f"gripper_base mass = {m:.6g} kg — must be a massless mounting frame. The real "
        f"gripper mass lives in gripper(0.087) + moving_jaw(0.012). A nonzero value here "
        f"is the phantom-mass regression (see this file's docstring)."
    )


def test_total_robot_mass(manifest, compiled):
    """Sum of all real robot bodies (incl. fixed base, excl. the massless mounting frame)
    equals the upstream total. The phantom-mass bug made this 0.932 kg."""
    model, prefix = compiled
    total = sum(
        float(model.body_mass[_body(model, prefix, name)])
        for name in manifest["bodies"]
    )
    assert total == pytest.approx(manifest["total_robot_mass_kg"], abs=1e-5)


def test_mesh_md5_matches_upstream(manifest):
    """robosuite's vendored STL bytes are identical to the pinned upstream meshes."""
    bad = []
    for name, up_md5 in manifest["meshes"].items():
        p = os.path.join(_MESH_DIR, name)
        if not os.path.exists(p):
            bad.append(f"{name}: MISSING in {_MESH_DIR}")
            continue
        with open(p, "rb") as f:
            got = hashlib.md5(f.read()).hexdigest()
        if got != up_md5:
            bad.append(f"{name}: got {got}, upstream {up_md5}")
    assert not bad, "mesh drift vs upstream:\n  " + "\n  ".join(bad)

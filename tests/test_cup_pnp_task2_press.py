"""Tests for the cupPnP_task2 press-mode knobs (2026-11-19).

Covers: legacy touch behaviour unchanged, press mode requires travel, button force-travel
matches the configured stiffness/preload, the click detent drops past the click point, and
the wrong-button rule still fires in press mode.
"""

import numpy as np
import pytest

from robosuite.environments.manipulation.cup_pnp_task2 import (
    BREW_BUTTON,
    cupPnP_task2,
)


def _make(**kw):
    kw.setdefault("has_renderer", False)
    kw.setdefault("has_offscreen_renderer", False)
    kw.setdefault("use_camera_obs", False)
    kw.setdefault("ignore_done", True)
    kw.setdefault("control_freq", 20)
    env = cupPnP_task2(**kw)
    env.reset()
    return env


def _set_brew_qpos(env, q):
    """Move the brew slide joint to absolute position q and refresh the state."""
    env.sim.data.set_joint_qpos(f"{env.coffee_machine.naming_prefix}brew_button_joint", q)
    env.sim.forward()


def _brew_dof(env):
    import mujoco

    m = env.sim.model._model
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                          f"{env.coffee_machine.naming_prefix}brew_button_joint")
    return int(m.jnt_dofadr[j])


def test_legacy_touch_mode_is_the_default():
    env = _make()
    assert env.success_mode == "touch"
    # Touching without travel must not set the press flag.
    env._brew_button_hit = True
    env._update_button_outcome()
    assert env.task_success is True
    assert env._pressed is False
    env.close()


def test_press_mode_requires_travel():
    env = _make(success_mode="press", press_travel_m=0.004, press_hold_steps=1)
    # No travel -> not success even if the stylus touched the brew button.
    env._brew_button_hit = True
    env._update_button_outcome()
    assert env._pressed is False
    assert env.task_success is False

    # Full travel -> press registers.
    _set_brew_qpos(env, 0.006)
    env._update_button_outcome()
    assert env._brew_travel_m == pytest.approx(0.006, abs=1e-5)
    assert env._pressed is True
    assert env.task_success is True
    env.close()


def test_press_hold_steps_debounce():
    env = _make(success_mode="press", press_travel_m=0.004, press_hold_steps=3)
    _set_brew_qpos(env, 0.006)
    env._update_button_outcome()   # streak 1
    assert env._press_streak == 1 and env._pressed is False
    env._update_button_outcome()   # streak 2
    assert env._press_streak == 2 and env._pressed is False
    env._update_button_outcome()   # streak 3 -> latched
    assert env._press_streak == 3 and env._pressed is True
    env.close()


def test_button_stiffness_and_preload_set_the_force_travel_curve():
    k, preload = 400.0, 1.0
    env = _make(button_stiffness=k, button_preload_n=preload)
    dof = _brew_dof(env)
    m = env.sim.model._model

    _set_brew_qpos(env, 0.0)
    at_rest = abs(float(env.sim.data.qfrc_spring[dof]))
    _set_brew_qpos(env, 0.006)
    at_full = abs(float(env.sim.data.qfrc_spring[dof]))

    assert float(m.jnt_stiffness[_brew_dof_joint(env)]) == pytest.approx(k)
    # Spring force is k*(q - springref); with a preload the force at q=0 is already `preload`.
    assert at_rest == pytest.approx(preload, rel=1e-3)
    assert at_full == pytest.approx(k * 0.006 + preload, rel=1e-3)
    env.close()


def _brew_dof_joint(env):
    import mujoco

    m = env.sim.model._model
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                             f"{env.coffee_machine.naming_prefix}brew_button_joint")


def test_click_detent_rises_then_drops_at_the_click_point():
    env = _make(click_detent=True, detent_travel_m=0.004, detent_drop_n=1.0)
    dof = _brew_dof(env)
    assert env._detent_dofs, "detent should be registered on this env instance"

    profile = {}
    for q in (0.0, 0.002, 0.0039, 0.0041, 0.006):
        _set_brew_qpos(env, q)
        env._apply_click_detent()
        profile[q] = float(env.sim.data.qfrc_applied[dof])

    assert profile[0.0] == pytest.approx(0.0, abs=1e-9)
    # rising toward the click
    assert abs(profile[0.002]) < abs(profile[0.0039])
    assert abs(profile[0.0039]) == pytest.approx(1.0, rel=0.05)
    # the click: force drops past the click point
    assert abs(profile[0.0041]) < abs(profile[0.0039])
    assert profile[0.006] == pytest.approx(0.0, abs=1e-9)
    env.close()


def test_detent_state_is_per_instance():
    """The libero-force detent used one global registry slot, so parallel envs overwrote
    each other. Two envs must keep independent detent state."""
    a = _make(click_detent=True, detent_drop_n=1.0)
    b = _make(click_detent=True, detent_drop_n=3.0)
    assert a._detent_dofs and b._detent_dofs
    assert a.detent_drop_n == 1.0 and b.detent_drop_n == 3.0
    _set_brew_qpos(a, 0.004)
    a._apply_click_detent()
    _set_brew_qpos(b, 0.004)
    b._apply_click_detent()
    fa = abs(float(a.sim.data.qfrc_applied[_brew_dof(a)]))
    fb = abs(float(b.sim.data.qfrc_applied[_brew_dof(b)]))
    assert fa == pytest.approx(1.0, rel=0.05)
    assert fb == pytest.approx(3.0, rel=0.05)
    a.close()
    b.close()


def test_wrong_button_rule_still_fires_in_press_mode():
    env = _make(success_mode="press", press_travel_m=0.004, press_hold_steps=1)
    _set_brew_qpos(env, 0.006)
    env._wrong_button_hit = True
    env._update_button_outcome()
    assert env.task_failed is True
    assert env.task_success is False
    env.close()


def test_wrist_ft_is_readable_and_travel_observable_exists():
    env = _make()
    w = env.read_wrist_ft()
    assert w["force"].shape == (3,) and w["torque"].shape == (3,)
    assert np.isfinite(w["force"]).all()
    obs = env._setup_observables()
    for name in ("brew_button_travel", "brew_button_pressed", "wrist_ft_force", "wrist_ft_torque"):
        assert name in obs, f"missing observable {name}"
    env.close()


@pytest.mark.parametrize("profile,expected", [
    ("light", 1.50), ("medium", 2.00), ("firm", 3.00), ("stiff", 3.90), ("preloaded", 3.00),
])
def test_button_profiles_land_in_the_1p5_to_4N_band(profile, expected):
    env = _make(button_profile=profile)
    f = env.button_force_at_full_travel()
    assert f == pytest.approx(expected, rel=0.02)
    assert 1.5 <= f <= 4.0, f"{profile} gives {f:.2f} N, outside the design band"
    env.close()


def test_default_profile_keeps_the_xml_0p48N():
    env = _make()
    assert env.button_profile == "xml"
    # 80 N/m x 0.006 m = 0.48 N -- below the band, and deliberately still the default.
    assert env.button_force_at_full_travel() == pytest.approx(0.48, rel=0.02)
    env.close()


def test_explicit_stiffness_overrides_the_profile():
    env = _make(button_profile="firm", button_stiffness=100.0)
    assert env.button_stiffness == 100.0
    assert env.button_force_at_full_travel() == pytest.approx(0.60, rel=0.02)
    env.close()


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="unknown button_profile"):
        _make(button_profile="squishy")

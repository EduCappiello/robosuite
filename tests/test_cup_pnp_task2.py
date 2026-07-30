import inspect
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import robosuite as suite
from robosuite.environments.manipulation.cup_pnp_task2 import (
    BREW_BUTTON,
    BUTTON_NAMES,
    TOUCH_STYLUS_PARENT_BODY,
    TOUCH_STYLUS_SHAFT_FROMTO,
    TOUCH_STYLUS_TIP_POS,
    add_task2_robot_pedestal,
    add_task2_touch_stylus,
    classify_button_contacts,
    cupPnP_task2,
)
from robosuite.environments.manipulation.soarm101_lift import SOARM101Lift


def test_task2_is_registered_and_independent():
    assert "cupPnP_task2" in suite.ALL_ENVIRONMENTS
    assert cupPnP_task2.__bases__ == (SOARM101Lift,)


def test_task2_button_contact_classification():
    zero = {name: 0.0 for name in BUTTON_NAMES}
    assert classify_button_contacts(zero, 1e-4, set()) == (False, False)

    brew = {**zero, BREW_BUTTON: 0.5}
    assert classify_button_contacts(brew, 1e-4, set()) == (False, False)
    assert classify_button_contacts(
        brew,
        1e-4,
        {BREW_BUTTON},
    ) == (True, False)

    wrong = {**zero, "button_top_left": 0.5}
    assert classify_button_contacts(wrong, 1e-4, set()) == (False, True)
    assert classify_button_contacts(
        zero,
        1e-4,
        {"button_top_left"},
    ) == (False, True)

    simultaneous = {**wrong, BREW_BUTTON: 0.5}
    assert classify_button_contacts(
        simultaneous,
        1e-4,
        {BREW_BUTTON},
    ) == (True, True)

def test_coffee_machine_exposes_seven_physical_buttons():
    asset = (
        Path(__file__).resolve().parents[1]
        / "robosuite"
        / "models"
        / "assets"
        / "objects"
        / "coffee_machine_block_xlerobot"
        / "model.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(asset))

    joint_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    }
    sensor_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, index)
        for index in range(model.nsensor)
    }

    assert joint_names == {f"{name}_joint" for name in BUTTON_NAMES}
    assert sensor_names == {f"{name}_touch" for name in BUTTON_NAMES}

def test_task2_outcome_latches_and_no_touch_times_out():
    task = cupPnP_task2.__new__(cupPnP_task2)
    task.button_contact_threshold = 1e-4
    task.button_timeout_steps = 10
    task._brew_button_hit = False
    task._wrong_button_hit = False
    task._button_steps = 0

    forces = {name: 0.0 for name in BUTTON_NAMES}
    task._read_button_forces = lambda: forces
    stylus_contacts = set()
    task._read_stylus_button_contacts = lambda: stylus_contacts
    task._update_button_outcome()
    assert not task.task_success
    assert not task.task_failed

    forces[BREW_BUTTON] = 0.5
    task._update_button_outcome()
    assert not task.task_success
    assert not task.task_failed

    stylus_contacts.add(BREW_BUTTON)
    task._update_button_outcome()
    assert task.task_success
    assert not task.task_failed

    forces["button_top_left"] = 0.5
    task._update_button_outcome()
    assert not task.task_success
    assert task.task_failed

    timeout = cupPnP_task2.__new__(cupPnP_task2)
    timeout.button_timeout_steps = 10
    timeout._button_steps = 10
    timeout._brew_button_hit = False
    timeout._wrong_button_hit = False
    assert timeout.task_failed


def test_task2_reads_only_stylus_tip_button_contact_pairs():
    task = cupPnP_task2.__new__(cupPnP_task2)
    task._stylus_tip_geom_id = 10
    task._button_geom_names_by_id = {
        20: BREW_BUTTON,
        21: "button_top_left",
    }
    task.sim = SimpleNamespace(
        data=SimpleNamespace(
            ncon=4,
            contact=[
                SimpleNamespace(geom1=10, geom2=20),
                SimpleNamespace(geom1=21, geom2=10),
                SimpleNamespace(geom1=30, geom2=20),
                SimpleNamespace(geom1=31, geom2=32),
            ],
        ),
    )

    assert task._read_stylus_button_contacts() == {
        BREW_BUTTON,
        "button_top_left",
    }

def test_task2_robot_is_shifted_30_cm_to_robot_right():
    signature = inspect.signature(cupPnP_task2)
    default_offset = signature.parameters["robot_on_cart_offset"].default
    assert np.allclose(default_offset, [-0.07, -0.20, 0.0])

    cart_top_height = signature.parameters["cart_top_height"].default
    pedestal_size = signature.parameters["robot_pedestal_full_size"].default
    assert np.isclose(cart_top_height, 0.72)
    assert np.isclose(cart_top_height + pedestal_size[2], 0.77)


def test_task2_pedestal_raises_robot_to_original_height():
    worldbody = ET.Element("worldbody")
    cart_top = np.array([-0.585, 0.0, 0.72])
    robot_base_pos = np.array([-0.655, -0.20, 0.77])
    full_size = np.array([0.16, 0.12, 0.05])

    pedestal = add_task2_robot_pedestal(
        worldbody,
        cart_top,
        robot_base_pos,
        full_size,
    )

    center = np.fromstring(pedestal.get("pos"), sep=" ")
    collision = pedestal.find("geom[@name='task2_robot_pedestal_collision']")
    assert collision is not None
    half_size = np.fromstring(collision.get("size"), sep=" ")
    assert np.allclose(center, [-0.655, -0.20, 0.745])
    assert np.allclose(half_size, [0.08, 0.06, 0.025])
    assert np.isclose(center[2] + half_size[2], robot_base_pos[2])


def test_task2_stylus_is_attached_to_fixed_right_jaw_and_protrudes():
    worldbody = ET.Element("worldbody")
    parent = ET.SubElement(worldbody, "body", {"name": TOUCH_STYLUS_PARENT_BODY})

    stylus = add_task2_touch_stylus(worldbody)

    assert TOUCH_STYLUS_PARENT_BODY == "robot0_gripper"
    assert stylus in list(parent)
    assert stylus.get("name") == "task2_touch_stylus"
    assert stylus.get("pos") is None
    assert stylus.get("euler") is None
    assert len(stylus.findall("geom")) == 4
    assert stylus.find("geom[@name='task2_touch_stylus_bar_visual']") is None
    fixed_jaw_front_z = -0.10465
    tip_front_z = TOUCH_STYLUS_TIP_POS[2] - 0.006
    assert TOUCH_STYLUS_SHAFT_FROMTO[5] < fixed_jaw_front_z
    assert TOUCH_STYLUS_TIP_POS[2] < TOUCH_STYLUS_SHAFT_FROMTO[5]
    assert np.isclose(fixed_jaw_front_z - tip_front_z, 0.02135, atol=0.002)

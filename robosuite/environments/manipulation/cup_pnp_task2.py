import xml.etree.ElementTree as ET

import numpy as np

from robosuite.environments.manipulation.soarm101_lift import SOARM101Lift
from robosuite.utils.observables import Observable, sensor


BREW_BUTTON = "brew_button"
TOUCH_STYLUS_PARENT_BODY = "robot0_gripper"
TOUCH_STYLUS_SHAFT_FROMTO = (
    -0.039,
    -0.0002,
    -0.025,
    -0.014,
    -0.0002,
    -0.113,
)
TOUCH_STYLUS_TIP_POS = (-0.012, -0.0002, -0.120)
BUTTON_NAMES = (
    "button_top_left",
    "button_top_right",
    "button_middle_left",
    BREW_BUTTON,
    "button_lower_left",
    "button_lower_right",
    "button_power",
)


def add_task2_touch_stylus(worldbody: ET.Element) -> ET.Element:
    """Rigidly attach a collision-enabled stylus to the fixed right jaw."""
    existing = worldbody.find(".//body[@name='task2_touch_stylus']")
    if existing is not None:
        return existing

    parent = worldbody.find(f".//body[@name='{TOUCH_STYLUS_PARENT_BODY}']")
    if parent is None:
        raise ValueError(f"Missing task2 stylus parent body: {TOUCH_STYLUS_PARENT_BODY}")

    stylus = ET.SubElement(parent, "body", {"name": "task2_touch_stylus"})
    shaft_fromto = " ".join(str(value) for value in TOUCH_STYLUS_SHAFT_FROMTO)
    tip_pos = " ".join(str(value) for value in TOUCH_STYLUS_TIP_POS)

    ET.SubElement(
        stylus,
        "geom",
        {
            "name": "task2_touch_stylus_shaft_visual",
            "type": "capsule",
            "fromto": shaft_fromto,
            "size": "0.004",
            "rgba": "0.88 0.90 0.90 1",
            "mass": "0",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
        },
    )
    ET.SubElement(
        stylus,
        "geom",
        {
            "name": "task2_touch_stylus_shaft_collision",
            "type": "capsule",
            "fromto": shaft_fromto,
            "size": "0.004",
            "density": "900",
            "friction": "0.8 0.02 0.001",
            "solimp": "0.995 0.999 0.001",
            "solref": "0.005 1",
            "group": "0",
            "rgba": "0.88 0.90 0.90 0.35",
        },
    )
    ET.SubElement(
        stylus,
        "geom",
        {
            "name": "task2_touch_stylus_tip_visual",
            "type": "sphere",
            "pos": tip_pos,
            "size": "0.006",
            "rgba": "0.05 0.45 0.75 1",
            "mass": "0",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
        },
    )
    ET.SubElement(
        stylus,
        "geom",
        {
            "name": "task2_touch_stylus_tip_collision",
            "type": "sphere",
            "pos": tip_pos,
            "size": "0.006",
            "density": "1000",
            "friction": "0.8 0.02 0.001",
            "solimp": "0.995 0.999 0.001",
            "solref": "0.005 1",
            "group": "0",
            "rgba": "0.05 0.45 0.75 0.35",
        },
    )
    return stylus


def add_task2_robot_pedestal(
    worldbody: ET.Element,
    cart_top: np.ndarray,
    robot_base_pos: np.ndarray,
    full_size: np.ndarray,
) -> ET.Element:
    """Add the task2-only fixed block that raises the robot above the cart."""
    existing = worldbody.find(".//body[@name='task2_robot_pedestal']")
    if existing is not None:
        return existing

    half_size = np.asarray(full_size, dtype=float) / 2.0
    center = np.array(
        [
            float(robot_base_pos[0]),
            float(robot_base_pos[1]),
            float(cart_top[2] + half_size[2]),
        ]
    )
    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "task2_robot_pedestal",
            "pos": " ".join(str(float(value)) for value in center),
        },
    )
    size = " ".join(str(float(value)) for value in half_size)
    ET.SubElement(
        body,
        "geom",
        {
            "name": "task2_robot_pedestal_collision",
            "type": "box",
            "size": size,
            "friction": "1.0 0.005 0.0001",
            "group": "0",
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": "task2_robot_pedestal_visual",
            "type": "box",
            "size": size,
            "rgba": "0.12 0.14 0.15 1",
            "mass": "0",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
        },
    )
    return body


def classify_button_contacts(
    button_forces: dict[str, float],
    threshold: float,
    stylus_button_contacts: set[str],
) -> tuple[bool, bool]:
    """Return whether the stylus tip hit brew and any wrong button was touched."""
    touched = {name for name, force in button_forces.items() if force >= threshold}
    brew_touched = (
        BREW_BUTTON in touched
        and BREW_BUTTON in stylus_button_contacts
    )
    wrong_touched = bool(
        (touched - {BREW_BUTTON})
        or (stylus_button_contacts - {BREW_BUTTON})
    )
    return brew_touched, wrong_touched

class cupPnP_task2(SOARM101Lift):
    """Press the coffee-machine brew button while the cup is in the bay."""

    def __init__(
        self,
        *args,
        coffee_target_half_size=(0.035, 0.040, 0.015),
        cup_start_yaw_deg=0.0,
        button_contact_threshold=1e-4,
        no_touch_timeout_s=30.0,
        # --- press semantics (2026-11-19) -------------------------------------------
        # "touch" is Patricia's original rule: stylus-tip contact on the brew button with
        # touch force >= button_contact_threshold. It scores *reaching* the button.
        # "press" additionally requires the button to actually travel (and optionally to
        # carry a real force), held for press_hold_steps. Default keeps legacy behaviour
        # so existing tests and recorded rollouts stay valid.
        success_mode="touch",
        press_travel_m=0.004,        # of the 0.006 m total travel
        press_force_n=None,          # None = travel only; set e.g. 1.5-4.0 for a real press
        press_hold_steps=5,
        # --- button force-travel ----------------------------------------------------
        # XML default is stiffness=80 N/m x 0.006 m = 0.48 N at full travel, far below the
        # 1.5-4 N design band. Nothing here is a hard-coded guess of the real fixture:
        # leave button_stiffness=None to keep the XML value and set it once the printed
        # button is measured. preload is applied via jnt_springref so force at q=0 is
        # already button_preload_n, which reaches the band without an absurd stiffness.
        button_stiffness=None,
        button_preload_n=0.0,
        # --- optional click detent --------------------------------------------------
        click_detent=False,
        detent_travel_m=0.004,       # click point: resistive force peaks here
        detent_drop_n=1.0,           # peak extra resistance at the click point [N]
        cart_full_size=(0.35, 0.45, 0.04),
        cart_top_height=0.72,
        main_table_top_height=0.865,
        cart_gap=0.01,
        robot_on_cart_offset=(-0.07, -0.20, 0.0),
        robot_pedestal_full_size=(0.16, 0.12, 0.05),
        coffee_machine_offset=(-0.19, 0.0, 0.0),
        head_camera_robot_offset=(-0.402, -0.250, 0.46065),
        reward_shaping=True,
        control_freq=20,
        **kwargs,
    ):
        self.coffee_target_half_size = np.array(coffee_target_half_size, dtype=float)
        self.cup_start_yaw_deg = float(cup_start_yaw_deg)
        self.button_contact_threshold = float(button_contact_threshold)
        self.no_touch_timeout_s = float(no_touch_timeout_s)
        if success_mode not in ("touch", "press"):
            raise ValueError(f"success_mode must be 'touch' or 'press', got {success_mode!r}")
        self.success_mode = success_mode
        self.press_travel_m = float(press_travel_m)
        self.press_force_n = None if press_force_n is None else float(press_force_n)
        self.press_hold_steps = max(1, int(press_hold_steps))
        self.button_stiffness = None if button_stiffness is None else float(button_stiffness)
        self.button_preload_n = float(button_preload_n)
        self.click_detent = bool(click_detent)
        self.detent_travel_m = float(detent_travel_m)
        self.detent_drop_n = float(detent_drop_n)
        self._press_streak = 0
        self._pressed = False
        self._brew_travel_m = 0.0
        self._brew_force_n = 0.0
        self._brew_joint_qposadr = -1
        self._brew_joint_range = (0.0, 0.006)
        # Per-INSTANCE detent state. The libero-force detent had a single global registry
        # slot, so parallel envs silently overwrote each other's detents. Everything here
        # lives on self and writes into self.sim.data, which is per-env by construction.
        self._detent_dofs = []
        self.button_timeout_steps = max(1, int(round(self.no_touch_timeout_s * control_freq)))
        self._brew_button_hit = False
        self._wrong_button_hit = False
        self._button_steps = 0
        self._button_sensor_ids: dict[str, int] = {}
        self._button_geom_names_by_id: dict[int, str] = {}
        self._stylus_tip_geom_id = -1

        table_full_size = np.array(kwargs.get("table_full_size", (0.8, 0.8, 0.05)), dtype=float)
        cart_full_size = np.array(cart_full_size, dtype=float)
        cart_center_x = -(table_full_size[0] + cart_full_size[0]) / 2.0 - float(cart_gap)
        cart_top = np.array([cart_center_x, 0.0, float(cart_top_height)], dtype=float)

        self.cart_full_size = cart_full_size
        self.cart_top = cart_top
        self.robot_pedestal_full_size = np.array(robot_pedestal_full_size, dtype=float)
        self.main_table_top_height = float(main_table_top_height)
        self.cart_gap = float(cart_gap)
        self.head_camera_robot_offset = np.array(head_camera_robot_offset, dtype=float)

        kwargs.setdefault("coffee_machine_offset", coffee_machine_offset)
        kwargs.setdefault("cup_model_path", "objects/coffee_cup_3/model.xml")
        kwargs.setdefault("robot_support_table_full_size", cart_full_size)
        kwargs.setdefault("robot_support_table_offset", cart_top)
        robot_support_offset = np.array(robot_on_cart_offset, dtype=float)
        robot_support_offset[2] += self.robot_pedestal_full_size[2]
        kwargs.setdefault("robot_support_robot_offset", robot_support_offset)
        self.robot_base_pos = (
            np.array(kwargs["robot_support_table_offset"], dtype=float)
            + np.array(kwargs["robot_support_robot_offset"], dtype=float)
        )
        super().__init__(
            *args,
            reward_shaping=reward_shaping,
            control_freq=control_freq,
            **kwargs,
        )

    def _load_model(self):
        self.table_offset = np.array([0.0, 0.0, self.main_table_top_height], dtype=float)
        super()._load_model()
        add_task2_robot_pedestal(
            self.model.worldbody,
            self.cart_top,
            self.robot_base_pos,
            self.robot_pedestal_full_size,
        )
        add_task2_touch_stylus(self.model.worldbody)

        head_camera_pos = self.robot_base_pos + self.head_camera_robot_offset
        for camera_name in ("top", "rear"):
            camera = self.model.worldbody.find(f".//camera[@name='{camera_name}']")
            if camera is None:
                raise ValueError(f"Missing expected camera: {camera_name}")
            camera.set("pos", " ".join(str(float(v)) for v in head_camera_pos))

    def _setup_references(self):
        super()._setup_references()
        prefix = self.coffee_machine.naming_prefix
        self._button_sensor_ids = {
            name: self.sim.model.sensor_name2id(f"{prefix}{name}_touch")
            for name in BUTTON_NAMES
        }
        button_geom_ids = {
            name: self.sim.model.geom_name2id(f"{prefix}{name}_collision")
            for name in BUTTON_NAMES
        }
        self._button_geom_names_by_id = {
            geom_id: name
            for name, geom_id in button_geom_ids.items()
            if geom_id >= 0
        }
        self._stylus_tip_geom_id = self.sim.model.geom_name2id(
            "task2_touch_stylus_tip_collision"
        )
        self._configure_buttons()

    def _configure_buttons(self) -> None:
        """Apply button_stiffness / button_preload_n / click_detent to the seven slide joints.

        The XML gives stiffness=80 N/m over 0.006 m = 0.48 N at full travel. The design band
        for Eduardo's real task is 1.5-4 N (the SO-101 sustains ~5 N; the real FT-02 force
        estimate resolves ~0.5 N), so either raise stiffness (~400 N/m for 2.5 N at full
        travel) or add a preload via jnt_springref. Preload is the better knob: it sets the
        force at q=0 without making the spring absurdly stiff near the stop.
        """
        import mujoco

        prefix = self.coffee_machine.naming_prefix
        model = self.sim.model._model
        for name in BUTTON_NAMES:
            jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{name}_joint")
            if jnt < 0:
                continue
            if self.button_stiffness is not None:
                model.jnt_stiffness[jnt] = self.button_stiffness
            if abs(self.button_preload_n) > 0.0:
                # Spring force is -k*(q - qpos_spring); at q=0 that is +k*qpos_spring. We want
                # it to *resist* pressing (pressing moves q positive), so qpos_spring is negative.
                k = float(model.jnt_stiffness[jnt])
                if k <= 0.0:
                    raise ValueError(
                        f"button_preload_n={self.button_preload_n} needs a positive joint "
                        f"stiffness, but {name} has {k}. Set button_stiffness too."
                    )
                model.qpos_spring[int(model.jnt_qposadr[jnt])] = -self.button_preload_n / k
            if name == BREW_BUTTON:
                self._brew_joint_qposadr = int(model.jnt_qposadr[jnt])
                self._brew_joint_range = tuple(float(v) for v in model.jnt_range[jnt])
            if self.click_detent:
                self._detent_dofs.append((int(model.jnt_dofadr[jnt]), name))

    def _brew_travel(self) -> float:
        # getattr, not a plain attribute: Patricia's outcome test builds the env with
        # cupPnP_task2.__new__(...) and sets only the fields it exercises.
        if getattr(self, "_brew_joint_qposadr", -1) < 0:
            return 0.0
        return float(self.sim.data.qpos[self._brew_joint_qposadr] - self._brew_joint_range[0])

    def _apply_click_detent(self) -> None:
        """Extra resistive force that rises to a peak at the click point, then drops off.

        Written into qfrc_applied on this env's own MjData, so parallel envs cannot clobber
        each other (the bug that bit the libero-force detent, which used one global slot).
        """
        if not self._detent_dofs:
            return
        data = self.sim.data._data
        span = max(1e-9, self._brew_joint_range[1] - self._brew_joint_range[0])
        for dof, _name in self._detent_dofs:
            q = float(data.qpos[int(self.sim.model._model.jnt_dofadr[dof])])
            travel = q - self._brew_joint_range[0]
            if travel <= 0.0:
                data.qfrc_applied[dof] = 0.0
            elif travel < self.detent_travel_m:
                # rising: resistance grows to detent_drop_n at the click point
                data.qfrc_applied[dof] = -self.detent_drop_n * (travel / self.detent_travel_m)
            else:
                # past the click: the detent gives way, force falls off to zero at full travel
                over = (travel - self.detent_travel_m) / max(1e-9, span - self.detent_travel_m)
                data.qfrc_applied[dof] = -self.detent_drop_n * max(0.0, 1.0 - over)

    def step(self, action):
        self._apply_click_detent()
        return super().step(action)

        missing_sensors = [
            name
            for name, sensor_id in self._button_sensor_ids.items()
            if sensor_id < 0
        ]
        if missing_sensors:
            raise ValueError(f"Missing coffee-machine button touch sensors: {missing_sensors}")
        missing_geoms = [
            name
            for name, geom_id in button_geom_ids.items()
            if geom_id < 0
        ]
        if missing_geoms:
            raise ValueError(f"Missing coffee-machine button collision geoms: {missing_geoms}")
        if self._stylus_tip_geom_id < 0:
            raise ValueError("Missing task2 stylus tip collision geom")
    def _cup_bay_center_and_bounds(self):
        cup_half_height = float(self.cup_size[1])
        center = np.array(self.coffee_target_center, dtype=float)
        center[2] = self.coffee_tray_top_z + cup_half_height + 0.002

        half_size = np.array(self.coffee_target_half_size, dtype=float)
        low = center - half_size
        high = center + half_size
        low[0] = max(low[0], float(self.coffee_tray_center_x_bounds[0]))
        high[0] = min(high[0], float(self.coffee_tray_center_x_bounds[1]))
        if np.any(low >= high):
            raise ValueError(f"Coffee cup start has no usable volume: low={low}, high={high}")
        return 0.5 * (low + high), low, high

    def _read_button_forces(self) -> dict[str, float]:
        forces = {}
        for name, sensor_id in self._button_sensor_ids.items():
            adr = int(self.sim.model.sensor_adr[sensor_id])
            dim = int(self.sim.model.sensor_dim[sensor_id])
            forces[name] = float(np.max(self.sim.data.sensordata[adr : adr + dim]))
        return forces

    def _read_stylus_button_contacts(self) -> set[str]:
        touched = set()
        for contact_index in range(int(self.sim.data.ncon)):
            contact = self.sim.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 == self._stylus_tip_geom_id:
                button_name = self._button_geom_names_by_id.get(geom2)
            elif geom2 == self._stylus_tip_geom_id:
                button_name = self._button_geom_names_by_id.get(geom1)
            else:
                continue
            if button_name is not None:
                touched.add(button_name)
        return touched

    def _update_button_outcome(self) -> None:
        brew_touched, wrong_touched = classify_button_contacts(
            self._read_button_forces(),
            self.button_contact_threshold,
            self._read_stylus_button_contacts(),
        )
        if wrong_touched:
            self._wrong_button_hit = True
        elif brew_touched and not self._wrong_button_hit:
            self._brew_button_hit = True
        self._update_press_state()

    def _update_press_state(self) -> None:
        """Press = the brew button actually travelled (and optionally carried enough force),
        held for press_hold_steps. Wrong-button and timeout rules are unchanged."""
        if getattr(self, "success_mode", "touch") != "press":
            return          # legacy rule only; keeps the __new__-built fixture working
        brew_force = float(self._read_button_forces().get(BREW_BUTTON, 0.0))
        self._brew_force_n = brew_force
        self._brew_travel_m = self._brew_travel()
        travel_ok = self._brew_travel_m >= self.press_travel_m
        force_ok = self.press_force_n is None or brew_force >= self.press_force_n
        if travel_ok and force_ok and not self._wrong_button_hit:
            self._press_streak += 1
        else:
            self._press_streak = 0
        if self._press_streak >= self.press_hold_steps:
            self._pressed = True

    def _goal_reached(self) -> bool:
        return self._pressed if getattr(self, "success_mode", "touch") == "press" \
            else self._brew_button_hit

    @property
    def task_success(self):
        return self._goal_reached() and not self._wrong_button_hit

    @property
    def task_failed(self):
        no_touch_timeout = (
            self._button_steps >= self.button_timeout_steps
            and not self._goal_reached()
        )
        return self._wrong_button_hit or no_touch_timeout

    def _check_success(self):
        self._update_button_outcome()
        return self.task_success

    def reward(self, action=None):
        self._button_steps += 1
        self._update_button_outcome()

        if self.task_failed:
            reward = -1.0
        elif self.task_success:
            reward = 1.0
        elif self.reward_shaping:
            target_site = f"{self.coffee_machine.naming_prefix}{BREW_BUTTON}_touch_site"
            distance = self._gripper_to_target(
                gripper=self.robots[0].gripper,
                target=target_site,
                target_type="site",
                return_distance=True,
            )
            reward = 0.25 * (1.0 - np.tanh(10.0 * distance))
        else:
            reward = 0.0

        if self.reward_scale is not None:
            reward *= float(self.reward_scale)
        return float(reward)

    def _setup_observables(self):
        observables = super()._setup_observables()
        if not self.use_object_obs:
            return observables

        modality = "object"

        @sensor(modality=modality)
        def brew_button_force(obs_cache):
            return np.array([self._read_button_forces()[BREW_BUTTON]], dtype=float)

        @sensor(modality=modality)
        def max_wrong_button_force(obs_cache):
            forces = self._read_button_forces()
            return np.array(
                [max(force for name, force in forces.items() if name != BREW_BUTTON)],
                dtype=float,
            )

        @sensor(modality=modality)
        def brew_button_hit(obs_cache):
            return np.array([float(self._brew_button_hit)])

        @sensor(modality=modality)
        def wrong_button_hit(obs_cache):
            return np.array([float(self._wrong_button_hit)])

        @sensor(modality=modality)
        def brew_button_travel(obs_cache):
            return np.array([self._brew_travel_m], dtype=float)

        @sensor(modality=modality)
        def brew_button_pressed(obs_cache):
            return np.array([float(self._pressed)])

        @sensor(modality=modality)
        def wrist_ft_force(obs_cache):
            return self.read_wrist_ft()["force"]

        @sensor(modality=modality)
        def wrist_ft_torque(obs_cache):
            return self.read_wrist_ft()["torque"]

        for observable_sensor in (
            brew_button_force,
            max_wrong_button_force,
            brew_button_hit,
            wrong_button_hit,
            brew_button_travel,
            brew_button_pressed,
            wrist_ft_force,
            wrist_ft_torque,
        ):
            observables[observable_sensor.__name__] = Observable(
                name=observable_sensor.__name__,
                sensor=observable_sensor,
                sampling_rate=self.control_freq,
            )
        return observables

    def read_wrist_ft(self) -> dict:
        """Raw wrist force/tensor wrench, straight from sim.data.sensordata.

        The MuJoCo sensors robot0_wrist_ft_force / robot0_wrist_ft_torque exist in the model
        but robosuite does not expose them as observations. This reads them live, so a 500 Hz
        logger can sample them once per physics substep without going through the observable
        sampler (which runs at control_freq). Un-tared: with an empty gripper it reads the
        gripper's own weight (~0.97 N), and the components are in the sensor site frame.
        """
        out = {"force": np.zeros(3), "torque": np.zeros(3)}
        # _setup_observables() probes every sensor with {} BEFORE the robot is fully wired up,
        # at which point self.robots[0] is a placeholder without naming_prefix. Fall back to
        # the default prefix and let the sensor-id lookup below no-op if it is not there yet.
        prefix = getattr(self.robots[0], "naming_prefix", "") if self.robots else ""
        prefix = prefix or "robot0_"
        for key, suffix in (("force", "wrist_ft_force"), ("torque", "wrist_ft_torque")):
            try:
                sid = self.sim.model.sensor_name2id(f"{prefix}{suffix}")
            except Exception:
                continue
            if sid < 0:
                continue
            adr = int(self.sim.model.sensor_adr[sid])
            dim = int(self.sim.model.sensor_dim[sid])
            out[key] = np.array(self.sim.data.sensordata[adr : adr + dim], dtype=float)
        return out

    def _reset_internal(self):
        self._brew_button_hit = False
        self._wrong_button_hit = False
        self._button_steps = 0
        self._press_streak = 0
        self._pressed = False
        self._brew_travel_m = 0.0
        self._brew_force_n = 0.0
        super()._reset_internal()
        self._reset_cup_to_coffee_start()

    def _reset_cup_to_coffee_start(self):
        pos, _, _ = self._cup_bay_center_and_bounds()
        yaw = np.deg2rad(self.cup_start_yaw_deg)
        quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=float)
        self.sim.data.set_joint_qpos(self.cube.joints[0], np.concatenate([pos, quat]))

import collections
import copy
from abc import abstractmethod

import carla
import numpy as np

from .carla_base_env import CarlaBaseEnv
from .toolkit import (BasePlanner, TTCCalculator, get_location_distance,
                      get_vehicle_pos, get_vehicle_rotation,
                      get_vehicle_velocity)


class CarlaWptEnv(CarlaBaseEnv):
    """
    This is the base env for all waypoint following tasks.
    An ``ego_planner`` is required to provide waypoints for the ego vehicle.
    **DO NOT** instantiate this class directly.

    All envs that inherit from this class also inherits the following config parameters:

    * ``gps_simulation``: Configuration for simulating GPS imperfection.
        * ``enable``: Whether to enable the simulation.
        * ``delay_steps``: Number of steps to delay the GPS pose.
        * ``simple_ego_representation``: Whether to render ego as a dot.

    * ``reward``: Reward configuration.

        * ``desired_speed``: Desired speed for the ego vehicle.
        * ``scales``: Dictionary of reward scales.

            * ``waypoint``: Reward for reaching waypoints.
            * ``speed``: Reward for speed.
            * ``collision``: Penalty for collision.
            * ``out_of_lane``: Penalty for going out of lane.
            * ``time``: Penalty for each time step.

    * ``terminal``: Terminal condition configuration.

        * ``time_limit``: Maximum number of time steps.
        * ``out_lane_thres``: Distance threshold for going out of lane.

    """

    def __init__(self, config):
        super().__init__(config)
        self._pose_buffer = None

    @abstractmethod
    def get_ego_planner(self) -> BasePlanner:
        """
        Override this method to return the ego vehicle planner.
        The default behavior is to return self.ego_planner.
        """
        return self.ego_planner

    def reset(self):
        obs = super().reset()
        if hasattr(self, "prev_wpt_dist"):
            del self.prev_wpt_dist

        # Traffic Law Trackers
        self._stop_time = 0
        self._entered = 0
        self._stop_sign_state = {}
        # EC-5: Standstill time-decay counter
        self._standstill_frames = 0
        # EC-19: Collision grace period
        self._grace_frames = 5

        if (
            hasattr(self._config, "gps_simulation")
            and self._config.gps_simulation.enable
        ):
            self._pose_buffer = collections.deque(
                maxlen=self._config.gps_simulation.delay_steps + 1
            )
            initial_pose = {
                "location": get_vehicle_pos(self.get_ego_vehicle()),
                "rotation": get_vehicle_rotation(self.get_ego_vehicle()),
            }
            # Pre-fill buffer with initial pose
            for _ in range(self._config.gps_simulation.delay_steps + 1):
                self._pose_buffer.append(copy.deepcopy(initial_pose))
        return obs

    def _get_delayed_pose(self):
        if (
            hasattr(self._config, "gps_simulation")
            and self._config.gps_simulation.enable
            and self._pose_buffer
            and len(self._pose_buffer) > 0
        ):
            return self._pose_buffer[0]  # Return oldest pose in buffer
        else:
            # If disabled or buffer not ready, return current true pose
            return {
                "location": get_vehicle_pos(self.get_ego_vehicle()),
                "rotation": get_vehicle_rotation(self.get_ego_vehicle()),
            }

    def get_state(self):
        state = {
            "ego_waypoints": self.waypoints,
            "timesteps": self._time_step,
        }
        if (
            hasattr(self._config, "gps_simulation")
            and self._config.gps_simulation.enable
        ):
            state["delayed_ego_pose"] = self._get_delayed_pose()
        return state

    def apply_control(self, action) -> None:
        control = self.get_vehicle_control(action)
        self.get_ego_vehicle().apply_control(control)

    def _is_ego_near_stop_sign(self, stop_sign: carla.Actor) -> bool:
        ego_location = self.get_ego_vehicle().get_location()
        stop_sign_location = stop_sign.get_location()
        return ego_location.distance(stop_sign_location) < getattr(
            self._config, "stop_sign_proximity_threshold", 15.0
        )

    def handle_stop_sign(self):
        stop_signs = self._world._get_world().get_actors().filter("traffic.stop")
        is_near_any = False
        for stop_sign in stop_signs:
            if self._is_ego_near_stop_sign(stop_sign):
                is_near_any = True
                break

        if is_near_any:
            self._stop_time += 1
            if self._entered == 0:
                self._entered = 1
        elif self._entered == 1:
            self._entered = 2

    def on_step(self) -> None:
        self.waypoints, self.planner_stats = self.get_ego_planner().run_step()
        self.num_completed = self.planner_stats["num_completed"]

        self.handle_stop_sign()

        if (
            hasattr(self._config, "gps_simulation")
            and self._config.gps_simulation.enable
        ):
            current_pose = {
                "location": get_vehicle_pos(self.get_ego_vehicle()),
                "rotation": get_vehicle_rotation(self.get_ego_vehicle()),
            }
            self._pose_buffer.append(current_pose)

    def reward(self):
        """
        ╔══════════════════════════════════════════════════════════════╗
        ║            HIERARCHICAL REWARD SYSTEM v2.0                  ║
        ║                                                              ║
        ║  Priority Tiers (higher tier OVERRIDES lower):               ║
        ║                                                              ║
        ║  T1 - SAFETY       (collision, proximity)     weight: 50x    ║
        ║  T2 - TRAFFIC LAW  (red light, stop sign)    weight: 10x    ║
        ║  T3 - NAVIGATION   (waypoints, speed, lane)  weight: 1-5x   ║
        ║  T4 - COMFORT      (smooth steering)         weight: 0.1x   ║
        ║                                                              ║
        ║  Rule: When stopped for T1/T2 reasons, T3/T4 penalties are   ║
        ║        fully amnestied so the agent sees a CLEAR positive    ║
        ║        reward signal for correct stopping behavior.          ║
        ╚══════════════════════════════════════════════════════════════╝
        """
        reward_scales = self._config.reward.scales
        ego = self.get_ego_vehicle()

        # Use delayed pose for reward calculation if enabled
        delayed_pose = self._get_delayed_pose()
        ego_location = np.array([*delayed_pose["location"]])

        # True values for collision and speed (reward is still based on true car behavior)
        true_ego_location = np.array([*get_vehicle_pos(ego)])
        ego_velocity = np.array([*get_vehicle_velocity(ego)])
        speed_norm = np.linalg.norm(ego_velocity)

        # ================================================================
        #  PHASE 0: SCENE UNDERSTANDING (compute shared state variables)
        # ================================================================

        # Compute TTC early so we can use it across all tiers
        # Trajectory-aware obstacle detection using path waypoints
        ttc, dist_to_front = TTCCalculator.get_ttc_and_distance(
            ego, self._world.carla_world, self._world.carla_map, self.waypoints
        )

        # Waypoint direction (used by multiple reward components)
        waypoint_direction = np.array([1.0, 0.0])  # default forward
        perp_direction_norm = 0.0
        if len(self.waypoints) > 0:
            next_waypoint = self.waypoints[0]
            next_location = np.array([next_waypoint[0], next_waypoint[1]])
            yaw_radius = next_waypoint[2] * np.pi / 180
            waypoint_direction = np.array([np.cos(yaw_radius), np.sin(yaw_radius)])
            goal_offset = next_location - ego_location
            perp_direction = (
                goal_offset
                - np.dot(goal_offset, waypoint_direction) * waypoint_direction
            )
            perp_direction_norm = np.linalg.norm(perp_direction)
            self.perp_direction_norm = perp_direction_norm

        # EC-14: Use CARLA's TRUE lane center for centering measurements
        ego_wp = self._world.carla_map.get_waypoint(ego.get_location())
        lane_center = np.array(
            [ego_wp.transform.location.x, ego_wp.transform.location.y]
        )
        true_lateral_offset = np.linalg.norm(true_ego_location[:2] - lane_center)

        # Use the MORE ACCURATE of the two measurements
        dist_from_path_center = max(perp_direction_norm, true_lateral_offset)

        # --- Blocking state machine ---
        is_path_blocked = False
        is_red_light_blocking = False
        is_obstacle_blocking = False

        # Obstacle detection (18m — sync with expert decel zone)
        if 0 < ttc < 2.5 or (0 < dist_to_front <= 18.0):
            is_path_blocked = True
            is_obstacle_blocking = True

        # PATH-AWARE Red Light Detection
        # Problem: ego.get_traffic_light() returns closest light, not the one for our
        # turn direction at intersections.
        # Fix: For each red light, get its affected_lane_waypoints and check if our
        # PLANNED path waypoints are on those same lanes (road_id + lane_id match).
        is_red_light_blocking = False
        dist_to_red_light = -1.0

        if len(self.waypoints) > 0:
            carla_map = self._world.carla_map
            vehicle_loc = ego.get_location()

            # Pre-compute: convert first few path waypoints to CARLA waypoints
            path_carla_wps = []
            for wp_data in self.waypoints[:15]:
                wp_loc = carla.Location(
                    x=float(wp_data[0]), y=float(wp_data[1]), z=vehicle_loc.z
                )
                path_carla_wps.append(carla_map.get_waypoint(wp_loc))

            for tl_actor in self._world.carla_world.get_actors().filter(
                "traffic.traffic_light"
            ):
                if tl_actor.get_state() not in (
                    carla.TrafficLightState.Red,
                    carla.TrafficLightState.Yellow,
                ):
                    continue
                if vehicle_loc.distance(tl_actor.get_location()) > 40.0:
                    continue

                try:
                    affected_wps = tl_actor.get_affected_lane_waypoints()
                    if affected_wps and len(affected_wps) > 0:
                        for awp in affected_wps:
                            # POINT OF NO RETURN: Use the stop line's orientation to check if we are past it
                            v_t = vehicle_loc - awp.transform.location
                            t_fwd = awp.transform.get_forward_vector()
                            if (v_t.x * t_fwd.x + v_t.y * t_fwd.y) > 0.5:
                                continue  # Already passed this stop line waypoint!

                            awp_fw = awp.transform.get_forward_vector()
                            for p_idx, pwp in enumerate(path_carla_wps):
                                if (
                                    awp.transform.location.distance(
                                        pwp.transform.location
                                    )
                                    < 2.5
                                ):
                                    pwp_fw = pwp.transform.get_forward_vector()
                                    if awp_fw.dot(pwp_fw) > 0.5:
                                        is_red_light_blocking = True
                                        dist_to_red_light = vehicle_loc.distance(
                                            pwp.transform.location
                                        )
                                        # Emergency override: if we are moving fast and the line is < 1.0m away, we can't physically stop.
                                        if (
                                            dist_to_red_light < 1.5
                                            and (ego.get_velocity().length() * 3.6)
                                            > 15.0
                                        ):
                                            is_red_light_blocking = False
                                        break
                            if is_red_light_blocking:
                                break
                    else:
                        raise ValueError("No affected wps")
                except Exception:
                    # Fallback (Uses Trigger Volume box explicitly)
                    tl_tf = tl_actor.get_transform()
                    tl_fw = tl_tf.get_forward_vector()
                    
                    # Estimate trigger location (usually center of volume)
                    trigger_loc = tl_tf.transform(tl_actor.trigger_volume.location)
                    v_t = vehicle_loc - trigger_loc
                    if (v_t.x * tl_fw.x + v_t.y * tl_fw.y) > 2.0:
                        continue # Past the volume center

                    for pwp in path_carla_wps:
                        if tl_actor.trigger_volume.contains(pwp.transform.location, tl_tf):
                            is_red_light_blocking = True
                            dist_to_red_light = vehicle_loc.distance(pwp.transform.location)
                            break

                if is_red_light_blocking:
                    break

        if is_red_light_blocking:
            is_path_blocked = True
            self.dist_to_red_light = dist_to_red_light
        else:
            self.dist_to_red_light = -1.0

        # Curvature detection (used to relax T3/T4 at turns)
        path_curvature = 0.0
        if len(self.waypoints) >= 2:
            wpt1 = self.waypoints[0]
            wpt2 = self.waypoints[1]
            yaw1 = wpt1[2] * np.pi / 180
            yaw2 = wpt2[2] * np.pi / 180
            dir1 = np.array([np.cos(yaw1), np.sin(yaw1)])
            dir2 = np.array([np.cos(yaw2), np.sin(yaw2)])
            path_curvature = max(0, 1 - np.dot(dir1, dir2))

        heading_error = 0.0
        heading_alignment = 1.0
        if len(self.waypoints) > 0:
            ego_yaw = delayed_pose["rotation"].yaw * np.pi / 180
            ego_direction = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])
            heading_alignment = np.dot(ego_direction, waypoint_direction)
            heading_error = max(0, 1 - heading_alignment)

        curvature_factor = np.clip(
            max(0.0, 1.0 - path_curvature - 0.5 * heading_error), 0.2, 1.0
        )
        # EC-11: Extra dampen in junctions
        if ego_wp.is_junction:
            curvature_factor = min(curvature_factor, 0.3)

        # Determine driving mode for tier logic
        is_stopped = speed_norm < 0.5
        is_valid_stop = is_stopped and is_path_blocked

        # EC-5: Standstill time-decay counter
        if is_valid_stop:
            self._standstill_frames = getattr(self, "_standstill_frames", 0) + 1
        else:
            self._standstill_frames = 0

        # EC-19: Grace period countdown
        grace = getattr(self, "_grace_frames", 0)
        if grace > 0:
            self._grace_frames = grace - 1

        # ================================================================
        #  TIER 1: SAFETY — Always applied, never masked
        # ================================================================
        r_collision = 0.0
        if reward_scales["collision"] > 0 and self.is_collision():
            # EC-19: Grace period for first frames
            if getattr(self, "_grace_frames", 0) <= 0:
                r_collision = -reward_scales["collision"] * max(speed_norm, 1.0)

        # T1 Proximity — EC-1: ALWAYS applies, even when stopped
        r_proximity = 0.0
        safe_dist = 10.0  # Tăng từ 6m → 10m
        safe_ttc = 2.0
        SAFE_STOP_MIN = 8.0  # Dưới mốc này: phạt dù đã dừng

        dist_penalty = 0.0
        if 0 < dist_to_front < safe_dist:
            dist_penalty = ((safe_dist - dist_to_front) / safe_dist) ** 2

        ttc_penalty = 0.0
        if 0 < ttc < safe_ttc:
            ttc_penalty = ((safe_ttc - ttc) / safe_ttc) ** 2

        prox_intensity = max(dist_penalty, ttc_penalty)
        if prox_intensity > 0:
            if is_valid_stop and 0 < dist_to_front >= SAFE_STOP_MIN:
                # Dừng an toàn ở khoảng cách >= 8m → miễn proximity
                pass
            elif is_valid_stop and 0 < dist_to_front < SAFE_STOP_MIN:
                # EC-1: Dừng nhưng QUÁ SÁT → vẫn phạt (giảm 50% so với đang chạy)
                r_proximity = (
                    -reward_scales.get("proximity", 10.0) * prox_intensity * 0.5
                )
            else:
                # Đang chạy mà sát → phạt full
                r_proximity = -reward_scales.get("proximity", 10.0) * prox_intensity

        # ================================================================
        #  TIER 2: TRAFFIC LAW — Always applied
        # ================================================================
        r_standstill = 0.0
        r_traffic_light = 0.0
        r_stop_sign = 0.0
        r_speeding = 0.0

        standstill_base = reward_scales.get("traffic_light_obey", 2.0)

        if is_stopped:
            if is_red_light_blocking:
                # EC-5: Only reward stopping if close to the stop line (within 8m)
                # This prevents the agent from stopping too far (e.g. 30m away) to farm rewards safely.
                if 0 < dist_to_red_light <= 8.0:
                    r_standstill = standstill_base
                elif dist_to_red_light > 12.0:
                    r_standstill = -0.5 # Penalty for stopping too early
                elif dist_to_red_light > 8.0:
                    # Scale reward between 8m and 12m
                    r_standstill = standstill_base * (1.0 - (dist_to_red_light - 8.0) / 4.0)
            elif is_obstacle_blocking:
                # EC-1: Standstill reward CONDITIONAL on safe distance
                if 0 < dist_to_front < SAFE_STOP_MIN:
                    r_standstill = 0.0  # Quá sát → không thưởng
                elif dist_to_front >= 12.0 or dist_to_front <= 0:
                    r_standstill = standstill_base * 0.8  # Full (80%)
                else:
                    # 8m-12m: scaled by distance
                    ratio = (dist_to_front - SAFE_STOP_MIN) / (12.0 - SAFE_STOP_MIN)
                    r_standstill = standstill_base * 0.8 * ratio
        else:
            # Vehicle is MOVING
            if is_red_light_blocking:
                if speed_norm > 1.5:
                    # EC-5: Chạy đèn đỏ = phạt CỰC NẶNG
                    r_traffic_light = -reward_scales.get("traffic_light_violate", 20.0)
                else:
                    r_traffic_light = -0.5  # Creeping → mild

        # Stop sign violation
        if self._entered == 2:
            if self._stop_time < getattr(self._config, "stopping_time", 20):
                r_stop_sign = -reward_scales.get("stop_sign", 10.0)
            self._entered = 0
            self._stop_time = 0

        # Speeding violation
        try:
            speed_limit_ms = getattr(ego, "get_speed_limit", lambda: 30.0)() / 3.6
            if speed_norm > speed_limit_ms * 1.1:
                r_speeding = -reward_scales.get("speeding", 5.0)
        except Exception:
            pass

        # ================================================================
        #  TIER 3: NAVIGATION — Dampened when validly stopped
        # ================================================================
        # Dampen factor: T3 penalties giảm 80% khi dừng đúng lý do
        # để T2_standstill (+2.0) luôn thắng T3 penalty (max -0.8)
        t3_dampen = 0.2 if is_valid_stop else 1.0

        # Dense waypoint reward
        current_wpt_dist = self.get_wpt_dist(ego_location)

        # EC-13: Detect path replan (waypoint jumps > 5m)
        prev_dist = getattr(self, "prev_wpt_dist", current_wpt_dist)
        if (
            abs(current_wpt_dist - prev_dist) > 5.0
            and getattr(self, "num_completed", 0) == 0
        ):
            prev_dist = current_wpt_dist  # Reset to avoid false penalty

        if getattr(self, "num_completed", 0) > 0:
            raw_wpt_delta = 1.0 * self.num_completed
        else:
            raw_wpt_delta = prev_dist - current_wpt_dist

        # ENHANCED PROTECTION: Fully amnesty waypoint penalty when stopped for red light
        if is_stopped and is_red_light_blocking:
            # Complete amnesty for red light stops
            r_waypoints = max(
                0.0, np.clip(raw_wpt_delta, -1.0, 1.0) * reward_scales["waypoint"]
            )
        else:
            r_waypoints = (
                np.clip(raw_wpt_delta, -1.0, 1.0)
                * reward_scales["waypoint"]
                * t3_dampen
            )
        self.prev_wpt_dist = current_wpt_dist

        # Speed and Laziness Reward
        r_speed = 0.0
        speed_parallel = 0.0
        speed_perpendicular = 0.0
        if len(self.waypoints) > 0:
            desired_speed = self._config.reward.desired_speed
            speed_parallel = np.dot(ego_velocity, waypoint_direction)
            perp_dir = np.array([0.0, 0.0])
            if perp_direction_norm > 0.05:
                goal_offset = (
                    np.array([self.waypoints[0][0], self.waypoints[0][1]])
                    - ego_location
                )
                perp_dir_raw = (
                    goal_offset
                    - np.dot(goal_offset, waypoint_direction) * waypoint_direction
                )
                perp_dir = perp_dir_raw / max(np.linalg.norm(perp_dir_raw), 0.05)
            speed_perpendicular = np.abs(np.dot(ego_velocity, perp_dir))

            speed_score = max(
                0.0, 1.0 - abs(speed_parallel - desired_speed) / desired_speed
            )
            lateral_penalty = min(speed_perpendicular / desired_speed, 0.5)
            speed_ratio = speed_parallel / max(1.0, desired_speed)
            laziness_penalty = 0.0

            # EC-5 + EC-9: Laziness penalty — NẶNG HƠN standstill reward
            # Đèn xanh mà đứng = -5.0 > thưởng đèn đỏ +2.0
            # EXTRA PROTECTION: Never penalty when stopped for red light
            if speed_ratio < 0.2 and not is_path_blocked and not is_red_light_blocking:
                laziness_penalty = min((0.2 - speed_ratio) * 25.0, 5.0)  # Max -5.0

            r_speed = (
                (speed_score - lateral_penalty - laziness_penalty)
                * reward_scales["speed"]
                * t3_dampen
            )

        # EC-14: Out-of-lane using CARLA true lane center
        r_out_of_lane = 0.0
        if len(self.waypoints) > 0:
            excess = max(0.0, dist_from_path_center - 0.3)
            r_out_of_lane = (
                -reward_scales["out_of_lane"]
                * min(excess**2, 4.0)
                * curvature_factor
                * t3_dampen
            )

        # EC-14: Centerline using CARLA true lane center
        r_centerline = 0.0
        if len(self.waypoints) > 0:
            lane_width = 3.5
            path_score = np.exp(-3.0 * (dist_from_path_center / lane_width) ** 2)
            r_centerline = (
                reward_scales.get("centerline", 0.5)
                * path_score
                * curvature_factor
                * t3_dampen
            )

        # EC-10: Heading alignment — capped at -2.0
        r_heading = 0.0
        if len(self.waypoints) > 0:
            excess_heading = max(0.0, heading_error - 0.01)
            r_heading = (
                -reward_scales.get("heading", 0.5)
                * excess_heading
                * 10.0
                * curvature_factor
                * t3_dampen
            )
            r_heading = max(r_heading, -2.0)  # Hard cap

        # Destination
        r_destination = 0.0
        if self.is_destination_reached():
            r_destination = reward_scales.get("destination_reached", 20.0)

        # Time penalty (EC-12: anti-loitering)
        time_penalty = -reward_scales["time"]

        # ================================================================
        #  TIER 4: COMFORT — Always applied, scaled by speed
        # ================================================================
        r_steer_rate = 0.0
        r_steer_jerk = 0.0
        r_speed_steer = 0.0
        steer_diff = 0.0

        # EC-16: Scale T4 by speed — no penalty when stopped
        t4_speed_scale = min(speed_norm / 2.0, 1.0)  # 0 at stop, 1.0 at 2+ m/s

        if self.prev_action is not None:
            if self._config.action.discrete:
                current_steer = self._config.action.discrete_steer[
                    self.current_action % self.n_steer
                ]
                prev_steer = self._config.action.discrete_steer[
                    self.prev_action % self.n_steer
                ]
                steer_diff = current_steer - prev_steer
            else:
                steer_diff = self.current_action[1] - self.prev_action[1]

            r_steer_rate = (
                -reward_scales.get("steer_rate", 0.5)
                * abs(steer_diff)
                * curvature_factor
                * t4_speed_scale
            )

        if hasattr(self, "prev_steer_diff"):
            steer_jerk = abs(steer_diff - self.prev_steer_diff)
            r_steer_jerk = (
                -reward_scales.get("steer_jerk", 0.25)
                * steer_jerk
                * curvature_factor
                * t4_speed_scale
            )
        self.prev_steer_diff = steer_diff

        if self.prev_action is not None and speed_norm > 1.0:
            speed_factor = min(speed_norm / 10.0, 2.0)
            r_speed_steer = (
                -reward_scales.get("speed_steer", 0.25)
                * speed_factor
                * abs(steer_diff)
                * curvature_factor
            )

        r_smoothness = r_steer_rate + r_steer_jerk + r_speed_steer

        # EC-15: Dampen smooth penalty during proximity danger (emergency swerve)
        if r_proximity < 0:
            r_smoothness *= 0.2

        # ================================================================
        #  FINAL ADDITIVE AGGREGATION (No more GATE system)
        # ================================================================
        total_reward = sum(
            [
                # T1: Safety (always)
                r_collision,
                r_proximity,
                # T2: Traffic law (always)
                r_standstill,
                r_traffic_light,
                r_stop_sign,
                r_speeding,
                # T3: Navigation (dampened when validly stopped)
                r_waypoints,
                r_speed,
                r_out_of_lane,
                r_centerline,
                r_heading,
                r_destination,
                time_penalty,
                # T4: Comfort (always, scaled by speed)
                r_smoothness,
            ]
        )

        # Get current control for visualization
        vehicle_control = self.get_ego_vehicle().get_control()

        info = {
            **self.planner_stats,
            "ego_x": true_ego_location[0],
            "ego_y": true_ego_location[1],
            "speed_parallel": speed_parallel,
            "speed_perpendicular": speed_perpendicular,
            "speed_norm": speed_norm,
            "wpt_dis": self.get_wpt_dist(true_ego_location),
            "r_waypoints": r_waypoints,
            "r_speed": r_speed,
            "r_collision": r_collision,
            "r_proximity": r_proximity,
            "r_out_of_lane": r_out_of_lane,
            "r_standstill": r_standstill,
            "r_smoothness": r_smoothness,
            "r_steer_rate": r_steer_rate,
            "r_steer_jerk": r_steer_jerk,
            "r_speed_steer": r_speed_steer,
            "r_centerline": r_centerline,
            "r_heading": r_heading,
            "r_traffic_light": r_traffic_light,
            "r_stop_sign": r_stop_sign,
            "r_speeding": r_speeding,
            "r_destination": r_destination,
            "time_penalty": time_penalty,
            "steer_diff": steer_diff,
            "ttc": ttc,
            "dist_to_front": dist_to_front,
            "is_red_light": is_red_light_blocking,
            "dist_to_red_light": getattr(self, "dist_to_red_light", -1.0),
            "time_limit": self._config.terminal.time_limit,
            "throttle": vehicle_control.throttle,
            "steer": vehicle_control.steer,
            "brake": vehicle_control.brake,
        }

        return total_reward, info

    def is_destination_reached(self):
        return len(self.waypoints) <= 3

    def get_terminal_conditions(self):
        terminal_config = self._config.terminal
        # Use true location for terminal conditions
        ego_location = get_vehicle_pos(self.get_ego_vehicle())

        # Out of lane is triggered based on perpendicular distance from center path, NOT absolute translation
        out_of_lane_dist = getattr(self, "perp_direction_norm", 0.0)

        conds = {
            "is_collision": self.is_collision(),
            "time_exceeded": self._time_step > terminal_config.time_limit,
            "out_of_lane": out_of_lane_dist > terminal_config.out_lane_thres,
            "destination_reached": self.is_destination_reached(),
        }
        return conds

    def get_wpt_dist(self, ego_location):
        if len(self.waypoints) == 0:
            return 0
        else:
            return get_location_distance(ego_location, self.waypoints[0])

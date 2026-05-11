from abc import abstractmethod
from typing import Dict, Tuple

import carla
import cv2
import numpy as np
from gym import spaces

from ...carla_manager import WorldManager
from .base_handler import BaseHandler


class SensorHandler(BaseHandler):
    """
    Base handler for sensor data endpoints.
    """

    def __init__(self, world: WorldManager, config):
        super().__init__(world, config)
        blueprint = self._world.get_blueprint(config.blueprint)
        if "transform" in config:
            self._transform = carla.Transform(carla.Location(**config.transform))
        else:
            self._transform = carla.Transform()
            
        # Đọc attributes từ config
        attrs = getattr(config, "attributes", {})
        if hasattr(attrs, "items"):
            for attr_name, attr_value in attrs.items():
                blueprint.set_attribute(attr_name, str(attr_value))
        
        # --- [FIX MỜ ẢNH] ÉP ĐỘ PHÂN GIẢI BluePrint THEO CONFIG SHAPE ---
        # Chỉ áp dụng cho Display Camera để tránh ghi đè sai các camera khác
        if "camera" in config.blueprint and "display" in config.key:
            h, w = config.shape[0], config.shape[1]
            blueprint.set_attribute("image_size_x", str(w))
            blueprint.set_attribute("image_size_y", str(h))
            print(f"📷 [HIRES FIX] Sensor {config.key} Blueprint forced to {w}x{h}")

        self._blueprint = blueprint
        self._sensor = None
        self._data = None
        
        # Register for synchronous synchronization if it's a data sensor
        if hasattr(self._world, 'register_sensor_queue'):
            self._world.register_sensor_queue(self._config.key)

    @property
    def _default_obs_type(self) -> np.dtype:
        obs_space = self._get_observation_space()
        if isinstance(obs_space, spaces.Box):
            return obs_space.dtype
        return np.uint8

    @property
    def _default_obs(self) -> np.ndarray:
        return np.zeros(self._config.shape, dtype=self._default_obs_type)

    @abstractmethod
    def _get_observation_space(self) -> spaces.Space:
        pass

    @abstractmethod
    def _update_data(self, data) -> None:
        pass

    def get_observation_space(self) -> Dict:
        return {self._config.key: self._get_observation_space()}

    def get_observation(self, env_state: Dict) -> Tuple[Dict, Dict]:
        # SYNC FIX: Wait for data from queue if possible
        if hasattr(self._world, 'get_sensor_data'):
            data = self._world.get_sensor_data(self._config.key)
            if data is not None:
                self._data = data

        obs = {
            self._config.key: (
                self._data if self._data is not None else self._default_obs
            )
        }
        info = {}
        return obs, info

    def destroy(self) -> None:
        self._data = None
        if self._sensor is not None:
            try:
                if self._sensor.is_alive:
                    self._sensor.destroy()
            except (RuntimeError, AttributeError):
                # Sensor already destroyed or invalid
                pass
            self._sensor = None

    def reset(self, ego: carla.Actor) -> None:
        try:
            # Destroy old sensor first to avoid conflicts
            if self._sensor is not None:
                try:
                    if self._sensor.is_alive:
                        self._sensor.destroy()
                except:
                    pass
                self._sensor = None

            # Check if ego vehicle is still valid before attaching sensors
            if ego is None:
                print(f"[SENSOR] Warning: ego vehicle is None, skipping sensor reset")
                self._data = None
                return

            # Additional checks for ego validity
            try:
                if not ego.is_alive:
                    print(f"[SENSOR] Warning: ego vehicle is not alive, skipping sensor reset")
                    self._data = None
                    return

                # Try to access ego location to verify it's valid
                _ = ego.get_location()
            except:
                print(f"[SENSOR] Warning: ego vehicle is invalid, skipping sensor reset")
                self._data = None
                return

            self._sensor = self._world.spawn_unmanaged_actor(
                self._transform, self._blueprint, attach_to=ego
            )
            self._sensor.listen(self._update_data)
        except Exception as e:
            print(f"[SENSOR] Failed to spawn sensor: {e}, using default data")
            self._sensor = None
            self._data = None


class CameraHandler(SensorHandler):
    def __init__(self, world: WorldManager, config):
        super().__init__(world, config)
        self._hires_data = None

    def _get_observation_space(self) -> spaces.Space:
        return spaces.Box(low=0, high=255, shape=self._config.shape, dtype=np.uint8)

    def _update_data(self, data) -> None:
        # ULTRA OPTIMIZE: Use shared memory for zero-copy if possible
        try:
            # Try direct memory mapping for zero-copy access
            if hasattr(data, 'raw_data') and hasattr(data.raw_data, '__array_interface__'):
                camera_data = np.array(data.raw_data, dtype=np.uint8, copy=False)
            else:
                # Fallback to frombuffer (still faster than array copy)
                camera_data = np.frombuffer(data.raw_data, dtype=np.uint8)

            camera_data = camera_data.reshape(data.height, data.width, 4)
        except Exception:
            # Ultimate fallback
            camera_data = np.frombuffer(data.raw_data, dtype=np.uint8)
            camera_data = camera_data.reshape(data.height, data.width, 4)

        # OPTIMIZE: Direct slice instead of copy for RGB extraction
        camera_data = camera_data[:, :, :3]

        # Convert BGR to RGB for proper color display
        camera_data = camera_data[:, :, ::-1]

        # Store hires data only if needed
        if hasattr(self._config, 'key') and 'display' in self._config.key:
            self._hires_data = camera_data
        else:
            self._hires_data = None

        target_height = self._config.shape[0]
        target_width = self._config.shape[1]

        # OPTIMIZE: Use fastest interpolation
        if data.height != target_height or data.width != target_width:
            self._data = cv2.resize(
                camera_data, (target_width, target_height), interpolation=cv2.INTER_NEAREST
            )
        else:
            self._data = camera_data
            
        # NOTIFY MANAGER for SYNC
        if hasattr(self._world, 'on_sensor_data'):
            self._world.on_sensor_data(self._config.key, self._data)

    def get_observation(self, env_state: Dict) -> Tuple[Dict, Dict]:
        # SYNC FIX: Wait for data from queue if possible to ensure zero-lag
        if hasattr(self._world, 'get_sensor_data'):
            data = self._world.get_sensor_data(self._config.key)
            if data is not None:
                self._data = data

        obs = {
            self._config.key: (
                self._data if self._data is not None else self._default_obs
            )
        }
        info = {}
        # Đảm bảo trả về ảnh Hires gốc trong info để Monitor hiển thị nét
        if self._hires_data is not None:
            info[self._config.key + "_display"] = self._hires_data
        return obs, info


class LidarHandler(SensorHandler):
    def __init__(self, world: WorldManager, config):
        super().__init__(world, config)
        self._obs_range = config.attributes.range
        self._lidar_z = config.transform.z
        self._lidar_bin = config.lidar_bin
        self._ego_offset = config.ego_offset

    def _update_data(self, data) -> None:
        self._data = data
        if hasattr(self._world, 'on_sensor_data'):
            self._world.on_sensor_data(self._config.key, self._data)

    def _get_observation_space(self) -> spaces.Space:
        return spaces.Box(low=0, high=255, shape=self._config.shape, dtype=np.uint8)

    def get_observation(self, env_state: Dict) -> Tuple[Dict]:
        if self._data is None:
            return {self._config.key: self._default_obs}, {}

        points = np.frombuffer(self._data.raw_data, dtype=np.dtype("f4")).reshape(-1, 4)
        points = points[np.linalg.norm(points[:, :3], axis=1) <= self._obs_range]
        points[1, :] = -points[1, :]

        intensities = np.interp(
            points[:, 3], (points[:, 3].min(), points[:, 3].max()), (0, 1)
        )
        colors = (intensities[:, np.newaxis] * np.array([[255, 0, 0]])).astype(np.uint8)

        y_bins = np.arange(
            -(self._obs_range - self._ego_offset),
            self._ego_offset + self._lidar_bin,
            self._lidar_bin,
        )
        x_bins = np.arange(
            -self._obs_range / 2, self._obs_range / 2 + self._lidar_bin, self._lidar_bin
        )
        z_bins = [-self._lidar_z - 1, -self._lidar_z + 0.25, 1]
        lidar, _ = np.histogramdd(points[:, :3], bins=(x_bins, y_bins, z_bins))

        lidar = lidar[: self._config.shape[0], : self._config.shape[1], :2]
        ground_mask = lidar[:, :, 0] > 0
        obstacle_mask = lidar[:, :, 1] > 0

        image = np.zeros((lidar.shape[0], lidar.shape[1], 3), dtype=np.uint8)
        image[ground_mask] = colors[: ground_mask.sum()]
        image[obstacle_mask] = np.array([0, 255, 0], dtype=np.uint8)
        image = np.flip(image, axis=0)

        obs = {self._config.key: image}
        info = {}
        return obs, info


class CollisionHandler(SensorHandler):
    def _get_observation_space(self) -> spaces.Space:
        return spaces.Box(
            low=0, high=np.inf, shape=self._config.shape, dtype=np.float32
        )

    def _update_data(self, data) -> None:
        impulse = data.normal_impulse
        collision_intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self._data = collision_intensity * np.ones(self._config.shape)
        if hasattr(self._world, 'on_sensor_data'):
            self._world.on_sensor_data(self._config.key, self._data)

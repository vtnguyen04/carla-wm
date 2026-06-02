import time
from functools import wraps
from typing import Callable, Dict, List, Union

import carla
import numpy as np

from .utils import ActorActionDict, ActorPolygonDict, ActorTransformDict, Command
from .vehicle_manager import VehicleManager
from carla_env.toolkit.utils import get_logger

log = get_logger(log_dir=".", job_name="world_manager")


def cached_step_wise(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        cache_key = (func.__name__,) + tuple(args) + tuple(kwargs.items())
        if not hasattr(self, "_cache") or self._cache["step"] != self._time_step:
            self._cache = {"step": self._time_step}
        if cache_key not in self._cache:
            self._cache[cache_key] = func(self, *args, **kwargs)
        return self._cache[cache_key]

    return wrapper


class WorldManager:
    """
    The class to manage the world in CARLA.
    You can spawn various actors using this class.
    The actors spawned by this class will be automatically destroyed when reset.
    This class also provides methods to get information about these actors.
    """

    def __init__(self, env_config):
        self._config = env_config.world
        self._env_config = env_config

        log.info(f"[CARLA] Connecting to Carla server at {self._config.carla_port}...")
        try:
            # TRY UNIX DOMAIN SOCKET FIRST (much faster than TCP for localhost)
            import os
            uds_path = f"/tmp/carla_{self._config.carla_port}.sock"

            if os.path.exists(uds_path):
                try:
                    log.info("[CARLA] IPC: Attempting Unix Domain Socket connection (faster than TCP)")
                    self._client = carla.Client(uds_path, 0)  # UDS doesn't need port
                except Exception as e:
                    log.warning(f"[CARLA] IPC: UDS failed ({e}), falling back to TCP")
                    self._client = carla.Client("127.0.0.1", self._config.carla_port)
            else:
                # Fall back to optimized TCP
                log.info("[CARLA] IPC: Using optimized TCP connection")
                self._client = carla.Client("127.0.0.1", self._config.carla_port)

            # Reasonable timeout to avoid crashes during map changes and JIT compilation
            self._client.set_timeout(300.0)

            # Access underlying socket for low-level optimization
            import socket

            # Get the actual socket from the client (deep inspection needed)
            client_socket = None
            if hasattr(self._client, '_client') and hasattr(self._client._client, '_socket'):
                client_socket = self._client._client._socket
            elif hasattr(self._client, 'get_client_version'):
                # Try to find socket in CARLA's implementation
                try:
                    # Force a connection to establish socket
                    _ = self._client.get_client_version()

                    # Now try to access socket
                    import threading
                    for thread in threading.enumerate():
                        if hasattr(thread, '_target') and thread._target:
                            if hasattr(thread._target, '__self__'):
                                obj = thread._target.__self__
                                if hasattr(obj, '_socket'):
                                    client_socket = obj._socket
                                    break
                except:
                    pass

            # Apply socket optimizations if we found the socket
            if client_socket:
                try:
                    # Maximum performance TCP settings
                    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)  # 1MB buffer
                    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1048576)  # 1MB buffer
                    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
                    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 0)  # Disable keepalive

                    # Linux-specific optimizations
                    try:
                        client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_CORK, 0)
                        client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_DEFER_ACCEPT, 0)
                    except:
                        pass

                    log.info("[CARLA] IPC: Socket optimizations applied")
                except Exception as e:
                    log.warning(f"[CARLA] IPC: Socket optimization failed: {e}")
            else:
                log.warning("[CARLA] IPC: Could not access socket for optimization")
            
            import ast
            town_cfg = self._config.town
            
            # If town_cfg is a string that looks like a list, parse it
            if isinstance(town_cfg, str) and town_cfg.strip().startswith('[') and town_cfg.strip().endswith(']'):
                try:
                    town_cfg = ast.literal_eval(town_cfg)
                except Exception:
                    pass

            if not isinstance(town_cfg, str):
                self._town_list = list(town_cfg)
                self._current_town = np.random.choice(self._town_list)
            else:
                self._town_list = [town_cfg]
                self._current_town = town_cfg
                
            self._world = self._client.load_world(self._current_town)
            self._map = self._world.get_map()
            log.info(f"[CARLA] Map {self._current_town} loaded")
        except Exception as e:
            log.error(f"[CARLA] Failed to connect to CARLA server or load world: {e}")
            log.error(f"[CARLA] Ensure CARLA server is running on port {self._config.carla_port}.")
            raise

        settings = self._world.get_settings()
        settings.synchronous_mode = True  # BACK TO SYNC FOR STABILITY
        settings.actor_active_distance = self._config.actor_active_distance
        settings.fixed_delta_seconds = self._config.fixed_delta_seconds  # FIXED TIMESTEP FOR CONTROL STABILITY
        settings.no_rendering_mode = False  # NEED RENDERING FOR DISPLAY
        
        # STABILITY FIX: Enable substepping to prevent physics oscillations (wobble/flip)
        # Internal physics will run at 100Hz (0.01) while world ticks at 10-20Hz
        settings.substepping = True 
        settings.max_substep_delta_seconds = 0.01
        settings.max_substeps = 10
        
        self._world.apply_settings(settings)
        self._settings = settings

        self._tm_port = self._config.get("tm_port", self._config.carla_port + 6000)
        self._vehicle_manager = VehicleManager(self._client, self._tm_port, self._config.traffic)

        self._on_reset = None
        self._apply_control = None
        self._on_step = None
        self._map_switched = False
        self.actor_dict = {}
        self._time_step = 0
        self._reset_count = 0

        # --- SYNC SENSOR QUEUE SYSTEM ---
        from queue import Queue
        self._sensor_queues = {} # Dict of Queue for each sensor key
        
    def register_sensor_queue(self, key: str):
        """Register a queue to track a specific sensor's data."""
        from queue import Queue
        self._sensor_queues[key] = Queue()
        
    def get_sensor_data(self, key: str, timeout: float = 5.0):
        """Wait and retrieve data from a specific sensor queue with improved retry logic."""
        if key not in self._sensor_queues:
            return None
            
        # NẾU LÀ CẢM BIẾN SỰ KIỆN (Collision/Lidar sparse): 
        # Không được block, nếu chưa có dữ liệu thì trả về None ngay.
        if key in ["collision", "lidar"]:
            if self._sensor_queues[key].empty():
                return None
            return self._sensor_queues[key].get_nowait()
            
        # NẾU LÀ CẢM BIẾN BẮT BUỘC (Camera): Đợi dữ liệu mới nhất
        # Retry mechanism to handle transient network/rendering delay
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # We want the LATEST data, so we check if there's more in queue
                # while not self._sensor_queues[key].empty():
                #    val = self._sensor_queues[key].get_nowait()
                return self._sensor_queues[key].get(timeout=1.0)
            except:
                continue
                
        log.warning(f"[CARLA] Timeout waiting for mandatory sensor {key} after {timeout}s")
        return None

    def on_sensor_data(self, key: str, data):
        """Callback for sensor handlers to push data into the manager."""
        if key in self._sensor_queues:
            # We only care about the latest data for the current tick
            # Clear old data if any (prevents queue buildup)
            while not self._sensor_queues[key].empty():
                try: self._sensor_queues[key].get_nowait()
                except: break
            self._sensor_queues[key].put(data)

    def on_reset(self, callback: Callable[[], None]) -> None:
        """
        Register a callback function to be called when the environment is reset.
        If called multiple times, it will overwrite the previous callback.
        """
        self._on_reset = callback

    def on_step(self, callback: Callable[[], None]) -> None:
        """
        Register a callback function to be called when the environment steps.
        If called multiple times, it will overwrite the previous callback.
        """
        self._on_step = callback

    def reset(self) -> None:
        # destroy all actors
        self._time_step = 0
        self._reset_count += 1
        self._client.apply_batch_sync([carla.command.DestroyActor(id) for id in self.actor_dict])
        self.actor_dict = {}

        self._set_synchronous_mode(False)
        
        # Switch maps dynamically if a list was provided (skip on first spawn)
        map_switch_freq = self._config.get("map_switch_episodes", 1)
        # DISABLED map rotation to stick to Town01 as requested
        if False and hasattr(self, '_town_list') and len(self._town_list) > 1:
            if self._reset_count > 1 and (self._reset_count - 1) % map_switch_freq == 0:
                available_towns = [t for t in self._town_list if t != self._current_town]
                next_town = np.random.choice(available_towns)
                if next_town != self._current_town:
                    log.info(f"[CARLA] WorldManager: Episode {self._reset_count} reached switch threshold. Rotating Map from {self._current_town} to {next_town}...")
                    self._current_town = next_town

                    # Set flag for map switching to help sensor reset
                    self._map_switched = True

                    # load_world takes a few seconds but ensures a clean slate
                    self._world = self._client.load_world(self._current_town)
                    self._map = self._world.get_map()
                    self._world.apply_settings(self._settings)

                    # Give the world time to stabilize after map switch
                    time.sleep(0.2)

        if self._on_reset is not None:
            self._on_reset()

        # --- DYNAMIC WEATHER ---
        weather_presets = [
            carla.WeatherParameters.ClearNoon,
            carla.WeatherParameters.ClearSunset,
            carla.WeatherParameters.CloudyNoon,
            carla.WeatherParameters.CloudySunset,
            carla.WeatherParameters.WetNoon,
            carla.WeatherParameters.WetSunset,
            carla.WeatherParameters.MidRainyNoon,
            carla.WeatherParameters.HardRainNoon,
            carla.WeatherParameters.SoftRainSunset,
        ]
        preset = np.random.choice(weather_presets)
        self._world.set_weather(preset)
        log.info(f"[CARLA] Weather dynamically set to: {preset}")

        self._set_synchronous_mode(True)
        # This prevents some synchronization bugs
        log.info("[CARLA] WorldManager: Resetting, waiting for sensor warm-up...")
        
        # Warm-up: Tick until we get at least one camera frame
        max_warmup_ticks = 20
        warmup_success = False
        for _ in range(max_warmup_ticks):
            self._world.tick()
            # Check if camera has data (assuming 'camera' is the key)
            if 'camera' in self._sensor_queues and not self._sensor_queues['camera'].empty():
                warmup_success = True
                break
            time.sleep(0.1)
            
        if not warmup_success:
            log.warning("[CARLA] WorldManager: Sensor warm-up timed out. First steps might fail.")
        else:
            log.info("[CARLA] WorldManager: Sensor warm-up complete.")

        time.sleep(0.5)

    def step(self) -> None:
        self._time_step += 1
        # Use synchronous tick for stable control
        snapshot = self._world.tick()
        if self._on_step is not None:
            self._on_step()

    def get_blueprint_library(self, pattern_filter: str, attribute_filter: Dict[str, str] = None) -> carla.BlueprintLibrary:
        """
        Get blueprint library based on the pattern filter and attribute filter.
        """
        bps = self._world.get_blueprint_library().filter(pattern_filter)
        if attribute_filter is not None:
            for name, value in attribute_filter.items():
                bps = bps.filter_by_attribute(name, value)
        return bps

    def get_blueprint(self, pattern_filter: str, attribute_filter: Dict[str, str] = None) -> carla.ActorBlueprint:
        """
        Randomly get a blueprint from the library based on the pattern filter and attribute filter.
        """
        bps = self.get_blueprint_library(pattern_filter, attribute_filter)
        assert len(bps) > 0, f"No blueprint found for filter {pattern_filter} {attribute_filter}"
        return np.random.choice(bps)

    def get_spawn_points(self) -> List[carla.Transform]:
        """
        Get spawn points of the map that are strictly on driving lanes.
        This prevents NPCs from spawning on sidewalks, parking spots, or overlapping structures.
        """
        spawn_points = self._map.get_spawn_points()
        valid_spawns = []
        for sp in spawn_points:
            try:
                wp = self._map.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
                # Ensure the spawn point is actually near the driving lane center
                if sp.location.distance(wp.transform.location) < 1.0:
                    valid_spawns.append(sp)
            except Exception:
                pass
        
        # Fallback in case a highly custom map has no valid driving lanes
        return valid_spawns if len(valid_spawns) > 0 else spawn_points

    def get_random_spawn_point(self) -> carla.Transform:
        """
        Get a random valid spawn point of the map.
        """
        spawn_points = self.get_spawn_points()
        assert len(spawn_points) > 0, "No spawn points found"
        return np.random.choice(spawn_points)

    def try_spawn_actor(
        self,
        transform: Union[carla.Transform, None] = None,
        blueprint: Union[carla.ActorBlueprint, None] = None,
    ) -> Union[carla.Actor, None]:
        """
        Spawn an actor with the given blueprint and transform.

        :param transform: if None, use a random spawn point.
        :param blueprint: if None, use vehicle.audi* with number_of_wheels in 4 as default.

        :return: the spawned actor. If fails, return None.
        """
        if transform is None:
            transform = self.get_random_spawn_point()
        if blueprint is None:
            blueprint = self.get_blueprint("vehicle.audi*", {"number_of_wheels": "4"})
            if blueprint.has_attribute("color"):
                color = np.random.choice(blueprint.get_attribute("color").recommended_values)
                blueprint.set_attribute("color", color)
            blueprint.set_attribute("role_name", "hero")
        actor = self._world.try_spawn_actor(blueprint, transform)
        if actor is not None:
            self.actor_dict[actor.id] = actor
        return actor

    def spawn_actor(
        self,
        transform: Union[carla.Transform, None] = None,
        blueprint: Union[carla.ActorBlueprint, None] = None,
        max_try_time: int = 20,
    ) -> carla.Actor:
        """
        Equivalent to ``try_spawn_actor(transform, blueprint)``, but retry if failed.

        :param max_try_time: if None, try until success, else raise an exception after ``max_try_time``.

        .. seealso:: :py:meth:`try_spawn_actor`
        """
        actor = self.try_spawn_actor(transform, blueprint)
        try_time = 0
        while actor is None and (max_try_time is None or try_time < max_try_time):
            log.warning("[CARLA] Failed to spawn actor, retrying...")
            time.sleep(0.1)
            actor = self.try_spawn_actor(transform, blueprint)
            try_time += 1
        if actor is None:
            raise Exception(f"Failed to spawn actor after {max_try_time} attempts. The map may be overcrowded or blocked.")
        return actor

    def spawn_unmanaged_actor(self, transform: carla.Transform, blueprint: carla.ActorBlueprint, **kwargs) -> carla.Actor:
        """
        Spawn an actor with the given blueprint and transform.
        Actors spawned by this method will be omitted by this manager.
        That is, they will not be included when retrieving actor information or destroyed when reset.
        This is useful when creating sensors for :py:class:`carla_env.toolkit.observer.handlers.SensorHandler`.
        """
        return self._world.spawn_actor(blueprint, transform, **kwargs)

    def spawn_auto_actors(
        self,
        n: int,
        transforms: List[carla.Transform] = None,
        blueprints: carla.BlueprintLibrary = None,
    ) -> List[carla.Actor]:
        """
        Spawn ``n`` actors that are automatically controlled by autopilot.

        :param n: number of actors to spawn.
        :param transforms: if None, use random spawn points.
        :param blueprints: if None, use vehicle.* with number_of_wheels 4 as default.

        :return: a list of spawned actors, note that the length of the list may be less than n.
        """
        if transforms is None:
            transforms = self.get_spawn_points()
        if blueprints is None:
            blueprints = self.get_blueprint_library("vehicle.*", {"number_of_wheels": "4"})
        
        spawn_transforms = transforms[:] # Copy the list
        np.random.shuffle(spawn_transforms)

        actors_to_spawn_count = n
        actors_spawned_count = 0
        actor_list = []
        MAX_SPAWN_ATTEMPTS = 10 # Max attempts to spawn all actors
        spawn_attempt = 0

        while actors_spawned_count < actors_to_spawn_count and spawn_attempt < MAX_SPAWN_ATTEMPTS:
            batch = []
            current_batch_transforms = spawn_transforms[spawn_attempt * actors_to_spawn_count % len(spawn_transforms) : ] # Cycle through spawn points

            for i in range(min(actors_to_spawn_count - actors_spawned_count, len(current_batch_transforms))):
                transform = current_batch_transforms[i]
                bp = np.random.choice(blueprints)
                if bp.has_attribute("color"):
                    color = np.random.choice(bp.get_attribute("color").recommended_values)
                    bp.set_attribute("color", color)
                if bp.has_attribute("driver_id"):
                    driver_id = np.random.choice(bp.get_attribute("driver_id").recommended_values)
                    bp.set_attribute("driver_id", driver_id)
                bp.set_attribute("role_name", "autopilot")
                batch.append(carla.command.SpawnActor(bp, transform).then(carla.command.SetAutopilot(carla.command.FutureActor, True, self._tm_port)))
            
            for response in self._client.apply_batch_sync(batch, True): # Use True for synchronous execution
                if response.error:
                    log.error(f"[CARLA] Spawn failed because of collision at spawn position: {response.error}")
                else:
                    actor = self._world.get_actor(response.actor_id)
                    actor_list.append(actor)
                    self.actor_dict[actor.id] = actor
                    self._vehicle_manager.set_auto_lane_change(actor, self._config.auto_lane_change)
                    
                    if "background_speed" in self._config:
                        self._vehicle_manager.set_desired_speed(actor, self._config.background_speed)
                    actors_spawned_count += 1
            
            spawn_attempt += 1
            np.random.shuffle(spawn_transforms) # Reshuffle for next attempt

        if actors_spawned_count < actors_to_spawn_count:
            log.warning(f"[CARLA] Only spawned {actors_spawned_count}/{actors_to_spawn_count} auto actors after {MAX_SPAWN_ATTEMPTS} attempts.")
        
        return actor_list
    def try_spawn_aggresive_actor(
        self,
        transform: Union[carla.Transform, None] = None,
        blueprint: Union[carla.ActorBlueprint, None] = None,
    ) -> Union[carla.Actor, None]:
        """
        Similar to ``try_spawn_actor(transform, blueprint)``.
        But the actor will be automatically controlled by autopilot and ignore traffic lights and other vehicles.

        .. seealso:: :py:meth:`try_spawn_actor`
        """
        vehicle = self.try_spawn_actor(transform, blueprint)
        if vehicle is None:
            return None
        vehicle.set_autopilot(True, self._tm_port)
        self._vehicle_manager.set_auto_lane_change(vehicle, True)
        if "background_speed" in self._config:
            self._vehicle_manager.set_desired_speed(vehicle, self._config.background_speed)
        self._vehicle_manager._tm.ignore_lights_percentage(vehicle, 100)
        self._vehicle_manager._tm.ignore_vehicles_percentage(vehicle, 100)
        return vehicle

    def destroy_actor(self, actor_id: int) -> None:
        """
        Destroy an actor. Call this method if you want to manually destroy an actor spawned by this manager.

        .. warning::
           Do not call this method for actors spawned by :py:meth:`spawn_unmanaged_actor`.
           Directly call :py:meth:`carla.Actor.destroy` instead.
        """
        actor = self.actor_dict.pop(actor_id)
        actor.destroy()

    @property
    def actor_ids(self) -> List[int]:
        """
        Get the ids of all actors spawned by this manager.
        """
        return list(self.actor_dict.keys())

    @property
    def actors(self) -> List[carla.Actor]:
        """
        Get all actors spawned by this manager.
        """
        return list(self.actor_dict.values())

    @cached_step_wise
    def _get_actor_polygons(self) -> ActorPolygonDict:
        actor_polygons: ActorPolygonDict = {}

        for actor in self.actors:
            actor_transform = actor.get_transform()
            x = actor_transform.location.x
            y = actor_transform.location.y

            yaw = actor_transform.rotation.yaw * np.pi / 180

            # Get length and width of the bounding box
            bb = actor.bounding_box
            l, w = bb.extent.x, bb.extent.y

            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).T

            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).T + np.repeat([[x, y]], 4, axis=0)
            actor_polygons[actor.id] = poly.tolist()

        return actor_polygons

    @property
    def actor_polygons(self) -> ActorPolygonDict:
        """
        Get the bounding box polygons of all actors spawned by this manager.

        :return: a dictionary mapping actor IDs to their bounding box polygons.
        :rtype: dict[int, list[tuple[float, float]]]
        """
        return self._get_actor_polygons()

    @cached_step_wise
    def _get_actor_actions(self) -> ActorActionDict:
        actor_actions: ActorActionDict = {}

        for actor in self.actor_dict.values():
            try:
                actions = self._vehicle_manager._tm.get_all_actions(actor)
                actor_actions[actor.id] = [(Command(command), waypoint) for command, waypoint in actions]
            except Exception as e:  # noqa: F841
                pass

        return actor_actions

    @property
    def actor_actions(self) -> ActorActionDict:
        """
        Get the actions of all actors spawned by this manager.

        :return: a dictionary mapping vehicle IDs to their known actions.
        :rtype: dict[int, list[tuple[Command, carla.Waypoint]]]

        .. warning::
           Actors not controlled by autopilot will not have actions.
           They will not be included in the returned dictionary.
           And some actors may have an empty list if there is no known action.
        """
        return self._get_actor_actions()

    @cached_step_wise
    def _get_actor_transforms(self) -> ActorTransformDict:
        return {actor.id: actor.get_transform() for actor in self.actor_dict.values()}

    @property
    def actor_transforms(self) -> ActorTransformDict:
        """
        Get the transforms of all actors spawned by this manager.

        :return: a dictionary mapping actor IDs to their transforms.
        :rtype: dict[int, carla.Transform]
        """
        return self._get_actor_transforms()

    def _set_synchronous_mode(self, synchronous=True):
        self._settings.synchronous_mode = synchronous
        self._world.apply_settings(self._settings)
        self._vehicle_manager.set_synchronous_mode(synchronous)

    def _get_world(self):
        return self._world

    @property
    def carla_world(self):
        return self._get_world()

    def _get_map(self):
        return self._map

    @property
    def carla_map(self):
        return self._get_map()

    @cached_step_wise
    def _get_carla_actors(self, actor_type: str = "") -> List[carla.Actor]:
        filtered_actors = []
        carla_actors = self._world.get_actors()
        for actor in carla_actors:
            if actor_type in actor.type_id:
                filtered_actors.append(actor)
        return filtered_actors

    def carla_actors(self, actor_type: str = "") -> List[carla.Actor]:
        """
        Get all actors of a specific type directly through CARLA APIs.

        :param actor_type: the type of the actors to retrieve (e.g., 'vehicle', 'traffic_light').
        :return: a list of actors of the specified type.
        """
        return self._get_carla_actors(actor_type)

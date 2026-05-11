from carla_env.toolkit.utils import get_logger

log = get_logger(log_dir=".", job_name="vehicle_manager")

class VehicleManager:
    def __init__(self, client, tm_port, traffic_config):
        self._client = client
        self._traffic_config = traffic_config

        self._tm = self._client.get_trafficmanager(tm_port)
        # EXTREME Performance optimizations
        self._tm.set_global_distance_to_leading_vehicle(8.0)  # Further increase to reduce checks
        # Skip unsupported API calls for older CARLA versions
        self._tm.set_random_device_seed(traffic_config.tm_seed)
        self._tm.set_hybrid_physics_mode(True)  # Enable hybrid physics
        self._tm.set_hybrid_physics_radius(30.0)  # Very small physics radius for performance
        # Skip problematic API calls that don't exist in this CARLA version
        log.info(f"[CARLA] Traffic Manager Port: {self._tm.get_port()}")

    def set_synchronous_mode(self, sync=True):
        self._tm.set_synchronous_mode(sync)

    def set_auto_lane_change(self, actor, enable):
        self._tm.auto_lane_change(actor, enable)

    def set_desired_speed(self, actor, speed):
        self._tm.set_desired_speed(actor, speed)

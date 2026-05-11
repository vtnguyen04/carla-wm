from .carla_wpt_env import CarlaWptEnv
from .toolkit import RandomDestinationPlanner


class CarlaNavigationEnv(CarlaWptEnv):
    """
    In this task, the ego vehicle navigates through randomly-generated routes.
    Every episode picks a random spawn point and a random destination,
    producing diverse training trajectories across the entire map.

    **Provided Tasks**: ``carla_navigation``

    Available config parameters:

    * ``num_vehicles``: Number of vehicles to spawn in the environment
    * ``success_dist``: The required travel distance in meters to successfully complete the task.

    """

    def on_reset(self) -> None:
        # Spawn ego at a random location each episode for diversity
        self.ego = self._world.spawn_actor()

        # Dynamically scale vehicle spawn count based on the map size
        current_town = self._world._current_town.split('/')[-1]
        base_vehicles = self._config.num_vehicles
        
        if "Town01" in current_town or "Town02" in current_town:
            # Small towns: restrict to very low density to prevent gridlocks
            dynamic_num_vehicles = min(base_vehicles, 14)
        elif "Town03" in current_town:
            # Medium/Complex towns: requires healthy density
            dynamic_num_vehicles = max(base_vehicles, 40)
        elif "Town04" in current_town or "Town05" in current_town or "Town10" in current_town:
            # Large sprawling maps/highways: needs high density
            dynamic_num_vehicles = max(base_vehicles, 70)
        else:
            dynamic_num_vehicles = base_vehicles

        self._world.spawn_auto_actors(dynamic_num_vehicles)

        # Use the new destination-based planner for diverse routes
        self.ego_planner = RandomDestinationPlanner(
            vehicle=self.ego,
            min_route_distance=getattr(self._config, 'min_route_distance', 80.0),
        )
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = self.planner_stats["num_completed"]
        self.sum_travel_distance = self.planner_stats["travel_distance"]

    def on_step(self) -> None:
        super().on_step()
        self.sum_travel_distance += self.planner_stats["travel_distance"]

    def is_destination_reached(self):
        return self.sum_travel_distance >= self._config.success_dist

    def get_state(self):
        state = super().get_state()
        state["sum_travel_distance"] = getattr(self, "sum_travel_distance", 0.0)
        state["success_dist"] = self._config.get("success_dist", 0.0)
        return state


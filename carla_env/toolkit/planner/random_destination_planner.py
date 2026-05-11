"""
RandomDestinationPlanner — picks a random spawn point on the map as a destination,
uses GlobalRoutePlanner to find the shortest road-legal path, and when the
vehicle gets close to the destination it picks a new one.  This gives true
path diversity across episodes and within long episodes.
"""

import carla
import numpy as np

from ..carla_manager import get_vehicle_pos
from .agents.navigation.global_route_planner import GlobalRoutePlanner
from .base_planner import BasePlanner


class RandomDestinationPlanner(BasePlanner):
    """Generate diverse routes by repeatedly picking random destinations."""

    def __init__(
        self,
        vehicle: carla.Actor,
        sampling_radius: float = 2.0,
        min_route_distance: float = 80.0,
        reroute_threshold: int = 20,
    ):
        """
        Args:
            vehicle:             The ego vehicle actor.
            sampling_radius:     Resolution for the GlobalRoutePlanner graph.
            min_route_distance:  Minimum Euclidean distance between the current
                                 position and the picked destination (metres).
                                 Prevents trivially short routes.
            reroute_threshold:   When fewer than this many waypoints remain in the
                                 queue, pick a new destination and extend.
        """
        super().__init__(vehicle)
        self._grp = GlobalRoutePlanner(self._map, sampling_resolution=sampling_radius)
        self._spawn_points = self._map.get_spawn_points()
        self._min_route_distance = min_route_distance
        self._reroute_threshold = reroute_threshold

    # ── BasePlanner interface ───────────────────────────────────────────

    def init_route(self):
        """Plan the first route from the current position to a random destination."""
        self._plan_to_random_destination()

    def extend_route(self):
        """When the queue is running low, plan a continuation to a new destination."""
        if self.get_waypoint_num() < self._reroute_threshold:
            self._plan_to_random_destination()

    # ── Internal ────────────────────────────────────────────────────────

    def _plan_to_random_destination(self):
        """Pick a random spawn point far enough away and trace a road-legal route."""
        origin = self._vehicle.get_location()

        # Shuffle and pick the first spawn point that is far enough
        candidates = list(self._spawn_points)
        np.random.shuffle(candidates)

        destination = None
        for sp in candidates:
            if origin.distance(sp.location) >= self._min_route_distance:
                destination = sp.location
                break

        # Fallback: if no point is far enough, just pick any random one
        if destination is None:
            destination = np.random.choice(candidates).location

        try:
            route = self._grp.trace_route(origin, destination)
            for waypoint, _road_option in route:
                self.add_waypoint(waypoint)
        except Exception:
            # If A* fails (disconnected graph segment), fall back to
            # the simple next-waypoint approach for a few steps.
            wp = self._map.get_waypoint(origin, project_to_road=True)
            for _ in range(50):
                nexts = wp.next(2.0)
                if not nexts:
                    break
                wp = np.random.choice(nexts)
                self.add_waypoint(wp)

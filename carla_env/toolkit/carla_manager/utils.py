from enum import Enum
from typing import Dict, List, Tuple

import carla
import numpy as np


class Command(Enum):
    """Enumeration for high-level commands."""

    LaneFollow = "LaneFollow"
    LaneChangeLeft = "ChangeLaneLeft"
    LaneChangeRight = "ChangeLaneRight"
    Straight = "Straight"
    Left = "Left"
    Right = "Right"


class FollowDirections(Enum):
    """Enumeration for the follow task directions"""

    STRAIGHT = 0
    RIGHT_TURN = 1
    LEFT_TURN = 2
    RANDOM = -1


Index2Command = {index: command for index, command in enumerate(Command, start=0)}
Command2Index = {command: index for index, command in enumerate(Command, start=0)}

ActorPolygon = List[Tuple[float, float]]
ActorPolygonDict = Dict[int, ActorPolygon]

ActorAction = Tuple[Command, carla.Waypoint]
ActorActionDict = Dict[int, List[ActorAction]]

ActorTransformDict = Dict[int, carla.Transform]


def get_vehicle_pos(vehicle: carla.Actor) -> Tuple[float, float]:
    """
    Get the position of a vehicle.

    :param vehicle: carla.Actor

    :return: x, y position of the vehicle
    """
    location = vehicle.get_transform().location
    return location.x, location.y


def get_vehicle_rotation(vehicle: carla.Actor) -> carla.Rotation:
    """
    Get the rotation of a vehicle.

    :param vehicle: carla.Actor

    :return: carla.Rotation of the vehicle
    """
    return vehicle.get_transform().rotation




def get_vehicle_orientation(vehicle: carla.Actor) -> float:
    """
    Get the orientation of a vehicle.

    :param vehicle: carla.Actor

    :return: orientation of the vehicle
    """
    return vehicle.get_transform().rotation.yaw


def get_vehicle_velocity(vehicle: carla.Actor) -> Tuple[float, float]:
    """
    Get the velocity of a vehicle.

    :param vehicle: carla.Actor

    :return: tuple(float, float), x, y velocity of the vehicle
    """
    velocity = vehicle.get_velocity()
    return velocity.x, velocity.y


def get_location_distance(location1: Tuple[float, float], location2: Tuple[float, float]) -> float:
    """
    Compute the distance between two locations

    :param location1: tuple(float, float)
    :param location2: tuple(float, float)

    :return: float, distance between the two locations
    """
    return np.linalg.norm(np.array([location1[0] - location2[0], location1[1] - location2[1]]))


class TTCCalculator:
    """
    A class for calculating Time-to-Collision (TTC) between vehicles as static methods.
    """

    TTC_THRESHOLD = 100.0
    DIST_THRESHOLD = 100.0

    @staticmethod
    def is_vehicle_ahead(ego_vehicle, map, target_location, ego_wpts=None):
        """
        Check if a target vehicle is ahead of the ego vehicle.
        Uses planned path waypoints (if provided) to ensure the target is actually
        blocking our trajectory, highly accurate for intersections.
        """
        ego_location = ego_vehicle.get_location()
        ego_waypoint = map.get_waypoint(ego_location)
        target_waypoint = map.get_waypoint(target_location)

        # 1. Trajectory-based check (if waypoints are provided)
        if ego_wpts is not None and len(ego_wpts) > 0:
            target_to_path_dist = 999.0
            # Check distance from target vehicle to our planned path points
            for wp in ego_wpts[:20]: # Check up to ~20 points ahead
                wp_loc = carla.Location(x=float(wp[0]), y=float(wp[1]), z=ego_location.z)
                dist = target_location.distance(wp_loc)
                if dist < target_to_path_dist:
                    target_to_path_dist = dist
            
            # If target is more than 3.0 meters away from our path center, it's not blocking
            if target_to_path_dist > 3.0:
                return False
                
        # 2. Heuristic check (if no waypoints, or as additional filter)
        else:
            # Same-road check
            if target_waypoint.road_id != ego_waypoint.road_id or target_waypoint.lane_id != ego_waypoint.lane_id:
                if not (ego_waypoint.is_junction or target_waypoint.is_junction):
                    return False

            if target_location.distance(target_waypoint.transform.location) > (target_waypoint.lane_width / 2.0):
                return False

            ego_forward_vector = ego_vehicle.get_transform().get_forward_vector()
            target_vector = target_location - ego_location
            if target_vector.length() > TTCCalculator.DIST_THRESHOLD:
                return False

            dot_product = ego_forward_vector.dot(target_vector.make_unit_vector())
            if dot_product <= 0.7:
                return False
                
        # Additional distance check for sanity
        if ego_location.distance(target_location) > TTCCalculator.DIST_THRESHOLD:
            return False

        return True

    @staticmethod
    def find_nearby_vehicles(world, ego_vehicle, map, ego_wpts=None):
        """
        Find nearby vehicles within the specified proximity threshold.
        """
        nearby_vehicles = []
        vehicle_list = world.get_actors().filter("vehicle.*")

        for target_vehicle in vehicle_list:
            if target_vehicle.id == ego_vehicle.id:
                continue

            target_location = target_vehicle.get_location()
            if TTCCalculator.is_vehicle_ahead(ego_vehicle, map, target_location, ego_wpts):
                nearby_vehicles.append(target_vehicle)

        return nearby_vehicles

    @staticmethod
    def get_ttc_to_target(ego_vehicle, target_vehicle):
        """
        Compute the Time-to-Collision (TTC) between the ego vehicle and a target vehicle.
        """
        ego_location = ego_vehicle.get_location()
        target_location = target_vehicle.get_location()

        ego_vel = ego_vehicle.get_velocity()
        target_vel = target_vehicle.get_velocity()

        # Vector pointing from ego to target
        distance_vector = np.array([target_location.x - ego_location.x, target_location.y - ego_location.y])
        distance = np.linalg.norm(distance_vector)

        # Relative velocity vector (Ego relative to Target)
        rel_vel = np.array([ego_vel.x - target_vel.x, ego_vel.y - target_vel.y])

        if distance > 0.1:
            direction = distance_vector / distance
            # Projection of relative velocity onto the line connecting the vehicles
            approach_speed = np.dot(rel_vel, direction)
        else:
            approach_speed = 0.0

        if approach_speed > 0.1: # Only calculate TTC if they are actively getting closer
            ttc = distance / approach_speed
        else:
            ttc = TTCCalculator.TTC_THRESHOLD

        return ttc

    @staticmethod
    def get_ttc_and_distance(ego_vehicle, world, map, ego_wpts=None):
        """
        Compute the minimum Time-to-Collision (TTC) and minimum Distance between the ego vehicle and nearby blocking vehicles.
        """
        nearby_vehicles = TTCCalculator.find_nearby_vehicles(world, ego_vehicle, map, ego_wpts)
        min_ttc = TTCCalculator.TTC_THRESHOLD
        min_dist = TTCCalculator.DIST_THRESHOLD

        for target_vehicle in nearby_vehicles:
            ttc = TTCCalculator.get_ttc_to_target(ego_vehicle, target_vehicle)
            dist = ego_vehicle.get_location().distance(target_vehicle.get_location())
            if ttc < min_ttc:
                min_ttc = ttc
            if dist < min_dist:
                min_dist = dist

        ans_ttc = min_ttc if min_ttc < TTCCalculator.TTC_THRESHOLD else 0.0
        ans_dist = min_dist if min_dist < TTCCalculator.DIST_THRESHOLD else -1.0
        return ans_ttc, ans_dist

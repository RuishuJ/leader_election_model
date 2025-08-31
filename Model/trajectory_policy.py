from typing import Dict, Tuple
from duckietown_simulator.Model.intersection_manager_V2 import find_intersection_centers
from demos.demo_pid_road_network import create_map_from_json
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


Center = Tuple[float, float]
Move = Tuple[str, str]   # (incoming, outgoing), each in {"N","E","S","W"}
RobotPolicy = Dict[Tuple[str, Center, str], str]  # (robot_id, center, incoming) -> outgoing


map_object = create_map_from_json(
    "/Users/ruijiang/PycharmProjects/gym-tsl-duckietown/duckietown_simulator/assets/maps/road_network.json")

# Detect centers
centers = find_intersection_centers(map_object, x_offset=0.3, y_offset=1.5)

# ("N","E","S","W") for headings
policy: RobotPolicy = {
    ("robot1", centers[0], "N"): "E",  # robot1 turn left when entering center0 from North
    ("robot1", centers[1], "W"): "N",  # robot1 turn left when entering center1 from West

    ("robot2", centers[0], "N"): "E",  # robot2 turns left when entering center0 from North
    ("robot2", centers[0], "S"): "N",  # robot2 straight at center0 from South
    ("robot2", centers[0], "E"): "S",  # robot2 turns left when entering center0 from East
    ("robot2", centers[1], "S"): "N",  # robot2 straight at center1 from South
    ("robot2", centers[1], "W"): "S",  # robot2 turns right when entering center1 from West
    ("robot2", centers[1], "N"): "W",  # robot2 turns right when entering center1 from North

    ("robot3", centers[0], "N"): "S",  # robot3 go straight when entering center0 from North
    ("robot3", centers[1], "S"): "N",  # robot3 go straight when entering center1 from South

    ("robot4", centers[0], "S"): "N",  # robot4 go straight when entering center0 from South
    ("robot4", centers[1], "N"): "S",  # robo4 go straight when entering center1 from North

    ("robot5", centers[0], "E"): "N",  # robot5 turns right when entering center0 from East
    ("robot5", centers[1], "N"): "W",  # robot5 turns right when entering center1 from North

    ("robot6", centers[0], "S"): "E",  # robot6 turns right when entering center0 from South
    ("robot6", centers[1], "W"): "S",  # robot6 turns right when entering center1 from West

    ("robot7", centers[0], "E"): "S",  # robot7 turns left when entering center0 from East
    ("robot7", centers[1], "S"): "W",  # robot7 turns left when entering center1 from South
}

# Map (incoming, outgoing) -> turn label
TURN_FROM_IN_OUT = {
    ("N","S"): "straight", ("N","E"): "left",  ("N","W"): "right",
    ("S","N"): "straight", ("S","W"): "left",  ("S","E"): "right",
    ("E","W"): "straight", ("E","S"): "left",  ("E","N"): "right",
    ("W","E"): "straight", ("W","N"): "left",  ("W","S"): "right",
}

# list SAFE groups by size
SAFE_GROUPS_BY_SIZE = {
    # size 1: single-move groups
    1: {
        frozenset({("N","E")}),
        frozenset({("N", "S")}),
        frozenset({("N", "W")}),
        frozenset({("S","N")}),
        frozenset({("S", "E")}),
        frozenset({("S", "W")}),
        frozenset({("E","N")}),
        frozenset({("E","S")}),
        frozenset({("W","N")}),
        frozenset({("W", "S")}),
    },

    # size 2: pairs allowed together
    2: {
        frozenset({("N", "S"), ("S", "N")}),
        frozenset({("N", "E"), ("E", "N")}),
        frozenset({("N", "S"), ("E", "N")}),
        frozenset({("S", "N"), ("E", "N")}),
        frozenset({("E", "S"), ("S", "E")}),
        frozenset({("W", "N"), ("N", "W")}),
        frozenset({("S", "N"), ("N", "S")}),
        frozenset({("S", "N"), ("N", "W")}),
        frozenset({("S", "N"), ("W", "S")}),
        frozenset({("S", "W"), ("W", "S")}),
    },

    # size 3: triples allowed together
    3: {
        frozenset({("N","S"), ("S","E"), ("E","N")}),
        frozenset({("S","N"), ("N","W"), ("W","S")}),
    },
}

# a flattened view
SAFE_GROUPS_ALL = set().union(*SAFE_GROUPS_BY_SIZE.values())


#avoid float-center equality pitfalls
def same_center(a: Center, b: Center, tol: float = 1e-6) -> bool:
    return abs(a[0]-b[0]) <= tol and abs(a[1]-b[1]) <= tol

def policy_lookup(traj_policy: RobotPolicy, rid: str, center: Center, incoming: str) -> str | None:
    """
    Return outgoing dir for (rid, center, incoming), matching center with small tolerance.
    """
    # exact fast path
    out = traj_policy.get((rid, center, incoming))
    if out is not None:
        return out
    # tolerant match
    for (rr, c, inc), o in traj_policy.items():
        if rr == rid and inc == incoming and same_center(c, center):
            return o
    return None
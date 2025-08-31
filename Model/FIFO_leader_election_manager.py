from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
from duckietown_simulator.Model.intersection_manager_V2 import IntersectionManager
from demos.demo_pid_road_network import create_map_from_json
from collections import defaultdict


Timed_Waypoint = tuple[float, float, float]  # (t, x, y)
map_object = create_map_from_json(
    "/Users/ruijiang/PycharmProjects/gym-tsl-duckietown/duckietown_simulator/assets/maps/road_network.json"
)
im = IntersectionManager(map_object, default_radius=0.99)

def compute_intersection_boundaries(intersection_centers: List[Tuple[float, float]],
                                    tile_size: float = 0.6) -> dict:
    """
    Given a list of intersection centers, compute square boundary corners for each.

    Returns:
        Dict mapping intersection IDs to 4-corner lists:
        {
            'intersection_0': [(x1, y1), ..., (x4, y4)],
        }
    """
    half = tile_size / 2
    boundaries = {}
    centers_dict = {}

    for center in intersection_centers:
        cx, cy = center
        corners = [
            (cx - half, cy - half),  # bottom-left
            (cx + half, cy - half),  # bottom-right
            (cx + half, cy + half),  # top-right
            (cx - half, cy + half),  # top-left
        ]
        boundaries[center] = corners
        centers_dict[center] = (cx, cy)

    return boundaries, centers_dict

def estimate_heading_from_traj(traj):
    if len(traj) < 2:
        return None
    _, x0, y0 = traj[0]
    _, x1, y1 = traj[1]
    dx = x1 - x0
    dy = y1 - y0
    if abs(dx) > abs(dy):
        return "E" if dx > 0 else "W"
    else:
        return "N" if dy > 0 else "S"



class FIFOLeaderElection:
    def __init__(self,
                 intersection_manager,
                 trajectories: Dict[str, list[Timed_Waypoint]],
                 intersection_boundaries: Dict[str, List[Tuple[float, float]]],
                 intersection_centers: Dict[str, tuple[float, float]],
                 FIFO_radius: float = 1.8):
        self.im = intersection_manager
        self.trajectories = trajectories
        self.boundaries = intersection_boundaries
        self.centers = intersection_centers
        self.FIFO_radius = FIFO_radius
        # self.leaders: Dict[str, Optional[str]] = {}
        self.current_priority_counter = defaultdict(int)
        self.priority_queue: Dict[Tuple[float, float], Dict[str]] = {}

    def update_spatial_queue(self, intersection_id: Any, agents: dict) -> None:
        # Get or initialize the priority queue for the intersection
        if intersection_id not in self.priority_queue:
            self.priority_queue[intersection_id] = {}
            self.current_priority_counter[intersection_id] = 0

        center = intersection_id
        queue = self.priority_queue.get(intersection_id, [])
        print(f"[FIFO] Priority queue at {intersection_id}: {queue}")

        counter = self.current_priority_counter.setdefault(intersection_id, 0)

        for aid, agent in agents.items():
            pos = getattr(agent, "pose", None)
            if pos is None:
                continue

            x, y = pos[0], pos[1]
            d2 = (x - center[0]) ** 2 + (y - center[1]) ** 2

            if d2 <= self.FIFO_radius ** 2:
                if aid not in queue:
                    queue[aid] = counter
                    self.current_priority_counter[intersection_id] += 1
                    print(f"[FIFO] {aid} entered selection zone @ {intersection_id} with priority {counter}")


    # First enter selection zone, first pass
    def FIFO_elect_leader(self, intersection_id, agents, winner_ids, use_candidates=True)  -> Optional[str]:
        center = intersection_id
        boundary = self.boundaries[intersection_id]

        # Filter queue entries by winners
        if use_candidates:
            candidates = self.im.candidate_position(agents, intersection_id, radius=self.FIFO_radius)
            ids_to_consider = [aid for aid in winner_ids if aid in candidates]
        else:
            ids_to_consider = winner_ids

        # Get current priority queue
        queue = self.priority_queue.get(intersection_id, {})
        filtered_queue = {aid: queue[aid] for aid in ids_to_consider if aid in queue}

        if not filtered_queue:
            print(f"[FIFO] No matching candidates in priority queue at {intersection_id}")
            return None

        # Select leader with the lowest priority number
        leader_id = min(filtered_queue.items(), key=lambda item: item[1])[0]
        print(f"[FIFO] Selected leader at {intersection_id}: {leader_id} ")
        return leader_id



def main():
    # Define mock intersection center
    intersection_centers = [(1.0, 1.0)]
    intersection_boundaries, intersection_centers_dict = compute_intersection_boundaries(intersection_centers)

    # Define mock trajectories for 3 robots
    # Format: list of (t, x, y) = Timed_Waypoint
    trajectories = {
        "robot1": [(0.0, 0.0, 1.0), (1.0, 0.5, 1.0), (2.0, 1.0, 1.0)],
        "robot2": [(0.0, 2.0, 1.0), (1.0, 1.5, 1.0), (1.5, 1.0, 1.0)],
        "robot3": [(0.0, 1.0, 2.0), (1.0, 1.0, 1.5), (1.8, 1.0, 1.0)]
    }

    # Instantiate FIFOLeaderElection
    fifo = FIFOLeaderElection(
        intersection_manager=None,
        trajectories=trajectories,
        intersection_boundaries=intersection_boundaries,
        intersection_centers=intersection_centers_dict,
        FIFO_radius=0.5
    )

    # Provide dummy robot data (not used in computation here)
    mock_agents = {
        "robot1": {"heading": "E"},  # entering from west (to east)
        "robot2": {"heading": "W"},
        "robot3": {"heading": "S"}
    }
    winner_ids = list(trajectories.keys())

    # Call FIFO_elect_leader
    intersection_id = (1.0, 1.0)
    fifo.update_spatial_queue(intersection_id, mock_agents)

    leader = fifo.FIFO_elect_leader(
        intersection_id=intersection_id,
        agents=mock_agents,
        winner_ids=winner_ids,
        use_candidates=False
    )
    print(f"🏁 Elected leader for intersection {intersection_id}: {leader}")

if __name__ == "__main__":
    main()

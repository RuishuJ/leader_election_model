from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional, Set, Iterable
import math
import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from duckietown_simulator.Model.intersection_manager_V2 import _extract_bot
from duckietown_simulator.Model.intersection_manager_V2 import IntersectionManager, find_intersection_centers
from duckietown_simulator.Model.trajectory_conflict import same_lane

from demos.demo_pid_road_network import create_map_from_json
from duckietown_simulator.Model.FIFO_leader_election_manager import FIFOLeaderElection, compute_intersection_boundaries
from tests.test_FIFO import load_trajectory
from itertools import combinations

map_object = create_map_from_json(
    "/Users/ruijiang/PycharmProjects/gym-tsl-duckietown/duckietown_simulator/assets/maps/road_network.json")

intersection_centers_world = find_intersection_centers(map_instance=map_object, x_offset=0.3, y_offset=1.5)
print(f"[centers] coords: {intersection_centers_world}")
intersection_boundaries, intersection_centers_dict = compute_intersection_boundaries(intersection_centers_world)

# ###for this script
# traj_dict = {
#     "robot1": "../../data/looped_trajectory_11.json",
#     "robot2": "../../data/looped_trajectory_2.json",
#     "robot3": "../../data/looped_trajectory_3.json",
#     "robot4": "../../data/looped_trajectory_41.json",
#     "robot5": "../../data/looped_trajectory_5.json",
#     "robot6": "../../data/looped_trajectory_6.json",
#     "robot7": "../../data/looped_trajectory_7.json",
# }
###for test
traj_dict = {
    "robot1": "../data/looped_trajectory_11.json",
    "robot2": "../data/looped_trajectory_2.json",
    "robot3": "../data/looped_trajectory_3.json",
    "robot4": "../data/looped_trajectory_41.json",
    "robot5": "../data/looped_trajectory_5.json",
    "robot6": "../data/looped_trajectory_6.json",
    "robot7": "../data/looped_trajectory_7.json",
}

trajectories = {
    rid: load_trajectory(path)
    for rid, path in traj_dict.items()
}
im = IntersectionManager(map_object, default_radius=0.80)

fifo = FIFOLeaderElection(intersection_manager = im,
    trajectories = trajectories,
    intersection_boundaries = intersection_boundaries,
    intersection_centers = intersection_centers_dict,
    FIFO_radius= 1.0)

# lower is better
PRIORITY_CLASS = {
    "ambulance": 0,
    "bus": 1,
    "private": 2,
}

Center = Tuple[float, float]

# Meta assignment
class VehicleMeta:
    priority_class: str  # "ambulance" | "bus" | "private"
    rank: int  # 0 best, 1, 2 worst
    intersection_id: Center
    incoming: Optional[str] = None
    outgoing: Optional[str] = None
    turn_type: str = "straight"

def assign_vehicle_meta(
    vehicles: Iterable[str],
    center: Center,
    traj_policy: Dict[Tuple[str, Center, str], str],
    turn_lookup: Optional[Dict[Tuple[str, str], str]] = None,
    priority_map: Optional[Dict[str, str]] = None,
) -> Dict[str, VehicleMeta]:
    """
    Fill VehicleMeta for each robot in `vehicles` at `center`.
    - priority_class -> rank
    - incoming/outgoing via traj_policy[(rid, center, incoming)] = outgoing
    - turn_type via turn_lookup[(incoming, outgoing)]
    """
    # Default setting
    if priority_map is None:
        priority_map = {"robot1": "ambulance", "robot7": "ambulance", "robot2": "bus", "robot5": "bus"}
    rank_map = {"ambulance": 0, "bus": 1, "private": 2}

    if turn_lookup is None:
        turn_lookup = {
            ("N","S"): "straight", ("N","E"): "left",  ("N","W"): "right",
            ("S","N"): "straight", ("S","W"): "left",  ("S","E"): "right",
            ("E","W"): "straight", ("E","S"): "left",  ("E","N"): "right",
            ("W","E"): "straight", ("W","N"): "left",  ("W","S"): "right",
        }

    meta: Dict[str, VehicleMeta] = {}
    for rid in vehicles:
        vm = VehicleMeta()
        vm.priority_class = priority_map.get(rid, "private")
        vm.rank = rank_map[vm.priority_class]
        vm.intersection_id = center

        # Find a matching incoming in policy for this center
        vm.incoming, vm.outgoing = None, None
        for incoming_dir in ("N", "E", "S", "W"):
            out = traj_policy.get((rid, center, incoming_dir))
            if out is not None:
                vm.incoming = incoming_dir
                vm.outgoing = out
                break

        vm.turn_type = turn_lookup.get((vm.incoming, vm.outgoing), "straight")
        meta[rid] = vm
    return meta


def build_membership_maps(im, robots, centers: List[Center], *, sel_radius: float, ctrl_radius: float):
    """
    Returns:
      zones_select: List[(center, radius, [ids])]
      zones_control: List[(center, radius, [ids])]
      members_by_center: Dict[center -> set(ids in selection)]
      control_by_center: Dict[center -> set(ids in control)]
    """
    zones_select = im.in_the_intersection_zone(robots, centers, radius=sel_radius)
    zones_control = im.in_the_intersection_zone(robots, centers, radius=ctrl_radius)
    members_by_center = {c: set(vs) for (c, _r, vs) in zones_select}
    control_by_center = {c: set(vs) for (c, _r, vs) in zones_control}
    return zones_select, zones_control, members_by_center, control_by_center


def share_conflicting_center(
    a: str,
    b: str,
    members_by_center: Dict[Center, Set[str]],
    conflict_matrix: Dict[str, Set[str]],
) -> Tuple[bool, Optional[Center]]:
    """True if a and b are in the same selection zone AND conflict by policy; also returns that center."""
    for center, members in members_by_center.items():
        if a in members and b in members:
            conflict = (b in conflict_matrix.get(a, set())) or (a in conflict_matrix.get(b, set()))
            return conflict, center if conflict else (False, None)
    return False, None


def is_approaching_center(im, robots, rid: str, center: Center, speed_thresh: float = 0.02) -> bool:
    """
    Approaching test: v > thresh and v · (center - pos) > 0.
    Uses robot.pose: (x, y, theta) and robot.linear_velocity.
    """
    bot = robots[rid]
    v = float(getattr(bot, "linear_velocity", 0.0))
    if v < speed_thresh:
        return False
    x, y, th = bot.pose
    cx, cy = center
    vx = v * np.cos(th)
    vy = v * np.sin(th)
    return (vx * (cx - x) + vy * (cy - y)) > 0.0


#  Co-leader selection
def select_co_leaders_with_leader(
    leader_id: str,
    vehicles: List[str],
    meta: Dict[str, VehicleMeta],
    conflict_matrix: Dict[str, Set[str]],
    SAFE_GROUPS_BY_SIZE: Dict[int, Set[frozenset]],
) -> List[str]:
    """
    Return a largest safe group that includes the leader.
    Prefers explicitly-safe groups from SAFE_GROUPS_BY_SIZE; falls back to greedy pairwise.
    """
    def can_pair(a: str, b: str) -> bool:
        return (b not in conflict_matrix.get(a, set())) and (a not in conflict_matrix.get(b, set()))

    # Candidates: leader + those that don't conflict with leader
    candidates = [leader_id] + [rid for rid in vehicles if rid != leader_id and can_pair(leader_id, rid)]

    # Build move map (incoming, outgoing) for SAFE_GROUPS check
    def move_of(rid: str) -> Optional[Tuple[str, str]]:
        m = meta.get(rid)
        if not m or m.incoming is None or m.outgoing is None:
            return None
        return (m.incoming, m.outgoing)

    moves_by_robot = {rid: move_of(rid) for rid in candidates}
    moves_by_robot = {rid: mv for rid, mv in moves_by_robot.items() if mv is not None}

    # Try exact SAFE_GROUPS match first
    ids = list(moves_by_robot.keys())
    for k in range(len(ids), 0, -1):
        for subset in combinations(ids, k):
            if leader_id not in subset:
                continue
            move_set = frozenset(moves_by_robot[r] for r in subset)
            if move_set in SAFE_GROUPS_BY_SIZE.get(k, set()):
                return list(subset)

    # Fallback: greedy pairwise mutual non-conflict
    group = [leader_id]
    for rid in candidates:
        if rid == leader_id:
            continue
        if all(can_pair(rid, g) for g in group):
            group.append(rid)
    return group


# Action combination & clamps
def combine_with_stop_dominance(
    global_overlay_actions: Dict[str, int],
    reservation_actions: Dict[str, int],
    all_ids: Iterable[str],
    *,
    stop_value: int,
    go_value: int,
) -> Dict[str, int]:
    """
    STOP dominates. Overlays applied first; reservation afterward.
    Reservation STOP always sticks; reservation GO only applies if not already STOP.
    """
    final_actions = {rid: go_value for rid in all_ids}

    # overlays: only set STOP
    for rid, act in global_overlay_actions.items():
        if act == stop_value:
            final_actions[rid] = stop_value

    # reservation: STOP sticks, GO only if not STOP
    for rid, act in reservation_actions.items():
        if act == stop_value:
            final_actions[rid] = stop_value
        else:
            if final_actions[rid] != stop_value:
                final_actions[rid] = go_value
    return final_actions


def enforce_hard_stop_in_control_zone(
    actions_dict: Dict[str, int],
    zones_control,
    leader_by_center: Dict[Center, str],
    *,
    stop_value: int,
) -> Dict[str, int]:
    """Force STOP for all non-leaders that are inside the control zone."""
    for (center, _r, vs) in zones_control:
        leader = leader_by_center.get(center)
        for rid in vs:
            if rid != leader:
                actions_dict[rid] = stop_value
    return actions_dict



class TrafficState:
    def __init__(self,
                 robots: Dict[str, Any],
                 meta: Dict[str, Any],
                 wait_time: Dict[str, float],
                 time_to_clear: Dict[str, float],
                 conflict_matrix: Dict[str, Set[str]],
                 im: IntersectionManager,
                 ):
        self.robots = robots
        self.meta = meta
        self.wait_time = wait_time  # seconds waited at stop
        self.time_to_clear = time_to_clear  # est. seconds to clear box
        self._conflict = conflict_matrix  # precomputed compatibility
        self.im = im

    def rank_of(self, i: str) -> int:
        return self.meta[i].rank

    def turn_type_of(self, i: str) -> str:
        return getattr(self.meta[i], "turn_type", "straight") # Default to 'straight' if not present

    def intersection_of(self, i: str) -> str:
        """Return the intersection id for robot i, or None if unknown."""
        m = self.meta.get(i)
        return getattr(m, "intersection_id", None) if m is not None else None

    def arrival_time(self, samples: List[Tuple[float, float, float]],
                      target: Tuple[float, float],
                      tolerance: float = 0.05) -> Optional[float]:
        """
        Return the first timestamp where (x, y) is within `tol` of `target`.
        """
        tx, ty = target # boundary point
        for t, x, y in samples:
            if math.hypot(x - tx, y - ty) <= tolerance:
                return t
        return None

    # def eta_to_boundary(self,
    #                     boundary: list[tuple[float, float]],
    #                     heading: str,
    #                     samples: list[tuple[float, float, float]],
    #                     vehicle_pos: tuple[float, float],
    #                     tol: float = 0.05) -> Optional[float]:
    #     """ETA for this robot to reach the closest point on the heading edge."""
    #     target = fifo.closest_point_on_heading_edge(boundary, heading, vehicle_pos)
    #     return self.arrival_time(samples, target, tolerance = tol)

    def conflicts_of(self, i: str) -> Set[str]:
        """Vehicles that cannot go simultaneously with i (same intersection)."""
        inter = self.intersection_of(i)
        if inter is None:
            return set()
        return {j for j in self._conflict.get(i, set())
                if j in self.meta and self.intersection_of(j) == inter}

    def non_conflicting(self, i: str) -> Set[str]:
        """Robots at the same intersection as i, present in meta, excluding conflicts and itself."""
        inter = self.intersection_of(i)
        if inter is None:
            return set()
        all_same = {j for j in self.robots.keys()
                    if self.intersection_of(j) == inter and j != i}
        return all_same - self.conflicts_of(i)

    def follower_id_if_within_headway_by_position(
            self,
            leader_id: str,
            s_headway_m: float,
            *,
            position_tolerance: float = 0.15,
            candidate_ids: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Pick the best follower behind `leader_id` in the *same lane* based on current positions,
        not ETA. The follower must:
          - be in the same lane (via `same_lane`)
          - be *behind* the leader along the leader's heading axis
          - be within `s_headway_m` longitudinal gap (meters)

        Preference: smallest longitudinal gap; tie-break by lower rank.
        """
        robots: Dict[str, Any] = self.robots
        if leader_id not in robots:
            return None

        # Leader pose and lane heading (cardinal)
        L = robots[leader_id]
        lx, ly, lth = _extract_bot(L).pose
        dir_lead = im.compass(lth)  # "N","E","S","W"

        ids = candidate_ids if candidate_ids is not None else list(robots.keys())

        best_id: Optional[str] = None
        best_gap: float = float("inf")
        best_rank: int = 10 ** 9

        for rid in ids:
            if rid == leader_id or rid not in robots:
                continue

            # Same lane check (you already defined this)
            try:
                if not same_lane(im, robots[leader_id], robots[rid], position_tolerance=position_tolerance):
                    continue
            except Exception:
                continue

            # Longitudinal gap based on leader's heading
            x, y, _ = _extract_bot(robots[rid]).pose
            gap = None  # positive if candidate is *behind* leader

            if dir_lead == "N":
                # forward +y → follower has smaller y
                if y < ly:
                    gap = ly - y
            elif dir_lead == "S":
                # forward -y → follower has larger y
                if y > ly:
                    gap = y - ly
            elif dir_lead == "E":
                # forward +x → follower has smaller x
                if x < lx:
                    gap = lx - x
            else:  # "W"
                # forward -x → follower has larger x
                if x > lx:
                    gap = x - lx

            if gap is None or gap > s_headway_m:
                continue

            r_rank = self.meta[rid].rank
            if (gap < best_gap) or (math.isclose(gap, best_gap) and r_rank < best_rank):
                best_id = rid
                best_gap = gap
                best_rank = r_rank

        return best_id

    def follower_rank_if_within_headway_by_position(
            self,
            leader_id: str,
            s_headway_m: float,
            *,
            position_tolerance: float = 0.15,
            candidate_ids: Optional[List[str]] = None,
    ) -> Optional[int]:
        fid = self.follower_id_if_within_headway_by_position(
            leader_id, s_headway_m,
            position_tolerance=position_tolerance,
            candidate_ids=candidate_ids,
        )
        return self.meta[fid].rank if fid is not None else None

    def update_wait_times(state: "TrafficState", dt: float = 0.05):
        """
        Increment wait_time for stopped robots, reset for moving ones.
        """
        for rid, info in state.robots.items():
            bot = _extract_bot(info)
            v = bot.linear_velocity
            if abs(v) < 1e-3:  # considered stopped
                state.wait_time[rid] = state.wait_time.get(rid, 0.0) + dt
            else:
                state.wait_time[rid] = 0.0

    # Geometry-based distances per move (tune radii as you like)
    _MOVEMENT_DISTANCE = {
        ("N", "S"): 0.6,
        ("S", "N"): 0.6,
        ("N", "E"): np.pi * 0.45 / 2,
        ("N", "W"): np.pi * 0.15 / 2,
        ("S", "E"): np.pi * 0.15 / 2,
        ("S", "W"): np.pi * 0.45 / 2,
        ("E", "N"): np.pi * 0.15 / 2,
        ("E", "S"): np.pi * 0.45 / 2,
        ("W", "N"): np.pi * 0.45 / 2,
        ("W", "S"): np.pi * 0.15 / 2,
    }

    def move_of(self, rid: str) -> tuple[str, str] | None:
        m = self.meta.get(rid)
        if not m:
            return None
        inc = getattr(m, "incoming", None)
        out = getattr(m, "outgoing", None)
        if inc is None or out is None:
            return None
        return (inc, out)

    def _speed_realtime(self, rid: str, v_avg: float = 0.25, min_speed: float = 0.05) -> float:
        """
        Use the robot's real-time forward speed (m/s). If nearly stopped, fall back to v_avg
        so we don't divide by ~0 and freeze decisions forever.
        """
        robot = self.robots.get(rid)
        v = float(getattr(robot, "linear_velocity", 0.0)) if robot is not None else 0.0
        return v if v >= min_speed else v_avg

    def time_to_clear_single(self, rid: str, *, v_avg: float = 0.25, min_speed: float = 0.05) -> float:
        """
        Time = distance(move) / real-time speed (with fallback to v_avg if too slow).
        """
        mv = self.move_of(rid)
        dist = self._MOVEMENT_DISTANCE.get(mv, 0.6)
        v = self._speed_realtime(rid, v_avg=v_avg, min_speed=min_speed)
        return dist / max(1e-6, v)

    def time_to_clear_group_if_leader(self, leader_id: str, *, v_avg: float = 0.25,
                                      min_speed: float = 0.05) -> float:
        """
        If leader_id goes, allow all its non-conflicting peers to go simultaneously.
        Return the *max* clearing time among that GO group (leader + compatible co-leaders),
        computed with *real-time* speeds for each robot.
        """
        group = [leader_id] + list(self.non_conflicting(leader_id))
        times = [self.time_to_clear_single(r, v_avg=v_avg, min_speed=min_speed) for r in group]
        return max(times) if times else 0.0


class HeuristicLeaderElectionManager:
    def __init__(self, parameters: Dict[str, float]):
        self.parameters = parameters
        # weights
        self.lambda1 = parameters.get("lambda1", 0.2)   # WaitSelf
        self.lambda2 = parameters.get("lambda2", 0.1)   # WaitOther
        self.lambda3 = parameters.get("lambda3", 0.4)   # PrioSelf
        self.lambda4 = parameters.get("lambda4", 0.1)   # PrioOther
        self.lambda5 = parameters.get("lambda5", 0.1)   # Trajectory
        self.lambda6 = parameters.get("lambda6", 0.1)   # Mate
        # thresholds
        self.T0 = parameters.get("T0", 3.0)
        self.T_cap = parameters.get("T_cap", 8.0)
        self.headway = parameters.get("headway", 1.5)


    def wait_self(self, i, state: TrafficState)-> float:
        wait_self = state.wait_time.get(i, 0.0)

        return 0.5 * (1.0 - np.tanh(wait_self - self.T0))

    def wait_priority_mapping(self, p: int) -> float:
        # map {0,1,2} to weight {1, 0.5, 0}
        p = max(0, min(2, int(p)))

        return (2 - p) / 2.0

    # def wait_other(self, i, state: TrafficState)-> float:
    #     t_self_pass = state.time_to_clear.get(i, 0.0) + self.headway
    #
    #     others: List[int] = list(state.conflicts_of(i)) if state.conflicts_of(i) else []
    #     if not others:
    #         return 0.0
    #
    #     # normalize by T_cap safely
    #     norm = t_self_pass / max(1e-6, self.T_cap)
    #
    #     vals = [self.wait_priority_mapping(state.rank_of(j)) * norm for j in others]
    #     score = sum(vals) / float(len(others))  # average over others
    #
    #     # keep in [0,1] (optional clamp)
    #     return max(0.0, min(1.0, score))

    # def wait_other(self, i, state: TrafficState) -> float:
    #     """
    #     Penalize leaders that would occupy the box longer (with their co-leaders),
    #     using *real-time* speeds. Normalize by T_cap to keep ~[0,1].
    #     """
    #     t_group = state.time_to_clear_group_if_leader(i, v_avg=0.25, min_speed=0.05)
    #     return min(1.0, t_group / max(1e-6, self.T_cap))

    def wait_other(self, i: str, state: TrafficState) -> float:
        """
        Penalize candidate i by the priority-weighted pressure from vehicles that
        CONFLICT with i at the same intersection, scaled by how long i's GO group
        (leader + compatible co-leaders) would occupy the box.

        Formula (bounded 0..1):
            wait_other = ( (1/N) * Σ_{j in conflicts(i)} w(rank(j)) ) * (t_self_pass / T_cap)

        where w(p) = (2 - p) / 2  in {1, 0.5, 0},  p ∈ {0,1,2}
        and t_self_pass uses real-time speeds via state.time_to_clear_group_if_leader.
        """
        # set of conflicting vehicles in the same intersection
        conflicts = list(state.conflicts_of(i)) if state.conflicts_of(i) else []
        N = len(conflicts)
        if N == 0:
            return 0.0

        # group clear time if i leads (includes compatible co-leaders)
        t_self_pass = state.time_to_clear_group_if_leader(i, v_avg=0.25, min_speed=0.05)

        # priority weight mapping
        def prio_w(rank_int: int) -> float:
            r = max(0, min(2, int(rank_int)))
            return (2 - r) / 2.0  # {0->1.0, 1->0.5, 2->0.0}

        # average priority weight across all conflicting vehicles
        avg_pressure = sum(prio_w(state.rank_of(j)) for j in conflicts) / float(N)

        # time normalization (keep ≤1.0)
        time_norm = min(1.0, t_self_pass / max(1e-6, self.T_cap))

        val = avg_pressure * time_norm
        return max(0.0, min(1.0, float(val)))

    def priority_self(self, i, state: TrafficState) -> float:
        """
        Lower is better. Prefer leaders whose same-lane follower (within headway)
        has an equal or better (lower) rank — that supports short platoons.
        """
        r_i = int(state.rank_of(i))
        r_f = state.follower_rank_if_within_headway_by_position(i, self.headway)
        if r_f is None:
            r_f = r_i

        r_star = min(r_i, int(r_f))

        r_star = max(0, min(2, r_star))
        return r_star / 2.0

    def priority_other(self, i, state: TrafficState) -> float:
        C: Set[int] = set(state.conflicts_of(i)) if state.conflicts_of(i) else set()
        if not C:
            return 0.0

        r_best = min(state.rank_of(j) for j in C)

        return 1.0 - (r_best / 2.0)      # priority rank 0 → 1.0

    def trajectory(self, i, state: TrafficState) -> float: # traight = 0, left/right = 1
        t = state.turn_type_of(i)

        return 0.0 if t == "straight" else 1.0

    def teammates(self, i, state):
        others = state.non_conflicting(i) or []
        if not others:
            return 1.0

        eta_i = state.eta(i)
        if eta_i is None:
            return 1.0  # worst

        diffs = []
        for j in state.non_conflicting(i):
            eta_j = state.eta(j)
            if eta_j is not None:
                diffs.append(abs(eta_i - eta_j))
        if not diffs:
            return 1.0

        min_diff = min(diffs)
        return min_diff / (1.0 + min_diff)

    def score_vehicle(self, i, state: TrafficState) -> float:
        return (self.lambda1 * self.wait_self(i, state) +
                self.lambda2 * self.wait_other(i, state) +
                self.lambda3 * self.priority_self(i, state) +
                self.lambda4 * self.priority_other(i, state) +
                self.lambda5 * self.trajectory(i, state) +
                self.lambda6 * self.teammates(i, state))

    def select_leader_and_compatible_group(
            self,
            state: TrafficState,
            candidate_ids: List[str]
    ) -> Tuple[str, List[str]]:
        """
        Return (leader_id, group_ids) — group contains leader + any non-conflicting candidates.
        """
        scores = {rid: self.score_vehicle(rid, state) for rid in candidate_ids}
        leader_id = min(scores, key=scores.get)
        group = [rid for rid in candidate_ids if rid == leader_id or rid in state.non_conflicting(leader_id)]
        return leader_id, group

    def assign_vehicle_meta(vehicles, center, traj_policy) -> dict[str, VehicleMeta]:
        """
        Assign VehicleMeta for each vehicle in this intersection center.
        - Sets priority_class and rank
        - Sets intersection_id
        - Looks up incoming/outgoing from policy
        """
        meta = {}
        priority_map = {"robot1": "ambulance", "robot7": "ambulance",
                        "robot2": "bus", "robot5": "bus"}

        rank_map = {"ambulance": 0, "bus": 1, "private": 2}

        for rid in vehicles:
            vm = VehicleMeta()
            vm.priority_class = priority_map.get(rid, "private")
            vm.rank = rank_map[vm.priority_class]
            vm.intersection_id = center

            # fill incoming/outgoing using policy
            for (rr, c, inc), out in traj_policy.items():
                if rr == rid and c == center:
                    vm.incoming = inc
                    vm.outgoing = out
                    # optional: vm.turn_type = TURN_FROM_IN_OUT.get((inc, out), "straight")
                    break

            meta[rid] = vm

        return meta

    def components(self, rid: str, state: TrafficState) -> dict:
        """Return each scalar term used in the score (for logging & debugging)."""
        comp = {}
        comp["wait_self"] = float(self.wait_self(rid, state))
        comp["wait_other"] = float(self.wait_other(rid, state))
        comp["prio_self"] = float(self.priority_self(rid, state))
        comp["prio_other"] = float(self.priority_other(rid, state))
        comp["trajectory"] = float(self.trajectory(rid, state))
        comp["teammates"] = float(self.teammates(rid, state))

        # final score
        comp["score"] = (self.lambda1 * comp["wait_self"] +
                         self.lambda2 * comp["wait_other"] +
                         self.lambda3 * comp["prio_self"] +
                         self.lambda4 * comp["prio_other"] +
                         self.lambda5 * comp["trajectory"] +
                         self.lambda6 * comp["teammates"])
        return comp

    def score_vehicle(self, rid, state: TrafficState) -> float:
        return self.components(rid, state)["score"]















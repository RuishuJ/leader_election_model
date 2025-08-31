from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Any
from duckietown_simulator.Model.FIFO_leader_election_manager import compute_intersection_boundaries
from duckietown_simulator.Model.intersection_manager_V2 import IntersectionManager, _extract_bot
from itertools import combinations
import math
from duckietown_simulator.Model.trajectory_policy import SAFE_GROUPS_BY_SIZE, Move


Center = Tuple[float, float]
Point  = Tuple[float, float]
Move = Tuple[str, str] # (incoming, outgoing) in {"N","E","S","W"}

class SafetyManager:
    def __init__(self,intersection_manager: IntersectionManager):
        self.im = intersection_manager

    def compute_sub_zone(self,
                         intersection_centers: List[Center],
                         tile_size: float = 0.6) -> Dict[Center, Dict[str, List[Point]]]:
        """
        Split each control square (tile_size x tile_size) into 4 quadrants:
          Zone A = top-left, Zone B = top-right, Zone C = bottom-left, Zone D = bottom-right.
        Returns corners for each sub-zone as [BL, BR, TR, TL].
        """
        # get the control square corners
        control_boundaries, _ = compute_intersection_boundaries(intersection_centers,
                                                                tile_size = tile_size)

        out: Dict[Center, Dict[str, List[Point]]] = {}

        for center, corners in control_boundaries.items():
            BL, BR, TR, TL = corners
            x_min, y_min = BL
            d = tile_size / 2

            # Zone(A): top_left
            Zone_A = [(x_min, y_min + d), (x_min + d, y_min + d), (x_min + d, y_min + 2 * d), (x_min, y_min + 2 * d)]
            # Zone(B): top_right
            Zone_B = [(x_min + d, y_min + d), (x_min + 2 * d, y_min + d), (x_min + 2 * d, y_min + 2 * d), (x_min + d, y_min + 2 * d)]
            # Zone(C): bottom_left
            Zone_C = [(x_min, y_min), (x_min + d, y_min), (x_min + d, y_min + d), (x_min, y_min + d)]
            # Zone(D): bottom_right
            Zone_D = [(x_min + d, y_min), (x_min + 2 * d, y_min), (x_min + 2 * d, y_min + d), (x_min + d, y_min + d)]

            out[center] = {"Zone A": Zone_A,
                           "Zone B": Zone_B,
                           "Zone C": Zone_C,
                           "Zone D": Zone_D
                           }
        return out

    @staticmethod
    def compute_sub_zone_boundary(
                                  subzones: Dict[Center, Dict[str, List[Point]]]
    ) -> Dict[Center, Dict[str, Dict[str, Tuple[Point, Point]]]]:
        """
        For each sub-zone, return its named edges:
          north_edge, south_edge, east_edge, west_edge
        using the sub-zone's [BL, BR, TR, TL] corners.
        """
        edges_boundaries: Dict[Center, Dict[str, Dict[str, Tuple[Point, Point]]]] = {}

        for center, zones in subzones.items():
            per_center: Dict[str, Dict[str, Tuple[Point, Point]]] = {}
            for name, corners in zones.items():
                bl, br, tr, tl = corners
                per_center[name] = {
                    "south_edge": (bl, br),  # South Boundary
                    "north_edge": (tl, tr),  # N
                    "west_edge": (tl, bl),  # W
                    "east_edge": (tr, br),  # E
                }
            edges_boundaries[center] = per_center
        return edges_boundaries

    @staticmethod
    def rect_edges(corners: List[Point]) -> Dict[str, Tuple[Point, Point]]:
        # corners in CW: BL, BR, TR, TL
        BL, BR, TR, TL = corners
        return {
            "S": (BL, BR),  # South edge
            "E": (BR, TR),  # East edge
            "N": (TR, TL),  # North edge
            "W": (TL, BL),  # West edge
        }

    @staticmethod
    def enumerate_safe_groups_from_robots(moves_by_robot: Dict[str, Move],
                                          SAFE_GROUPS_BY_SIZE: Dict[int, Set[frozenset]],
                                          ) -> List[List[str]]:
        """
        moves_by_robot: {"robot2": ("N","S"), "robot4": ("S","N"), ...}
        Returns all robot ID groups whose moves match a manually-listed SAFE group.
        """
        rids = list(moves_by_robot.keys())
        results: List[List[str]] = []

        for k, safe_groups in SAFE_GROUPS_BY_SIZE.items():
            if k == 0:
                continue
            for rid_group in combinations(rids, k):
                move_group = frozenset(moves_by_robot[r] for r in rid_group)
                if move_group in safe_groups:
                    results.append(list(rid_group))
        return results

    @staticmethod
    def _project_point_to_segment(p: Point, a: Point, b: Point) -> Point:
        ax, ay = a; bx, by = b; px, py = p
        vx, vy = (bx - ax, by - ay)
        L2 = vx * vx + vy * vy
        if L2 <= 1e-12:
            return a
        t = ((px - ax) * vx + (py - ay) * vy) / L2
        t = max(0.0, min(1.0, t))
        return (ax + t * vx, ay + t * vy)

    def _nearest_edge_for_robot_in_center(
        self,
        rid: str,
        robots: Dict[str, Any],
        center_xy: Center,
        boundary: List[Point],
        selection_radius: float,
    ) -> Optional[str]:
        """
        Returns incoming direction "N"/"E"/"S"/"W" by picking the nearest boundary edge;
        None if robot is not in this center zone.
        """
        # short-circuit if outside
        if self.im.distance_to_center(rid, robots, center_xy) > selection_radius:
            return None

        # project to each edge and pick the min distance
        bot = _extract_bot(robots[rid])
        px, py, _ = bot.pose
        best_d = float("inf")
        best_edge = None
        for h, (p1, p2) in self.rect_edges(boundary).items():
            qx, qy = self._project_point_to_segment((px, py), p1, p2)
            d = math.hypot(px - qx, py - qy)
            if d < best_d:
                best_d = d
                best_edge = h
        return best_edge

    def build_moves_for_center(
        self,
        center_xy: Center,
        robots: Dict[str, Any],               # env.robots-like dict: {rid: {..., 'robot': Duckiebot}}
        policy: Dict[Tuple[str, Center, str], str],  # your per-robot policy: (rid, center, incoming)->outgoing
        selection_radius: float = 0.99,
    ) -> Dict[str, Move]:
        """
        For robots currently inside this center's selection zone, build their (incoming, outgoing) moves.
        - 'incoming' is determined by nearest boundary edge (geometry).
        - 'outgoing' comes from your policy[(rid, center, incoming)].
        """
        # who is inside this center's zone now?
        zone_members = self.im.vehicles_in_zone(robots, center_xy, radius = selection_radius)
        if not zone_members:
            return {}

        # boundaries for this center (control square)
        control_boundaries, _ = compute_intersection_boundaries([center_xy], tile_size = 0.6)
        boundary = control_boundaries[center_xy]  # [BL,BR,TR,TL]

        moves_by_robot: Dict[str, Move] = {}
        for rid in zone_members:
            # determine incoming edge by nearest-edge geometry
            nb = self._nearest_edge_for_robot_in_center(rid, robots, center_xy, boundary, selection_radius)
            if nb is None:
                continue
            incoming = nb  # "N","E","S","W"
            # map to outgoing via the robot-specific policy
            outgoing = policy.get((rid, center_xy, incoming))
            if outgoing is None:
                # if policy is missing, print error
                print(f"Policy Missing")
                continue
            moves_by_robot[rid] = (incoming, outgoing)

        return moves_by_robot

    def actions_for_center(
        self,
        center_xy: Center,
        robots: Dict[str, Any],
        policy: Dict[Tuple[str, Center, str], str],
        SAFE_GROUPS_BY_SIZE: Dict[int, Set[frozenset]],
        selection_radius: float = 0.99,
        go: int = 1,
        stop: int = 0,
        cap_group_size: Optional[int] = None,  # e.g., 2 or 3; None = allow any size listed
    ) -> Dict[str, int]:
        """
        Returns {rid: 1 (GO) or 0 (STOP)} for robots inside center_xy selection zone,
        allowing only non-conflicting groups as per SAFE_GROUPS_BY_SIZE, favoring the
        largest feasible group (optionally capped).
        """
        actions: Dict[str, int] = {}

        moves_by_robot = self.build_moves_for_center(center_xy, robots, policy, selection_radius)
        members = list(moves_by_robot.keys())
        if not members:
            return actions

        # enumerate all safe groups among present members
        groups = self.enumerate_safe_groups_from_robots(moves_by_robot, SAFE_GROUPS_BY_SIZE)
        if not groups:
            # no combo allowed → everyone STOP except maybe pick a single if singles are safe
            singles_allowed = SAFE_GROUPS_BY_SIZE.get(1, set())
            # choose any single that is listed as safe
            for rid in members:
                mv = frozenset({moves_by_robot[rid]})
                actions[rid] = go if mv in singles_allowed else stop
            # if you prefer strict freeze when nothing matches, replace above with all STOP
            return actions

        # choose the "best" group: largest size (respecting cap), tie-break by something simple (lexicographic)
        if cap_group_size is not None:
            groups = [g for g in groups if len(g) <= cap_group_size]
            if not groups:
                # fallback like above: only singles allowed under cap
                singles_allowed = SAFE_GROUPS_BY_SIZE.get(1, set())
                for rid in members:
                    mv = frozenset({moves_by_robot[rid]})
                    actions[rid] = go if mv in singles_allowed else stop
                return actions

        groups.sort(key=lambda g: (-len(g), sorted(g)))   # prefer the biggest groups
        go_group = set(groups[0])

        # apply actions: GO to selected group; STOP to all others in zone
        for rid in members:
            actions[rid] = go if rid in go_group else stop

        return actions
    @staticmethod
    def build_conflict_matrix(traj_policy: Dict[Tuple[str, Center, str], str]) -> Dict[str, Set[str]]:
        """
        Build a conflict matrix from a trajectory policy.

        Input:
            traj_policy: dict mapping (robot_id, center_xy, incoming_dir) -> outgoing_dir

        Output:
            conflict_matrix: dict robot_id -> set of robot_ids that conflict with it
                             (considered per-center, pairwise, using SAFE_GROUPS_BY_SIZE[2]).
        """
        conflict_matrix: Dict[str, Set[str]] = defaultdict(set)
        by_center: Dict[Tuple[float, float], Dict[str, Move]] = defaultdict(lambda: defaultdict(set))

        # Group moves by center
        for (rid, center, incoming), outgoing in traj_policy.items():
            by_center[center][rid] = (incoming, outgoing)

        safe_pairs = SAFE_GROUPS_BY_SIZE.get(2, set())

        # Check pairwise moves for conflicts
        for center, moves_by_robot in by_center.items():
            rids = list(moves_by_robot.keys())
            for i, j in combinations(rids, 2):
                move_i = moves_by_robot[i]
                move_j = moves_by_robot[j]

                # Two robots conflict at this center if ALL their (move_a, move_b) combinations
                # are NOT in the safe set (i.e., no safe pairing exists).
                is_conflict = True
                for mi in move_i:
                    for mj in move_j:
                        if frozenset({mi, mj}) in safe_pairs:
                            is_conflict = False
                            break
                    if not is_conflict:
                        break

                if is_conflict:
                    conflict_matrix[i].add(j)
                    conflict_matrix[j].add(i)

        return dict(conflict_matrix)





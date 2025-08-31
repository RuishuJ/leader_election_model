from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional

from duckietown_simulator.Model.intersection_manager_V2 import _extract_bot


class LeaderElectionManager:
    """
    Tracks a leader per intersection. Reselects when the previous leader leaves the zone
    (or after a max hold time). Optionally gates reselection with VTL.
    """

    def __init__(self,
                 centers: List[Tuple[float, float]],
                 exit_radius: float = 0.7) -> None:
        """
        Args:
            centers: list of intersection centers (cx, cy) in WORLD coordinates (origin bottom-left)
            exit_radius: radius used ONLY for 'leader left' detection
        """
        self.centers: List[Tuple[float, float]] = [tuple(map(float, c)) for c in centers]
        self.exit_radius: float = float(exit_radius)
        # Per-center leader store
        self._leader: Dict[Tuple[float, float], Optional[str]] = {c: None for c in self.centers}

    # ---- leader state ----
    def set_leader(self, center: Tuple[float, float], leader_id: Optional[str]) -> None:
        c = (float(center[0]), float(center[1]))
        if c not in self._leader:
            self._leader[c] = None  # allow late-added centers
        self._leader[c] = leader_id

    def get_leader(self, center: Tuple[float, float]) -> Optional[str]:
        c = (float(center[0]), float(center[1]))
        return self._leader.get(c, None)

    def clear_leader(self, center: Tuple[float, float]) -> None:
        c = (float(center[0]), float(center[1]))
        if c in self._leader:
            self._leader[c] = None

    def leader_has_left(
            self,
            agents: Dict[str, Any],
            center: Tuple[float, float],
            leader_id: Optional[str] = None,
            exit_radius: Optional[str] = None,
    ) -> bool:
        """
        Return True iff there is no leader OR the leader is outside the exit_radius.
        """
        c = (float(center[0]), float(center[1]))
        rid = self._leader.get(c) if leader_id is None else leader_id

        if not rid:
            return True  # no leader means "treat as left"

        info = agents.get(rid)
        if info is None:
            return True  # leader not in agents anymore

        try:
            robot = _extract_bot(info)  # your existing helper
            x, y, _ = robot.pose
        except Exception:
            return True  # can't read pose -> assume left

        cx, cy = center
        r = float(exit_radius) if exit_radius is not None else self.exit_radius

        # use squared distance to avoid sqrt
        dx, dy = x - cx, y - cy
        return (dx * dx + dy * dy) > (r * r)

    def update_and_get_exited(self,
                              agents: Dict[str, Any],
                              exit_radius: Optional[float] = None) -> List[Tuple[float, float]]:
        """
        Check all centers; for any whose leader has left, clear it and
        return centers needing reselection (by your separate logic).
        """
        need_reselect: List[Tuple[float, float]] = []
        for c in list(self._leader.keys()):
            if self.leader_has_left(agents, c, exit_radius=exit_radius):
                self._leader[c] = None
                need_reselect.append(c)
        return need_reselect

    # def actions_with_conflict_gate(
    #         self,
    #         center: Tuple[float, float],
    #         vehicles_in_zone: List[str],
    #         *,
    #         waypoints_map: Optional[Dict[str, list]] = None,
    #         samples: Optional[Dict[str, List[Timed_Waypoint]]] = None,
    #         zone_radius: float = 1.5,
    #         dt: float = 0.1,
    #         horizon: float = 2.0,
    #         safety_radius: float = 0.6,
    #         pad: float = 0.2,
    # ) -> Dict[str, int]:
    #     """
    #     Given the CURRENT leader stored for `center`, return {agent_id: 0|1}
    #     where 1=GO, 0=STOP for the vehicles currently in this intersection's zone.
    #
    #     Policy:
    #       - If no leader is set OR leader not in `vehicles_in_zone` → STOP everyone.
    #       - Otherwise, leader=GO. Each other vehicle=GO only if its sampled path
    #         does NOT conflict with the leader's sampled path inside the zone
    #         (radius=zone_radius, with extra pad).
    #       - If samples are missing for an agent, that agent is STOP (conservative).
    #
    #     You can pass precomputed `samples` (dict[id -> [(t,x,y)]]) or a `waypoints_map`
    #     (dict[id -> waypoints in json format]). If `samples` not given, this builds
    #     them from `waypoints_map` for just the vehicles in `vehicles_in_zone`.
    #     """
    #     leader_id = self.get_leader(center)
    #     ids = list(vehicles_in_zone)
    #
    #     # No leader (yet) → STOP all
    #     if not leader_id or (leader_id not in ids):
    #         return {rid: 0 for rid in ids}
    #
    #     # Ensure we have samples
    #     if samples is None:
    #         if waypoints_map is None:
    #             # Cannot decide safely → STOP all non-leaders, allow leader only
    #             acts = {rid: 0 for rid in ids}
    #             acts[leader_id] = 1
    #             return acts
    #         samples = build_samples_from_waypoints(
    #             wp_map=waypoints_map, ids=ids, dt=dt, horizon=horizon
    #         )
    #
    #     # Ask the conflict gate: everyone can GO only if non-conflicting with leader in-zone
    #     acts = non_conflicting_with_priority_in_zone(
    #         center=center,
    #         zone_radius=zone_radius,
    #         priority_id=leader_id,
    #         ids=ids,
    #         samples=samples,
    #         safety_radius=safety_radius,
    #         pad=pad,
    #     )
    #
    #     # Make sure leader stays GO (defensive)
    #     acts[leader_id] = 1
    #     return acts



#####Check
if __name__ == "__main__":
    # --- Minimal runtime test for LeaderElectionManager ---
    import time
    from duckietown_simulator.robot.duckiebot import Duckiebot

    # A tiny test bot that only carries a pose (compatible with _extract_bot)
    class TestDuckiebot(Duckiebot):
        def __init__(self, x: float, y: float, theta: float = 0.0):
            # don't call super(); we just need a pose tuple
            self.pose = (float(x), float(y), float(theta))

    centers = [(0.25, 1.50), (2.25, 1.50)]
    lem = LeaderElectionManager(centers = centers, exit_radius = 0.7)

    # Build a minimal agents dict: id -> {'robot': Duckiebot-like}
    agents = {
        "robot1": {"robot": TestDuckiebot(2.4, 2.0, 0.0)},  # start 0.4m from center (inside R=1.0)
        "robot3": {"robot": TestDuckiebot(4.0, 1.0, 0.0)},  # irrelevant for this demo
    }

    # Manually assign 'robot1' as the leader for this intersection
    lem.set_leader(centers, "robot1")

    def dist(a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

    print("\n[TEST] Leader leave detection (exit_radius=1.0)")
    print("Center:", centers)
    print("Initial leader:", lem.get_leader(centers))

    # Move robot1 outward in a few steps: inside -> just outside -> farther outside
    positions = [
        (2.6, 2.0),   # d=0.6  (inside)
        (3.05, 2.0),  # d=1.05 (just outside -> should trigger 'left')
        (3.20, 2.0),  # d=1.20 (farther outside)
    ]

    for step, (x, y) in enumerate(positions, start=1):
        # update pose
        agents["robot1"]["robot"].pose = (x, y, 0.0)
        d = dist((x, y), centers)

        # Option A: one-shot check
        left = lem.leader_has_left(agents, centers)
        # Option B: bulk update that also clears the leader automatically
        exited_centers = lem.update_and_get_exited(agents)

        # Print status
        print(f"\nStep {step}: robot1 -> pose=({x:.2f}, {y:.2f}), dist={d:.2f}")
        print("  leader_has_left? ", left)
        if exited_centers:
            print("  update_and_get_exited -> centers needing reselection:", exited_centers)
        else:
            print("  update_and_get_exited -> none")

        print("  stored leader now:", lem.get_leader(centers))

        # slow down printing for readability (optional)
        # time.sleep(0.3)

    print("\n[TEST DONE]")


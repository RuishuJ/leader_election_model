#Post Encroachment Time
from __future__ import annotations
from collections import defaultdict, deque
from typing import Dict, Tuple, Optional, List
from typing import Dict, Tuple, Optional, Any
from duckietown_simulator.Model.trajectory_conflict import Timed_Waypoint

ROBOT_LENGTH: float = 0.18
MIN_STATIC_GAP: float = 0.25    # meters
POSITION_TOL: float = 0.12      # meters


Cell = Tuple[int, int]
Timed_Waypoint = Tuple[float, float, float]

class PostEncroachmentTimeMonitor:

    def __init__(
        self,
        *,
        cell_size: float = 0.25,
        pet_threshold: float = 0.8,     # seconds
        max_event_age: float = 3.0,     # seconds
        dt: Optional[float] = None,     # optional fixed step
        max_logged_events: int = 200
    ):
        self.cell_size = float(cell_size)
        self.pet_threshold = float(pet_threshold)
        self.max_event_age = float(max_event_age)
        self.dt = None if dt is None else float(dt)

        # Time bookkeeping
        self.time: float = 0.0

        # Per-robot current cell + enter time
        self._current_cell: Dict[str, Cell] = {}
        self._enter_time: Dict[str, float] = {}

        # Per-cell recent leave events: cell -> deque[(id, t_leave)]
        self._leave_log: Dict[Cell, deque] = defaultdict(lambda: deque())

        # Recent PET events for debugging
        self._pet_events: deque = deque(maxlen=max_logged_events)

    # ---------- public API ----------

    def reset(self) -> None:
        """Clear all state."""
        self.time = 0.0
        self._current_cell.clear()
        self._enter_time.clear()
        self._leave_log.clear()
        self._pet_events.clear()

    # def update_positions(
    #     self,
    #     positions: Dict[str, Tuple[float, float]],
    #     t: Optional[float] = None
    # ) -> Dict[str, Dict]:
    #     """
    #     Update monitor with current positions.
    #
    #     Args:
    #       positions: {robot_id: (x, y)} at the *same* time instant
    #       t: current time (s). If None and dt was set in __init__, time is advanced by dt.
    #
    #     Returns:
    #       alerts: {robot_id: {"pet": float, "cell": Cell, "other_id": str, "should_slow": bool}}
    #               Only includes robots that triggered a PET event on this update.
    #     """
    #     # Advance / set time
    #     if t is None:
    #         if self.dt is None:
    #             raise ValueError("Provide `t` or set `dt` in the constructor.")
    #         self.time += self.dt
    #         t_now = self.time
    #     else:
    #         self.time = float(t)
    #         t_now = self.time
    #
    #     alerts: Dict[str, Dict] = {}
    #
    #     # Determine current cell for each robot
    #     for rid, (x, y) in positions.items():
    #         cell = self.xy_to_cell(x, y)
    #
    #         prev_cell = self._current_cell.get(rid)
    #         if prev_cell is None:
    #             # First observation: mark enter
    #             self._current_cell[rid] = cell
    #             self._enter_time[rid] = t_now
    #             continue
    #
    #         if cell == prev_cell:
    #             # Still in same cell → nothing to do
    #             continue
    #
    #         # Robot leaves prev_cell at t_now
    #         self.log_leave(prev_cell, rid, t_now)
    #
    #         # Robot enters new cell
    #         self._current_cell[rid] = cell
    #         self._enter_time[rid] = t_now
    #
    #         # Compute PET against most recent *other* leave in this cell
    #         other_id, t_leave_other = self.latest_other_leave(cell, rid, t_now)
    #         if other_id is not None and t_leave_other is not None:
    #             pet = t_now - t_leave_other
    #             if pet >= 0.0:
    #                 evt = {
    #                     "a_left_id": other_id,
    #                     "b_enter_id": rid,
    #                     "cell": cell,
    #                     "t_leave": t_leave_other,
    #                     "t_enter": t_now,
    #                     "pet": pet,
    #                 }
    #                 self._pet_events.append(evt)
    #
    #                 if pet < self.pet_threshold:
    #                     alerts[rid] = {
    #                         "pet": pet,
    #                         "cell": cell,
    #                         "other_id": other_id,
    #                         "should_slow": True
    #                     }
    #
    #     # Prune old leaves
    #     self.prune_old_leaves(t_now)
    #
    #     return alerts

    # def get_recent_pet_events(self) -> List[Dict]:
    #     """Return a list of recent PET events (for logging/inspection)."""
    #     return list(self._pet_events)

    # ---------- internals ----------

    # def xy_to_cell(self, x: float, y: float) -> Cell:
    #     cs = self.cell_size
    #     return (int(x // cs), int(y // cs))
    #
    # def log_leave(self, cell: Cell, rid: str, t_leave: float) -> None:
    #     dq = self._leave_log[cell]
    #     dq.append((rid, float(t_leave)))
    #
    # def latest_other_leave(
    #     self, cell: Cell, entrant_id: str, t_now: float
    # ) -> Tuple[Optional[str], Optional[float]]:
    #     """
    #     Find the most recent leave event in `cell` by someone other than `entrant_id`
    #     within the `max_event_age` window.
    #     """
    #     dq = self._leave_log.get(cell)
    #     if not dq:
    #         return (None, None)
    #
    #     # Walk from the end (most recent)
    #     cutoff = t_now - self.max_event_age
    #     for rid, t_leave in reversed(dq):
    #         if t_leave < cutoff:
    #             break
    #         if rid != entrant_id:
    #             return rid, t_leave
    #     return (None, None)
    #
    # def prune_old_leaves(self, t_now: float) -> None:
    #     cutoff = t_now - self.max_event_age
    #     to_del = []
    #     for cell, dq in self._leave_log.items():
    #         # Pop from left while too old
    #         while dq and dq[0][1] < cutoff:
    #             dq.popleft()
    #         if not dq:
    #             to_del.append(cell)
    #     for c in to_del:
    #         del self._leave_log[c]

    @staticmethod
    def pet_from_samples(
                         a: List[Timed_Waypoint],
                         b: List[Timed_Waypoint],
                         safety_radius: float = 0.25,
                         max_time: Optional[float] = 0.5) -> Optional[Tuple[float, float, float]]:
        """
        Post-Encroachment Time between two time-aligned trajectories a and b.

        Returns the minimum |ta - tb| over all sample pairs whose spatial distance
        is within safety_radius. If no such pair exists, returns None.

        Args:
            a, b: Lists of (t, x, y) sorted by ascending t.
            safety_radius: spatial proximity (meters) used to declare an encroachment.
            max_time: optional cap; pairs with ta>max_time or tb>max_time can be skipped.

        Complexity: O(len(a)*len(b)). For small horizons this is fine.
        """
        if not a or not b:
            return None

        r2 = safety_radius * safety_radius
        best_dt: Optional[float] = None
        best_ta: Optional[float] = None
        best_tb: Optional[float] = None

        for ta, xa, ya in a:
            if max_time is not None and ta > max_time:
                break
            for tb, xb, yb in b:
                if max_time is not None and tb > max_time:
                    break
                dx, dy = xa - xb, ya - yb
                if dx * dx + dy * dy <= r2:
                    dt = abs(tb - ta)
                    if best_dt is None or dt < best_dt:
                        best_dt = dt
                        best_ta = ta
                        best_tb = tb
                        if best_dt <= 1e-9:
                            return (0.0, best_ta, best_tb) # can't do better
        if best_dt is None:
            return None
        return (best_dt, best_ta, best_tb)

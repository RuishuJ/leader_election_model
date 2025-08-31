import math
from typing import Any, Dict, List, Tuple, Optional
from duckietown_simulator.world.map import Map
from duckietown_simulator.robot.duckiebot import Duckiebot, create_duckiebot
from duckietown_simulator.utils.trajectory_utils import create_map_from_json

Point  = Tuple[float, float]
Center = Tuple[float, float]

def find_intersection_centers(map_instance, *, x_offset=0.30, y_offset=1.5):
    H = map_instance.height_tiles
    W = map_instance.width_tiles
    intersection_centers = []
    for r in range(H):
        for c in range(W):
            tid = map_instance.get_tile_type(r, c)
            if 12 <= tid <= 15:
                x = c + x_offset
                y = H - r - y_offset
                intersection_centers.append((x, y))
    if len(intersection_centers) == 2:
        i_right = 0 if intersection_centers[0][0] > intersection_centers[1][0] else 1
        x, y = intersection_centers[i_right]
        intersection_centers[i_right] = (x - 1.2, y)

    return intersection_centers

# Heading direction v.s. Boundary location
def _cardinal_from_theta(theta: float) -> str:
    a = math.atan2(math.sin(theta), math.cos(theta))  # normalize to (-pi, pi]
    if -math.pi / 4 <= a <  math.pi / 4:   return "E"
    if  math.pi / 4 <= a <  3*math.pi / 4: return "N"
    if -3*math.pi / 4 <= a < -math.pi / 4: return "S"
    return "W"

_OPPOSITE = {"N":"S","S":"N","E":"W","W":"E"}

def _extract_bot(info: Any) -> Duckiebot:
    """Accept either a Duckiebot directly, or a dict with a 'robot' key."""
    if isinstance(info, Duckiebot):
        return info
    if isinstance(info, dict) and 'robot' in info and isinstance(info['robot'], Duckiebot):
        return info['robot']
    raise TypeError(
        f"Expected Duckiebot or dict with 'robot': Duckiebot, got {type(info)}. "
        "Pass env.robots or a dict mapping id -> {'robot': Duckiebot}."
    )

class IntersectionManager:
    def __init__(self, map_object: Map, default_radius: float = 0.99):
        self.map = map_object
        self.default_radius = float(default_radius)
        self.intersection_centers: List[Tuple[float, float]] = find_intersection_centers(self.map)

    def in_the_intersection_zone(
        self,
        agents: Dict[str, Any],
        intersections_centers: List[tuple[float, float]],
        radius: float = 0.99)-> List[tuple[tuple[float, float], int, list[str]]]:
        """
            Return a list of agent_ids whose robot_state (x,y) lies within radius of zone center.
        """
        zones = []

        for (cx, cy) in intersections_centers:
            inside = []
            for agent_id, info in agents.items():
                robot = _extract_bot(info)
                x, y, theta = robot.pose
                print(agent_id, type(x), type(y), "pose:", robot.pose)
                if math.hypot(x - cx, y - cy) <= radius:
                    inside.append(agent_id)
            zones.append(((cx, cy), len(inside), inside))
        return zones

    def candidate_position(self,
                           agents: Dict[str, Any],
                           intersection_centers: Tuple[float, float],
                           radius: Optional[float] = None
                           )-> List[Dict[str, Tuple[float, float]]]:
        """
        Positions of vehicles inside the zone of a single intersection center.
        Returns: [{"id": <agent_id>, "pos": (x, y)}, ...]
        #For FIFO leader election
        """
        r = self.default_radius if radius is None else float(radius)
        position_results: List[Dict[str, Tuple[float, float]]] = []
        cx, cy = intersection_centers
        for aid, info in agents.items():
            robot = _extract_bot(info)
            x, y, theta = robot.pose
            if math.hypot(x - cx, y - cy) <= r:
                position_results.append({"id": aid, "pos": (float(x), float(y))})
        return position_results

    def compass (self, angle: float) -> str:
        a = math.atan2(math.sin(angle), math.cos(angle)) # normalize to (-pi, pi]
        if -math.pi/4 <= a < math.pi/4:
            return "E"
        elif math.pi/4 <= a < 3*math.pi/4:
            return "N"
        elif -3*math.pi/4 <= a < -math.pi/4:
            return "S"
        else:
            return "W"

# For each 3-way intersection center, select at most 3 candidate vehicles.
    def vehicle_in_the_front(self,
                             agents: Dict[str, Any],
                             centers: Optional[List[Tuple[float, float]]] = None,
                             radius: Optional[float] = None,
                             ) -> List[Tuple[Tuple[float, float], List[Dict[str, Any]]]]:
        centers = self.intersection_centers if centers is None else centers
        r = self.default_radius if radius is None else float(radius)
        front_veh: List[Tuple[Tuple[float, float], List[Dict[str, Any]]]] = []

        for (cx, cy) in centers:
            # keep the nearest vehicle per sector
            best_per_sector: Dict[str, Dict[str, Any]] = {}

            for aid, info in agents.items():
                robot = _extract_bot(info)
                x, y, _ = robot.pose
                dx, dy = (x - cx), (y - cy)
                dist = math.hypot(dx, dy)
                if dist > r:
                    continue  # not in this intersection zone

                sector = self.compass(math.atan2(dy, dx))
                cand = {"id": aid, "pos": (float(x), float(y)), "dist": float(dist), "sector": sector}

                cur = best_per_sector.get(sector)
                if cur is None or cand["dist"] < cur["dist"]:
                    best_per_sector[sector] = cand

            # take up to three sector-winners, sorted by distance (closest first)
            winners = sorted(best_per_sector.values(), key=lambda c: c["dist"])[:3]
            front_veh.append(((cx, cy), winners))

        return front_veh

    def winners_for_center(
        self,
        agents: Dict[str, Any],
        center: Tuple[float, float],
        radius: Optional[float] = None,
        restrict_to: Optional[set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Sector winners for a single center. Optionally restrict to a set of ids."""
        fl = self.vehicle_in_the_front(agents, centers=[center], radius=radius)
        _, winners = fl[0] if fl else (center, [])
        if restrict_to is not None:
            winners = [w for w in winners if w["id"] in restrict_to]
        return winners

    def winners_by_zone(
            self,
            agents: Dict[str, Any],
            zones: List[Tuple[Tuple[float, float], int, List[str]]],
            radius: Optional[float] = None,
            filter_to_zone_members: bool = True,
    ) -> Dict[Tuple[float, float], List[Dict[str, Any]]]:
        """
        Map each zone center -> its (≤3) sector winners, computed independently.
        If filter_to_zone_members=True, winners are intersected with that zone's 'vehicles' list.
        """
        out: Dict[Tuple[float, float], List[Dict[str, Any]]] = {}
        for (cx, cy), _, vehicles in zones:
            if not vehicles:
                out[(cx, cy)] = []
                continue
            restrict = set(vehicles) if filter_to_zone_members else None
            out[(cx, cy)] = self.winners_for_center(
                agents, (cx, cy), radius=radius, restrict_to=restrict
            )
        return out

    def _extract_ids_from_winners(self, winners):
        ids = []
        for w in winners:
            if isinstance(w, str):
                ids.append(w)
            elif isinstance(w, dict):
                if isinstance(w.get("id"), str):
                    ids.append(w["id"])
                elif isinstance(w.get("agent_id"), str):
                    ids.append(w["agent_id"])
                elif isinstance(w.get("robot_id"), str):
                    ids.append(w["robot_id"])
        # de-dup, preserve order
        seen = set();
        out = []
        for i in ids:
            if i not in seen:
                seen.add(i);
                out.append(i)
        return out

    def distance_to_center(
            self,
            rid: str,
            robots: Dict[str, Any],
            center: Tuple[float, float]
    ) -> float:
        """
        Euclidean distance of robot `rid` to `center` (cx, cy).

        Args:
            rid: robot id key in `robots`
            robots: dict like env.robots: {rid: {... "robot": Duckiebot-like ...}}
            center: (cx, cy) world coordinates of the intersection center

        Returns:
            float distance in meters. If pose is unavailable, returns +inf.
        """
        info = robots.get(rid)
        if info is None:
            return float("inf")
        try:
            bot = _extract_bot(info)
            x, y, _ = bot.pose
            cx, cy = center
            dx, dy = (x - cx), (y - cy)
            return (dx * dx + dy * dy) ** 0.5
        except Exception:
            return float("inf")

    def _edge_segments_from_boundary(self, boundary: List[Point]) -> Dict[str, Tuple[Point, Point]]:
        """boundary is ordered CW: [BL, BR, TR, TL]."""
        BL, BR, TR, TL = boundary
        return {"S": (BL, BR), "E": (BR, TR), "N": (TR, TL), "W": (TL, BL)}

    # def nearest_boundary_for_robot(
    #     self,
    #         rid: str,
    #         robots: Dict[str, Any],
    #         centers_by_name: Dict[str, Center],                  # {"center0": (cx,cy), ...}
    #         control_boundaries: Dict[Center, List[Point]],       # {(cx,cy): [BL,BR,TR,TL], ...}
    #     *,
    #         selection_radius: float = 0.99,
    #         closest_point_on_heading_edge: object = None,                  # pass your function
    # ) -> Optional[Dict[str, Any]]:
    #     """
    #     Inside selection zone: determine
    #       - nearest boundary edge (incoming_by_geometry)
    #       - heading_cardinal from robot.theta
    #       - incoming_from_heading = opposite(heading_cardinal)
    #     """
    #     info = robots.get(rid)
    #     if info is None:
    #         return None
    #     bot = _extract_bot(info)
    #     px, py, th = bot.pose
    #
    #     # 1) find which center we’re in (within selection_radius)
    #     best_center_name = None
    #     best_center_xy: Optional[Center] = None
    #     best_center_d = float("inf")
    #     for name, (cx, cy) in centers_by_name.items():
    #         d = math.hypot(px - cx, py - cy)
    #         if d < best_center_d and d <= selection_radius:
    #             best_center_name, best_center_xy, best_center_d = name, (cx, cy), d
    #     if best_center_xy is None:
    #         return None
    #
    #     boundary = control_boundaries.get(best_center_xy)
    #     if not boundary:
    #         return None
    #
    #     # 2) nearest edge by geometry
    #     best = None  # (distance, edge_label, closest_point)
    #     for edge_label in ("N", "E", "S", "W"):
    #         if closest_point_on_heading_edge is not None:
    #             qx, qy = closest_point_on_heading_edge(boundary, edge_label, (px, py))
    #         else:
    #             p1, p2 = self._edge_segments_from_boundary(boundary)[edge_label]
    #             qx, qy = self._project_point_to_segment((px, py), p1, p2)
    #         d = math.hypot(px - qx, py - qy)
    #         if (best is None) or (d < best[0]):
    #             best = (d, edge_label, (qx, qy))
    #     dist_nearest, incoming_by_geometry, entry_point = best
    #
    #
    #     # 3) incoming from heading (your statement): if heading is S, comes from N, etc.
    #     heading_cardinal = _cardinal_from_theta(th)
    #     incoming_from_heading = _OPPOSITE[heading_cardinal]
    #
    #
    #     # 4) consistency check (they should typically match)
    #     consistent = (incoming_by_geometry == incoming_from_heading)
    #
    #     return {
    #         "center_name": best_center_name,
    #         "center_xy": best_center_xy,
    #         "incoming_by_geometry": incoming_by_geometry,   # edge closest in the box
    #         "incoming_from_heading": incoming_from_heading, # opposite of heading
    #         "heading_cardinal": heading_cardinal,           # robot’s pointing dir
    #         "entry_point": entry_point,                     # on that nearest edge
    #         "distance_to_edge": dist_nearest,
    #         "consistent": consistent,
    #     }


    # # fallback projector if you don't pass your own
    # def _project_point_to_segment(p: Point, a: Point, b: Point) -> Point:
    #     ax, ay = a;
    #     bx, by = b;
    #     px, py = p
    #     vx, vy = (bx - ax, by - ay)
    #     L2 = vx * vx + vy * vy
    #     if L2 <= 1e-12:
    #         return a
    #     t = ((px - ax) * vx + (py - ay) * vy) / L2
    #     t = max(0.0, min(1.0, t))
    #     return (ax + t * vx, ay + t * vy)

    # def sort_by_distance_to_center(
    #         self,
    #         rids: list[str],
    #         robots: Dict[str, Any],
    #         center: Tuple[float, float]
    # ) -> list[str]:
    #     """Return rids sorted nearest-first to `center`."""
    #     return sorted(rids, key = lambda r: self.distance_to_center(r, robots, center))

    def vehicles_in_zone(
            self,
            robots: Dict[str, Any],
            center: Tuple[float, float],
            *,
            radius: float
    ) -> List[str]:
        """
        Return the list of robot ids currently inside the given (center, radius) zone.
        """
        zones = self.in_the_intersection_zone(robots, [center], radius=radius)
        if not zones:
            return []
        _center, _r, ids = zones[0]
        return list(ids)

    # def is_zone_empty(
    #         self,
    #         robots: Dict[str, Any],
    #         center: Tuple[float, float],
    #         *,
    #         radius: float
    # ) -> bool:
    #     """True iff no vehicles are currently inside the given zone."""
    #     return len(self.vehicles_in_zone(robots, center, radius=radius)) == 0

# # Test
# #####Code for test
# class TestDuckiebot(Duckiebot):
#     def __init__(self, x: float, y: float, theta: float = 0.0, speed: float = 0.0):
#         # Don't call super().__init__; we only need a pose for these utilities.
#         self.pose = (float(x), float(y), float(theta))
#         self.speed = float(speed)
#         self.v = float(speed)
#
#     def build_test_agents(raw: Dict[str, Any]) -> Dict[str, Any]:
#         out: Dict[str, Any] = {}
#         for aid, spec in raw.items():
#             if isinstance(spec, Duckiebot) or (isinstance(spec, dict) and isinstance(spec.get('robot'), Duckiebot)):
#                 bot = spec if isinstance(spec, Duckiebot) else spec['robot']
#                 x, y, th = bot.pose
#             else:
#                 src = spec.get('robot_state', spec) if isinstance(spec, dict) else {}
#                 if 'pose' in src:
#                     x, y, th = src['pose']
#                 else:
#                     x = src.get('x'); y = src.get('y'); th = src.get('theta', 0.0)
#                     if x is None or y is None:
#                         raise KeyError(f"Missing x/y for agent '{aid}'. Provide x,y,(theta) or pose/robot.")
#                 bot = TestDuckiebot(x, y, th)
#             out[aid] = {'robot': bot, 'robot_state': {'x': float(x), 'y': float(y), 'theta': float(th)}}
#         return out


def main():
    # load the map
    map_object = create_map_from_json("/Users/ruijiang/PycharmProjects/gym-tsl-duckietown/duckietown_simulator/assets/maps/road_network.json")
    # obtain centers
    intersections = find_intersection_centers(map_object)

    # ###### code for test
    # raw_agents = {
    #     "robot1": {"robot_state": {"x": 1.0, "y": 2.0, "theta": 0.0}},
    #     "robot2": {"robot_state": {"x": 1.0, "y": 3.0, "theta": 0.0}},
    #     "robot3": {"robot_state": {"x": 1.0, "y": 1.5, "theta": 0.0}},
    #     "robot4": {"robot_state": {"x": 1.5, "y": 2.5, "theta": 0.0}},
    #     "robot5": {"robot_state": {"x": 0.0, "y": 3.5, "theta": 0.0}},
    #     "robot6": {"robot_state": {"x": 3.7, "y": 2.0, "theta": 0.0}},
    #     "robot7": {"robot_state": {"x": 3.7, "y": 1.5, "theta": 0.0}},
    #     "robot8": {"robot_state": {"x": 2.7, "y": 2.0, "theta": 0.0}},
    #     "robot9": {"robot_state": {"x": 3.2, "y": 3.5, "theta": 0.0}},
    #     "robot10": {"robot_state": {"x": 3.2, "y": 3.0, "theta": 0.0}},
    # }
    #
    # agents = TestDuckiebot.build_test_agents(raw_agents)
    # #######

####Code for Env
    # agents dict
    agents = {
        #"robot1": {"robot_state": {"x": 1.0, "y": 2.0, "theta": 0.0}},
        #"robot2": {"robot_state": {"x": 5.0, "y": 2.0, "theta": 0.0}},
    }

    for agent_id, st in agents.items():
        robot = create_duckiebot(st['x'], st['y'], st['theta'])
        agents[agent_id] = {
            'robot':       robot,     # store the full Duckiebot
            "robot_state": st,
        }
####

    radius = 0.99
    # are they in the zone?

    im = IntersectionManager(map_object, default_radius=0.99)

    # zone_status = im.in_the_intersection_zone(agents, intersections, radius)
    # for (cx, cy), count, members in zone_status:
    #     print(f"Zone({cx},{cy}): {count} agents: {members}")

    zone_status = im.in_the_intersection_zone(agents, intersections, radius)
    front_lists = im.vehicle_in_the_front(agents, centers=im.intersection_centers, radius=0.99)

    for (cx, cy), count, members in zone_status:
        print(f"Zone({cx},{cy}): {count} agents: {members}")

    # positions in a specific intersection’s zone
    for center in im.intersection_centers:
        print(center, im.candidate_position(agents, center))

    for (cx, cy), winners in front_lists:
        print(f"Intersection @ ({cx:.2f}, {cy:.2f}) -> {len(winners)} candidate(s)")
        for w in winners:
            print(f"  {w['sector']}: {w['id']}  pos={w['pos']}  d={w['dist']:.2f}")


if __name__ == "__main__":
    main()
    print("End of program")









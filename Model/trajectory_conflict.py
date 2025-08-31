import math, json
from typing import List, Tuple, Optional, Dict, Iterable, Any
from pathlib import Path
from duckietown_simulator.Model.intersection_manager_V2 import _extract_bot


Waypoint = Tuple[float, float, float]          # (x, y, speed)
Timed_Waypoint  = Tuple[float, float, float]   # (t, x, y)

############## for test only
def normalize_waypoints(obj: any, default_speed: float = 0.5) -> List[Tuple[float, float, float]]:
    """
    Convert common JSON shapes to [(x, y, speed), ...].
    Accepts:
      - list of (x, y, speed) or [x, y, speed]
      - list of (x, y) → adds default_speed
      - list of dicts {'x','y', ['speed'|'v']}
      - dict with key 'waypoints' → value is one of the above
      - dict of index->dict (or index->list) → values extracted (keys sorted if numeric)
    """
    if obj is None:
        return []

    # unwrap container dict
    if isinstance(obj, dict):
        if "waypoints" in obj:
            obj = obj["waypoints"]
        else:
            # dict of index -> waypoint
            try:
                obj = [v for k, v in sorted(obj.items(), key=lambda kv: float(kv[0]))]
            except Exception:
                obj = list(obj.values())

    if not isinstance(obj, list):
        raise TypeError(f"Waypoints must be a list or dict; got {type(obj)}")

    out: List[Waypoint] = []
    for i, w in enumerate(obj):
        if isinstance(w, dict):
            x = w.get("x"); y = w.get("y")
            v = w.get("speed", w.get("v", default_speed))
            if x is None or y is None:
                # skip malformed entries
                continue
            out.append((float(x), float(y), float(v)))
        elif isinstance(w, (list, tuple)):
            if len(w) == 3:
                x, y, v = w
                out.append((float(x), float(y), float(v)))
            elif len(w) == 2:
                x, y = w
                out.append((float(x), float(y), float(default_speed)))
        # else: silently ignore unsupported types
    return out

def load_waypoints_json(pathlike) -> list:
    path = Path(pathlike)
    with path.open("r") as f:
        return json.load(f)

def build_waypoints_map(trajectory_files: dict) -> dict:
    """
    trajectory_files: {robot_id: PathLike or str}
    returns: {robot_id: waypoints_list}
    """
    out: Dict[str, Any] = {}
    for rid, pp in trajectory_files.items():
        try:
            out[rid] = load_waypoints_json(pp)
        except Exception:
            # keep robust in tests even if a file is missing
            out[rid] = {"waypoints": []}
    return out

# ###### for trajectory_conflict.py
# trajectory_files = {
#     "robot1": "../../data/exp_traj_1.json",
#     "robot2": "../../data/exp_traj_2.json",
#     "robot3": "../../data/exp_traj_3.json",
# }
# ######

##### #for test
trajectory_files = {
    "robot1": "../data/looped_trajectory_11.json",
    "robot2": "../data/looped_trajectory_2.json",
    "robot3": "../data/looped_trajectory_3.json",
    "robot4": "../data/looped_trajectory_4.json",
    "robot5": "../data/looped_trajectory_5.json",
    "robot6": "../data/looped_trajectory_6.json",
}
#####

# Build the dict the VTL code expects:
# waypoints_map = {rid: load_waypoints_json(path) for rid, path in trajectory_files.items()}
waypoints_map = build_waypoints_map(trajectory_files)
# def load_waypoints_json(path: str) -> List[Tuple[float,float,float]]:
#     data = json.load(open(path, "r"))
#     wps = data["waypoints"]
#     return [(float(w["x"]), float(w["y"]), float(w.get("speed", 1.0))) for w in wps]

# def time_parameterise_waypoints(waypoints, v_floor=0.01) -> List[Timed_Waypoint]:
#     # waypoints: [(x,y,speed), ...] -> [(t,x,y), ...]
#     t, out = 0.0, []
#     if not waypoints:
#         return out
#     out.append((0.0, waypoints[0][0], waypoints[0][1]))
#     for i in range(1, len(waypoints)):
#         x0,y0,v0 = waypoints[i-1]
#         x1,y1,_  = waypoints[i]
#         seg_len = math.hypot(x1-x0, y1-y0)
#         t += seg_len / max(abs(v0), v_floor)
#         out.append((t, x1, y1))
#     return out

########### for test only
def time_parameterise_waypoints(waypoints, v_floor=0.01) -> List[Timed_Waypoint]:
    wps = normalize_waypoints(waypoints)  # normalize for test
    t, out = 0.0, []
    if not wps:
        return out
    out: List[Timed_Waypoint] = [(0.0, wps[0][0], wps[0][1])]
    for i in range(1, len(wps)):
        x0, y0, v0 = wps[i - 1]
        x1, y1, _  = wps[i]
        seg_len = math.hypot(x1 - x0, y1 - y0)
        t += seg_len / max(abs(v0), v_floor)
        out.append((t, x1, y1))
    return out
############

def sample_trajectory(timed_point: List[Timed_Waypoint], dt: float = 0.1, horizon: float = 2.0):
    if not timed_point:
        return []
    samples, i = [], 0
    T = min(horizon, timed_point[-1][0]) # set T to the time of last point
    t = 0.0
    while t <= T + 1e-9:
        while i + 1 < len(timed_point) and t > timed_point[i + 1][0]:
            i += 1
        if i + 1 >= len(timed_point):
            samples.append((t, timed_point[-1][1], timed_point[-1][2]))
            break
        t0, x0, y0 = timed_point[i]
        t1, x1, y1 = timed_point[i + 1]
        u = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        x = x0 + u * (x1 - x0)
        y = y0 + u * (y1 - y0)
        samples.append((t, x, y))
        t += dt
    return samples

def build_samples_from_waypoints(
    wp_map: Dict[str, List[Waypoint]],
    ids: Optional[Iterable[str]] = None,
    dt: float = 0.1,
    horizon: float = 2.0,
) -> Dict[str, List[Timed_Waypoint]]:
    if not isinstance(wp_map, dict):
        raise TypeError(
            f"wp_map must be dict[id -> list[(x,y,speed)]], got {type(wp_map).__name__}"
        )
    """{id: [(x,y,speed), ...]} -> {id: [(t,x,y) ...]} sampled at fixed dt."""
    ids = list(ids) if ids is not None else list(wp_map.keys())
    out: Dict[str, List[Timed_Waypoint]] = {}
    for rid in ids:
        tp = time_parameterise_waypoints(wp_map.get(rid, []))
        out[rid] = sample_trajectory(tp, dt=dt, horizon=horizon)
    return out

##########for test
# def build_samples_from_waypoints(wp_map: Dict[str, Any],
#                                  ids: List[str],
#                                  dt: float,
#                                  horizon: float):
#     samples: Dict[str, List[Timed_Waypoint]] = {}
#     for rid in ids:
#         tp = time_parameterise_waypoints(wp_map.get(rid, []))
#         if not tp:          # <<< skip if no usable waypoints
#             continue
#         samples[rid] = sample_trajectory(tp, dt=dt, horizon=horizon)
#     return samples
#############

def normalise_wps(raw) -> List[Tuple[float,float,float]]:
    """Accepts your JSON shapes; returns [(x,y,speed), ...]."""
    if isinstance(raw, dict) and "waypoints" in raw:
        raw = raw["waypoints"]
    out=[]
    for w in raw or []:
        if isinstance(w, dict):
            out.append((float(w["x"]), float(w["y"]), float(w.get("speed", 1.0))))
        else:
            x, y = w[:2]
            v = w[2] if len(w) >= 3 else 1.0
            out.append((float(x), float(y), float(v)))
    return out

def closest_segment_to_point(points: List[Tuple[float,float]], qx: float, qy: float) -> Tuple[int, float]:
    """
    Return (i, u) for the closest segment P[i] -> P[i+1] and param u in [0,1].
    Assumes len(points) >= 2. Uses Euclidean distance in world coords.
    """
    best_i, best_u, best_d2 = 0, 0.0, float('inf')

    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        vx, vy = (x1 - x0), (y1 - y0)
        seg_len2 = vx * vx + vy * vy
        if seg_len2 <= 1e-9:
            u = 0.0
            px, py = x0, y0
        else:
            wx, wy = qx - x0, qy - y0
            u = max(0.0, min(1.0, (wx * vx + wy * vy ) / seg_len2))
            px, py = x0 + u * vx, y0 + u * vy
        d2 = (px - qx) ** 2 + (py - qy) ** 2
        if d2 < best_d2:
            best_d2, best_i, best_u = d2, i, u
    return best_i, best_u

def time_from_index(wps: List[Tuple[float,float,float]], i: int, u: float, v_floor = 0.01) -> List[Timed_Waypoint]:
    """Build timed waypoints starting at the projection inside segment i with param u."""
    if len(wps) < 2:
        return []
    x0, y0, v0 = wps[i]
    x1, y1, v1 = wps[i + 1]
    # current point at u
    cx, cy = x0 + u *(x1 - x0), y0 + u * (y1 - y0)
    # residual of current segment then subsequent segments
    out: List[Timed_Waypoint] = [(0.0, cx, cy)]
    t = 0.0
    # remaining part of current segment
    seg_len = math.hypot(x1 - x0, y1 - y0)
    if seg_len > 1e-9:
        remain = (1.0 - u) * seg_len
        t += remain / max(abs(v0), v_floor)
        out.append((t, x1, y1))
    # rest of the segments
    for k in range(i + 1, len(wps) - 1):
        a,b = wps[k], wps[k+1]
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        t += seg / max(abs(a[2]), v_floor)
        out.append((t, b[0], b[1]))
    return out

def sample_txyz(tp: List[Timed_Waypoint], dt = 0.1, horizon = 2.5) -> List[Timed_Waypoint]:
    if not tp:
        return []
    T = min(horizon, tp[-1][0])
    samples: List[Timed_Waypoint] = []
    idx = 0
    t = 0.0
    while t <= T + 1e-9:
        while idx + 1 < len(tp) and t > tp[idx + 1][0]:
            idx += 1
        if idx+1 >= len(tp):
            samples.append((t, tp[-1][1], tp[-1][2]))
            break
        t0, x0, y0 = tp[idx]
        t1, x1, y1 = tp[idx + 1]
        u = 0.0 if t1 == t0 else (t - t0)/(t1 - t0)
        x = x0 + u * (x1 - x0)
        y = y0 + u * (y1 - y0)
        samples.append((t, x, y))
        t += dt
    return samples

def build_aligned_samples_from_waypoints(
    waypoints_map: Dict[str, Any],
    robots: Dict[str, Any],
    ids: Iterable[str],
    *,
    dt: float = 0.1,
    horizon: float = 2.5,
    v_floor: float = 0.01,
) -> Dict[str, List[Timed_Waypoint]]:
    """
    For each id, start sampling FROM ITS CURRENT POSE projected on its path.
    Returns {id: [(t,x,y), ...]} with t=0 aligned to 'now'.
    """
    out: Dict[str, List[Timed_Waypoint]] = {}
    for rid in ids:
        info = robots.get(rid)
        if info is None:
            continue
        try:
            bot = _extract_bot(info)
            rx, ry, _ = bot.pose
        except Exception:
            continue
        raw = waypoints_map.get(rid)
        if not raw:
            continue
        wps = normalise_wps(raw)
        if len(wps) < 2:
            continue
        pts = [(x, y) for (x, y, _) in wps]
        i,u = closest_segment_to_point(pts, rx, ry)
        tp = time_from_index(wps, i, u, v_floor = v_floor)
        samples = sample_txyz(tp, dt = dt, horizon = horizon)
        out[rid] = samples
    return out


# -------------------------- Same-lane / spacing -------------------------------
def same_lane(im, robot_a, robot_b, *, position_tolerance: float = 0.15) -> bool:
    """
    True iff robots A and B are in the same lane:
      - Same heading sector via im.compass(theta)
      - For N/S lanes, x is similar; for E/W lanes, y is similar.

    Args:
      im: IntersectionManager (must implement im.compass(angle)->{'N','E','S','W'})
      ra, rb: robot objects or info dicts (must have .pose -> (x, y, theta))
      position_tolerance: tolerance for "similar" coordinate (meters)
    """
    A = _extract_bot(robot_a)
    B = _extract_bot(robot_b)

    xa, ya, tha = A.pose
    xb, yb, thb = B.pose

    dir_a = im.compass(tha)
    dir_b = im.compass(thb)
    if dir_a != dir_b:
        return False
        # If both face N/S, lanes are separated mainly by X; if E/W, by Y.
    if dir_a in ("N", "S"):
        return abs(xa - xb) <= position_tolerance
    else:  # "E" or "W"
        return abs(ya - yb) <= position_tolerance

def _forward_axis_and_sign(direction: str):
    """
    Map compass sector -> (axis, sign) for forward progress.
      E: x, +1   W: x, -1   N: y, +1   S: y, -1
    """
    if direction == "E": return ("x", +1)
    if direction == "W": return ("x", -1)
    if direction == "N": return ("y", +1)
    if direction == "S": return ("y", -1)
    # Fallback
    return ("x", +1)

def same_lane_safety(
    im,
    id_a: str, robot_a,
    id_b: str, robot_b,
    *,
    position_tolerance: float = 0.12,
    min_gap: float = 0.60
 ) -> dict:
    """
    Safety check for two robots in the same lane.
    """
    A = _extract_bot(robot_a)
    B = _extract_bot(robot_b)

    xa, ya, tha = A.pose
    xb, yb, thb = B.pose

    dir_a = im.compass(tha)
    dir_b = im.compass(thb)

    # not same lane -> no constraint here
    if dir_a != dir_b or not same_lane(im, robot_a, robot_b, position_tolerance = position_tolerance):
        return {
            "same_lane": False,
            "behind_id": None,
            "ahead_id": None,
            "gap": None,
            "should_stop": False
        }

    # check lateral alignment: not same lane -> too far away
    if dir_a in ("N", "S"):
        lateral_close = abs(xa - xb) <= position_tolerance
    else:  # "E" or "W"
        lateral_close = abs(ya - yb) <= position_tolerance

    if not lateral_close:
        return {
            "same_lane": False,
            "behind_id": None,
            "ahead_id": None,
            "gap": None,
            "should_stop": False
        }

    # compute forward axis & progress
    axis, sign = _forward_axis_and_sign(dir_a)
    prog_a = (xa if axis == "x" else ya) * sign
    prog_b = (xb if axis == "x" else yb) * sign

    # determine leader/behind by progress along forward axis
    if prog_a == prog_b:
        # perfectly aligned longitudinally: treat as too close
        ahead_id, behind_id = id_a, id_b
        gap = 0.0
    elif prog_a > prog_b:
        ahead_id, behind_id = id_a, id_b
        gap = prog_a - prog_b
    else:
        ahead_id, behind_id = id_b, id_a
        gap = prog_b - prog_a

    should_stop = gap < min_gap

    return {
        "same_lane": True,
        "behind_id": behind_id,
        "ahead_id": ahead_id,
        "gap": gap,
        "should_stop": should_stop
    }



###### Test
def main():
    """
    Quick test for trajectory conflict helpers.

    Usage (synthetic test):
        python trajectory_conflict.py

    Usage (with JSONs):
        python trajectory_conflict.py --json robot1 path/to/traj1.json --json robot2 path/to/traj2.json \
            --center 1.0 1.0 --zone-radius 1.5 --dt 0.1 --horizon 2.0 --safety-radius 0.6 --pad 0.2
    """
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Test trajectory conflict helpers")
    parser.add_argument("--json", nargs=2, action="append",
                        metavar=("ID", "PATH"),
                        help="Add a vehicle by (id, json_path) with {'waypoints':[{'x','y','speed'},...]}")
    parser.add_argument("--center", nargs=2, type=float, default=(0.0, 0.0),
                        help="Intersection center (cx cy). Only used for zone-scoped checks.")
    parser.add_argument("--zone-radius", type=float, default=1.5, help="Intersection zone radius (m)")
    parser.add_argument("--dt", type=float, default=0.1, help="Sampling dt (s)")
    parser.add_argument("--horizon", type=float, default=2.0, help="Lookahead horizon (s)")
    parser.add_argument("--safety-radius", type=float, default=0.6, help="Collision safety radius (m)")
    parser.add_argument("--pad", type=float, default=0.2, help="Extra spatial pad (m) when filtering to zone")
    args = parser.parse_args()

    def load_waypoints_json(path: str) -> List[Waypoint]:
        with open(path, "r") as f:
            data = json.load(f)
        wps = data["waypoints"]
        return [(float(w["x"]), float(w["y"]), float(w.get("speed", 1.0))) for w in wps]

    # --- Build waypoint map ---
    wp_map: Dict[str, List[Waypoint]] = {}

    if args.json:
        # load from provided json files
        for vid, p in args.json:
            if not os.path.exists(p):
                raise FileNotFoundError(p)
            wp_map[vid] = load_waypoints_json(p)
    else:
        # Synthetic demo: two paths crossing near (0,0)
        # robotA goes left->right through (0,0)
        wp_map["robotA"] = [(-1.0, 0.0, 1.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0)]
        # robotB goes bottom->top through (0,0)
        wp_map["robotB"] = [(0.0, -1.0, 1.0), (0.0, 0.0, 1.0), (0.0, 1.0, 1.0)]
        # robotC arrives later (no conflict if horizon is small)
        wp_map["robotC"] = [(0.0, -1.0, 0.5), (0.0, 0.0, 0.5), (0.0, 1.0, 0.5)]
        print("[Demo] Using synthetic trajectories for robotA/robotB/robotC")

    # --- Sample everything on a common time grid ---
    samples = build_samples_from_waypoints(
        wp_map=wp_map, ids=list(wp_map.keys()), dt=args.dt, horizon=args.horizon
    )

    # Print some samples for sanity
    for vid in samples:
        snip = samples[vid][:5]
        print(f"{vid} samples (first 5): {snip}")

    ids = list(wp_map.keys())
    cx, cy = args.center
    zone_r = args.zone_radius



if __name__ == "__main__":
    main()


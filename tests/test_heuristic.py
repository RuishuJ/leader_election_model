"""
Test script for the Multi-Agent PID Road Network Gym Environment.
Demonstrates multi-agent discrete action space {STOP, GO}.
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from examples.multi_agent_gym_env import make_multi_agent_env
from duckietown_simulator.Model.intersection_manager_V2 import IntersectionManager, find_intersection_centers
from duckietown_simulator.Model.trajectory_conflict import (waypoints_map,
                                                            build_aligned_samples_from_waypoints,
                                                            same_lane_safety)
from duckietown_simulator.Model.leader_election_manager import LeaderElectionManager
from demos.demo_pid_road_network import create_map_from_json
from duckietown_simulator.Model.post_encroachment_time import PostEncroachmentTimeMonitor
import wandb
from duckietown_simulator.Model.heuristic_leader_election_manager import (HeuristicLeaderElectionManager,
                                                                          TrafficState,
                                                                          VehicleMeta,
                                                                          combine_with_stop_dominance,
                                                                          enforce_hard_stop_in_control_zone,
                                                                          assign_vehicle_meta,
                                                                          select_co_leaders_with_leader,
                                                                          )
from duckietown_simulator.Model.safety_manager import SafetyManager

from duckietown_simulator.Model.trajectory_policy import (
    policy as TRAJ_POLICY,
    SAFE_GROUPS_BY_SIZE,
    TURN_FROM_IN_OUT,
    policy_lookup,
)
from collections import deque, defaultdict


# Weights & Biases configuration
wandb.init(mode="disabled")

# wandb.init(
#     project="duckie-heuristic",
#     name="run-heuristic",
#     config={
#         "selection_method": "random",
#         "num_agents": 7,
#         "map": "road_network.json",
#         "pet_threshold": 0.2,
#     },
#     tags=["heuristic-selection"]
# )

STEP_KEY = "global_step"
wandb.define_metric(STEP_KEY)
wandb.define_metric("*", step_metric=STEP_KEY)

def wb_log(data: dict, *, commit=False, total_steps: int = 0):
    payload = {STEP_KEY: total_steps}
    payload.update(data)
    wandb.log(payload, commit=commit)

def test_multi_agent_basic():
    """Test basic multi-agent functionality."""
    print("Testing Multi-Agent PID Road Network Gym Environment")
    print("=" * 60)

    # Safety Management - PET Check
    PET_TOL = 0.20       # seconds
    PET_SAFETY_R = 0.25  # meters
    PET_DT = 0.1
    PET_HORIZON = 1.5

    # Same-lane safety
    SAME_LANE_LATERAL_TOL = 0.20
    SAME_LANE_MIN_GAP = 0.30

    # Create environment and trajectories
    traj_dict = {
        "robot1": "../data/looped_trajectory_11.json",
        "robot2": "../data/looped_trajectory_2.json",
        "robot3": "../data/looped_trajectory_3.json",
        "robot4": "../data/looped_trajectory_41.json",
        "robot5": "../data/looped_trajectory_5.json",
        "robot6": "../data/looped_trajectory_6.json",
        "robot7": "../data/looped_trajectory_7.json",
    }

    env = make_multi_agent_env(
        num_agents=7,
        trajectory_files= traj_dict,
        render_mode='human'  # Use human rendering for visualization
    )

    STOP, GO, SLOWDOWN = 0, 1, 2
    print(f"[mapping] STOP={STOP}, GO={GO}")

    print(f"Agent IDs: {env.agent_ids}")
    print(f"Action spaces: {env.action_space}")
    print(f"Number of agents: {env.num_agents}")


    # Load the map and compute the intersection centers
    map_object = create_map_from_json(
        "/Users/ruijiang/PycharmProjects/gym-tsl-duckietown/duckietown_simulator/assets/maps/road_network.json")

    intersection_centers_world = find_intersection_centers(map_instance = map_object, x_offset=0.3, y_offset=1.5)
    print(f"[centers] coords: {intersection_centers_world}")

    # Managers
    im = IntersectionManager(map_object, default_radius=0.80)
    lem = LeaderElectionManager(centers=intersection_centers_world, exit_radius=0.60)
    hlem = HeuristicLeaderElectionManager({})
    sm = SafetyManager(intersection_manager=im)
    conflict_matrix = sm.build_conflict_matrix(TRAJ_POLICY)

    # Reset environment
    obs, infos = env.reset()
    PET_Monitor = PostEncroachmentTimeMonitor(
        cell_size = 0.2,
        pet_threshold = 0.2,  # seconds
        max_event_age = 3.0,
        dt = 0.1
    )
    print(f"Initial observations shape: {list(obs.keys())} -> {[obs[k].shape for k in obs.keys()]}")
    print(f"Observation details for robot1: {obs['robot1']}")


    total_steps = 0
    CONTROL_RADIUS = lem.exit_radius # 0.5
    num_steps = 250000

    WINNER_PET_TOL = 0.40  # s
    WINNER_MIN_GAP = 0.25  # m

    # For evaluation
    # Safety - PET counters
    pet_violations_total = 0
    min_pet_seen = float("+inf")

    # Fairness - starvation - track consecutive STOP
    stop_streak = {rid: 0 for rid in env.agent_ids}

    # Fairness - priority-respect
    prio_respect_hits = 0
    prio_respect_total = 0

    # Efficiency: waiting time
    wait_time = {rid: 0.0 for rid in env.agent_ids}
    cumulative_wait_total = 0.0
    # conflict_matrix = sm.build_conflict_matrix(RobotPolicy)

    # Efficiency: time-base throughput
    dt_env = float(getattr(env, "dt", 0.1))
    sim_time_s = 0.0  # simulated time
    throughput_total = 0 # total throughput
    throughput_per_center = defaultdict(int) # center throughput
    center_exit_times = defaultdict(lambda: deque(maxlen=5000))  # center exits
    recent_exit_times_s = deque(maxlen=10000)  # all exits
    veh_center_prev = {rid: None for rid in env.agent_ids}
    veh_enter_time_s = {}
    dwell_time_samples_s = deque(maxlen=10000)

    if True:
        print(f"\n--- {num_steps} steps ---")

        for step in range(num_steps):

            if env.render_mode == 'human':
                if env.renderer.paused:
                    env.render()
                    continue

            # PET update the current samples
            samples_all = build_aligned_samples_from_waypoints(
                waypoints_map = waypoints_map,
                robots = env.robots,
                ids = env.agent_ids,
                dt = PET_DT,
                horizon = PET_HORIZON,
            )

            # Clear leaders who left
            exited = lem.update_and_get_exited(env.robots, exit_radius = CONTROL_RADIUS)
            for c in exited:
                print(f"*******step{total_steps}: [leader] previous leader left @ {c} -> needs reselection")

            zones_control = im.in_the_intersection_zone(
                env.robots, intersection_centers_world, radius = CONTROL_RADIUS
            )

            # Policy 1: Baseline policy: everyone GO by default
            actions_dict = {agent_id: GO for agent_id in env.agent_ids}
            reservation_actions = {}

            # if robots are in (2, 0) or (2, 3), slows down
            for agent_id in env.agent_ids:
                if env.robots[agent_id].current_tile == (0, 3) or env.robots[agent_id].current_tile == (4, 3):
                    actions_dict[agent_id] = SLOWDOWN  # SLOWDOWN

            # For logging: Record leaders for this step
            leaders_this_step = []
            leader_by_center = {}

            # Heuristic Leader Election
            for (cx, cy), _, vehicles in zones_control:
                center = (cx, cy)
                if not vehicles:
                    continue

                leader_id = lem.get_leader(center)
                print(f'Leader ID: {leader_id}')

                # Build meta for vehicles at this center with incoming/outgoing
                meta = assign_vehicle_meta(vehicles, center, TRAJ_POLICY, TURN_FROM_IN_OUT)
                for rid in vehicles:
                    vm = VehicleMeta()
                    # Define priority
                    vm.priority_class = {"robot1": "ambulance", "robot7": "ambulance",
                                     "robot2": "bus", "robot5": "bus"}.get(rid, "private")
                    vm.rank = {"ambulance": 0, "bus": 1, "private": 2}[vm.priority_class]
                    vm.intersection_id = center
                    # Check incoming/outgoing from trajectory policy
                    for incoming_dir in ("N", "E", "S", "W"):
                        out = policy_lookup(TRAJ_POLICY, rid, center, incoming_dir)
                        if out is not None:
                            vm.incoming = incoming_dir
                            vm.outgoing = out
                            if vm.incoming is not None and vm.outgoing is not None:
                                vm.turn_type = TURN_FROM_IN_OUT.get((vm.incoming, vm.outgoing), "straight")
                            else:
                                vm.turn_type = "straight"
                            break
                    meta[rid] = vm

                # Build the traffic state
                ts = TrafficState(
                    robots = {rid: env.robots[rid] for rid in vehicles},
                    meta = meta,
                    wait_time = wait_time,
                    conflict_matrix = conflict_matrix,
                    im = im,
                    time_to_clear={}
                )

                # Score vehicles, select leader
                scores = {rid: hlem.score_vehicle(rid, ts) for rid in vehicles}
                if (leader_id is None) or (leader_id not in vehicles):
                    leader_id = min(scores, key=scores.get)
                    lem.set_leader(center, leader_id)
                    leader_by_center[center] = leader_id

                # Search co-leaders by trajectory conflict
                co_leaders = select_co_leaders_with_leader(
                    leader_id, vehicles, meta, conflict_matrix, SAFE_GROUPS_BY_SIZE
                )
                print(
                    f"  - Max time to clear (GO group, real-time): {ts.time_to_clear_group_if_leader(leader_id):.2f}s")
                print(f"  - Co-leaders: {co_leaders}")
                if co_leaders:
                    print(f" Co-leaders elected with leader {leader_id}: {co_leaders}")
                else:
                    print(f"  No co-leaders elected for leader {leader_id}")

                # W&B: score lines
                center_key = f"{center}"
                leader_score = scores[leader_id]
                coleader_scores = [scores[r] for r in co_leaders if r != leader_id]
                other_scores = [scores[r] for r in vehicles if r not in co_leaders]

                wandb.log({
                    "step": total_steps,
                    f"{center_key}/leader/score": leader_score,
                    f"{center_key}/coleaders/score_mean": (np.mean(coleader_scores) if coleader_scores else np.nan),
                    f"{center_key}/others/score_mean": (np.mean(other_scores) if other_scores else np.nan),
                })

                # W&B: log a table for this center this step
                rows = [{"robot": r, "role": "leader" if r==leader_id else ("coleader" if r in co_leaders else "other"),
                         "score": scores[r]} for r in vehicles]
                wandb.log({
                    "step": total_steps,
                    f"{center_key}/scores_table": wandb.Table(dataframe = pd.DataFrame(rows))
                })

                # W&B: Fairness Priority respect
                ranks_here = {rid: ts.rank_of(rid) for rid in vehicles}
                best_rank = min(ranks_here.values()) if ranks_here else 99
                prio_respect_total += 1
                prio_respect_hits += int(ranks_here.get(leader_id, 99) == best_rank)

                wandb.log({
                    "step": total_steps,
                    "fairness/priority_respect_fraction_step": prio_respect_hits / max(1, prio_respect_total),
                })

                # Policy2: leader & co-leaders GO, others STOP
                for rid in vehicles:
                    reservation_actions[rid] = GO if rid in co_leaders else STOP
                    action = "GO" if reservation_actions[rid] == GO else "STOP"
                    reason = "leader or compatible" if rid in co_leaders else "conflict with leader"
                    print(f"    {rid}: {action} ({reason}) score={scores[rid]:.3f}")

                actions_dict = combine_with_stop_dominance(
                    actions_dict,
                    reservation_actions,
                    all_ids = list(env.agent_ids),
                    stop_value = STOP,
                    go_value = GO
                )

                actions_dict = enforce_hard_stop_in_control_zone(
                    actions_dict, zones_control, leader_by_center, stop_value=STOP
                )

                # Safety: Policy 3 Gap-based Rear-end safety in control zone (check when front & same-lane GO)
                vs_list = list(vehicles)
                for i in range(len(vs_list)):
                    for j in range(i + 1, len(vs_list)):
                        a, b = vs_list[i], vs_list[j]
                        try:
                            sa = same_lane_safety(
                                im, a, env.robots[a], b, env.robots[b],
                                position_tolerance=SAME_LANE_LATERAL_TOL,
                                min_gap=SAME_LANE_MIN_GAP
                            )
                        except Exception:
                            continue

                        if sa["same_lane"] and sa["should_stop"]:
                            rear = sa["behind_id"];
                            front = sa["ahead_id"]
                            if actions_dict.get(front) == GO:
                                actions_dict[rear] = STOP
                                print(f"***Policy3-gap-lane safety*** {rear}=STOP behind {front} (gap={sa['gap']:.2f})")

                # W&B: logging with REAL-TIME times
                max_t = ts.time_to_clear_group_if_leader(leader_id, v_avg=0.25, min_speed=0.05)
                print(f"\n[step {total_steps}] Intersection @ {center}:")
                print(f"  - Leader: {leader_id} (score={scores[leader_id]:.3f})")
                print(f"  - Max time to clear (GO group, real-time): {max_t:.2f}s")
                for rid in vehicles:
                    t_single = ts.time_to_clear_single(rid, v_avg=0.25, min_speed=0.05)
                    reason = "leader or compatible" if rid in co_leaders else "conflict with leader"
                    v_now = float(getattr(env.robots[rid], 'linear_velocity', 0.0))
                    print(
                        f"    {rid}: {'GO' if actions_dict[rid] == GO else 'STOP'} ({reason})  t≈{t_single:.2f}s  v={v_now:.2f} m/s  score={scores[rid]:.3f}")

                    # Safety: Policy 3* rear STOP (Conflict zone) (check when front & same-lane STOP)
                    vs_list = list(vehicles)
                    stopped_fronts = [rid for rid in vs_list if actions_dict.get(rid, GO) == STOP]

                    for rear in vs_list:
                        if actions_dict.get(rear, GO) == STOP:
                            continue
                        for front in stopped_fronts:
                            if rear == front:
                                continue

                            # 1) Same-lane gap-based
                            try:
                                sa_res = same_lane_safety(
                                    im, rear, env.robots[rear], front, env.robots[front],
                                    position_tolerance=SAME_LANE_LATERAL_TOL,
                                    min_gap=SAME_LANE_MIN_GAP
                                )
                                if sa_res["same_lane"] and sa_res["should_stop"] and sa_res["behind_id"] == rear:
                                    actions_dict[rear] = STOP
                                    print(
                                        f"***Policy3*-gap*** {rear}=STOP behind stopped {front}; gap={sa_res['gap']:.2f} m")
                                    continue
                            except Exception:
                                pass

                            # 2) Policy 4: PET-based
                            sa = samples_all.get(front, [])
                            sb = samples_all.get(rear, [])
                            res = PET_Monitor.pet_from_samples(a=sa, b=sb, safety_radius=PET_SAFETY_R)
                            if res is not None:
                                pet, t_front, t_rear = res
                                if (pet is not None) and (t_front is not None) and (t_rear is not None):
                                    if (t_rear > t_front) and (pet < WINNER_PET_TOL):
                                        actions_dict[rear] = STOP
                                        print(f"***Policy4-PET*** {rear}=STOP; PET={pet:.2f}s vs stopped {front}")

                # Safety: Policy5- PATCH: STOPPED-FRONT PROPAGATION (inside-outside)
                # Any vehicle STOP inside the control zone becomes a "front".
                # Force any true follower (even if outside Conflict Zone) to STOP, by same-lane gap or PET timing.
                vehicles_in_control_zone = set()
                for (_c, _r, vs) in zones_control:
                    vehicles_in_control_zone.update(vs)

                stopped_fronts = [vid for vid in vehicles if actions_dict.get(vid, GO) == STOP]

                for front in stopped_fronts:
                    for rear in env.agent_ids:
                        if rear == front:
                            continue
                        if leader_by_center.get((cx, cy)) == rear: # Leader GO
                            continue

                        try:
                            sa_res = same_lane_safety(
                                im, rear, env.robots[rear], front, env.robots[front],
                                position_tolerance=SAME_LANE_LATERAL_TOL,
                                min_gap=WINNER_MIN_GAP
                            )
                            if sa_res["same_lane"] and sa_res["behind_id"] == rear and sa_res["should_stop"]:
                                actions_dict[rear] = STOP
                                print(f"***Policy5-PROP-GAP*** {rear}=STOP behind stopped {front}; gap={sa_res['gap']:.2f} m")
                                continue
                        except Exception:
                            pass

            free_ids = [aid for aid in env.agent_ids if aid not in vehicles_in_control_zone]
            # Policy6&7: Global safety Same-lane spacing and PET
            for i in range(len(free_ids)):
                for j in range(i + 1, len(free_ids)):
                    a, b = free_ids[i], free_ids[j]
                    try:
                        sa = same_lane_safety(
                            im, a, env.robots[a], b, env.robots[b],
                            position_tolerance=SAME_LANE_LATERAL_TOL,
                            min_gap=SAME_LANE_MIN_GAP
                        )
                    except Exception:
                        continue
                    if sa["same_lane"] and sa["should_stop"]:
                        rear = sa["behind_id"];
                        front = sa["ahead_id"]
                        actions_dict[rear] = STOP
                        print(f"***Policy6-OUTSIDE-LANE*** {rear}=STOP behind {front} (gap={sa['gap']:.2f})")

            # Policy7: PET overlay (for cross-approach)
            for i in range(len(free_ids)):
                for j in range(i + 1, len(free_ids)):
                    a, b = free_ids[i], free_ids[j]
                    sa_samps = samples_all.get(a, [])
                    sb_samps = samples_all.get(b, [])
                    res = PET_Monitor.pet_from_samples(sa_samps, sb_samps, safety_radius=PET_SAFETY_R)
                    if res is None:
                        continue
                    pet, ta, tb = res
                    if pet < PET_TOL:
                        later = b if (tb is not None and ta is not None and tb > ta) else a
                        other = a if later == b else b
                        actions_dict[later] = STOP
                        print(f"***Policy7-OUTSIDE-PET*** {later}=STOP vs {other} (PET={pet:.2f}s)")

            # W&B: Throughput & dwell-time
            veh_center_now = {rid: None for rid in env.agent_ids}
            for (cx, cy), _, vs in zones_control:
                ck = f"({cx:.2f},{cy:.2f})"
                for rid in vs:
                    veh_center_now[rid] = ck

            exits_this_tick = []
            for rid in env.agent_ids:
                prev = veh_center_prev.get(rid)
                cur = veh_center_now.get(rid)

                # Entered a Conflict zone
                if prev is None and cur is not None:
                    veh_enter_time_s[rid] = sim_time_s
                    veh_center_prev[rid] = cur

                # Exited Conflict zone
                elif prev is not None and cur is None:
                    enter_t = veh_enter_time_s.pop(rid, sim_time_s)
                    dwell_s = max(0.0, sim_time_s - enter_t)
                    dwell_time_samples_s.append(dwell_s)

                    throughput_total += 1
                    throughput_per_center[prev] += 1
                    recent_exit_times_s.append(sim_time_s)
                    center_exit_times[prev].append(sim_time_s)
                    exits_this_tick.append(prev)

                    veh_center_prev[rid] = None

                # Switched centers- count as an exit+enter
                elif prev is not None and cur is not None and prev != cur:
                    enter_t = veh_enter_time_s.get(rid, sim_time_s)
                    dwell_s = max(0.0, sim_time_s - enter_t)
                    dwell_time_samples_s.append(dwell_s)

                    throughput_total += 1
                    throughput_per_center[prev] += 1
                    recent_exit_times_s.append(sim_time_s)
                    center_exit_times[prev].append(sim_time_s)
                    exits_this_tick.append(prev)

                    veh_enter_time_s[rid] = sim_time_s  # start timing in new center
                    veh_center_prev[rid] = cur

            # W&B tick
            if exits_this_tick:
                log_payload = {
                    "throughput/total_crossings": throughput_total,
                    "throughput/exits_this_tick": len(exits_this_tick),
                    "time/sim_seconds": sim_time_s,
                }
                for ck in set(exits_this_tick):
                    per_tick = sum(1 for c in exits_this_tick if c == ck)
                    log_payload[f"throughput/{ck}/exits_this_tick"] = per_tick
                    log_payload[f"throughput/{ck}/total_crossings"] = throughput_per_center[ck]
                wandb.log(log_payload, step=total_steps)

            # W&B Moving-window time rates (vehicles per 60s & 300s)
            def _count_within(deq, now_s, window_s):
                thresh = now_s - window_s
                return sum(1 for t in deq if t > thresh)

            wandb.log({"step": total_steps,
                "throughput/rate_per_60s": _count_within(recent_exit_times_s, sim_time_s, 60.0),
                "throughput/rate_per_300s": _count_within(recent_exit_times_s, sim_time_s, 300.0),
                "time/sim_seconds": sim_time_s,
            })

            # W&B: Dwell-time
            if (total_steps % 200 == 0) and len(dwell_time_samples_s) > 0:
                wandb.log({
                    "step": total_steps,
                    "throughput/dwell_time_s_hist": wandb.Histogram(list(dwell_time_samples_s)),
                    "throughput/dwell_time_mean_s": float(np.mean(dwell_time_samples_s)),
                    "throughput/dwell_time_median_s": float(np.median(dwell_time_samples_s)),
                    "time/sim_seconds": sim_time_s,
                })

            # Final: Protect leaders and control-zone vehicles
            vehicles_in_control_zone: set[str] = set()
            for (_c, _r, vs) in zones_control:
                vehicles_in_control_zone.update(vs)

            # As a final safety, make sure each zone leader is GO
            for (cx, cy), _, vs in zones_control:
                L = lem.get_leader((cx, cy))
                if L in vs:
                    actions_dict[L] = GO
                for rid in vs:
                    if reservation_actions.get(rid, STOP) == GO:
                        actions_dict[rid] = GO

            # Step the environment
            obs, rewards, terminated, truncated, infos = env.step(actions_dict)
            # W&B Simulated time
            sim_time_s += dt_env
            for rid in env.agent_ids:
                if actions_dict.get(rid, GO) == STOP:
                    wait_time[rid] += env.dt
                else:
                    wait_time[rid] = 0.0

            mean_wait = float(np.mean([wait_time[rid] for rid in env.agent_ids])) if env.agent_ids else 0.0
            cumulative_wait_total += sum(wait_time.values())
            wandb.log({
                "step": total_steps,
                "step/plot": total_steps,
                "efficiency/mean_wait_time_s": mean_wait,
                "episode/efficiency/total_wait_time_s": cumulative_wait_total
            })

            free_ids = [rid for rid in env.agent_ids if actions_dict.get(rid, GO) != STOP]
            mean_speed_free = float(np.mean([getattr(env.robots[rid], "linear_velocity", 0.0)
                                             for rid in free_ids])) if free_ids else 0.0
            wb_log({"efficiency/mean_speed_free_mps": mean_speed_free}, total_steps=total_steps)

            for rid in env.agent_ids:
                stop_streak[rid] = stop_streak.get(rid, 0) + 1 if actions_dict.get(rid, GO) == STOP else 0
            max_stop_streak = max(stop_streak.values()) if stop_streak else 0
            wb_log({"fairness/max_stop_streak_frames": max_stop_streak}, total_steps=total_steps)

            wandb.log({
                "step": total_steps,
                "safety/pet_violations_total": pet_violations_total,
                "safety/min_pet_seen_s": (min_pet_seen if np.isfinite(min_pet_seen) else np.nan),
            })

            mean_speed_free = float(
                np.mean([getattr(env.robots[rid], "linear_velocity", 0.0) for rid in free_ids])) if free_ids else 0.0

            wandb.log({
                "step": total_steps,
                "efficiency/mean_speed_free_mps": mean_speed_free
            })

            for rid in env.agent_ids:
                if actions_dict.get(rid, GO) == STOP:
                    stop_streak[rid] = stop_streak.get(rid, 0) + 1
                else:
                    stop_streak[rid] = 0

            max_stop_streak = max(stop_streak.values()) if stop_streak else 0
            wandb.log({
                "step": total_steps,
                "fairness/max_stop_streak_frames": max_stop_streak})

            if total_steps % 25 == 0 and leader_id in scores:
                lc = hlem.components(leader_id, ts)
                wandb.log({
                    "step": total_steps,
                    f"{center_key}/leader/comp_wait_self": lc["wait_self"],
                    f"{center_key}/leader/comp_wait_other": lc["wait_other"],
                    f"{center_key}/leader/comp_prio_self": lc["prio_self"],
                    f"{center_key}/leader/comp_prio_other": lc["prio_other"],
                    f"{center_key}/leader/comp_trajectory": lc["trajectory"],
                    f"{center_key}/leader/comp_teammates": lc["teammates"],
                    f"{center_key}/leader/score_total": lc["score"],
                })

            # Render environment
            if not env.render():
                print("Window closed!")
                return

            # Print status every 25 steps
            if step % 25 == 0:
                show_info(actions_dict, env, infos, obs, rewards, total_steps)
                print(f"  Step {total_steps}:")
                for agent_id in env.agent_ids:
                    action_name = "STOP" if actions_dict[agent_id] == STOP else "GO"
                    print(f"    {agent_id}: Action={action_name}, Reward={rewards[agent_id]:.2f}, "
                          f"Progress={infos[agent_id]['waypoint_progress']['progress_ratio'] * 100:.1f}%, "
                          f"Collisions={infos[agent_id]['collisions']}")
                    # show robots position and speed
                    print(f"    {agent_id} Position: ({obs[agent_id][0]:.2f}, {obs[agent_id][1]:.2f}), "
                          f"Speed: {infos[agent_id]['robot_speeds']['linear']:.2f} m/s")

            # final commit for this step
            wb_log({}, commit=True, total_steps=total_steps)
            total_steps += 1

            # Check if any agent completed or episode ended
            if any(terminated.values()) or any(truncated.values()):
                print(f"\nEpisode ended at step {total_steps}!")
                for agent_id in env.agent_ids:
                    print(f"  {agent_id}: Terminated={terminated[agent_id]}, Truncated={truncated[agent_id]}")
                env.close()
                return

            if leaders_this_step:
                print("  Leaders selected (FIFO):")
                for L in leaders_this_step:
                    cx, cy = L["center"]
                    cand_str = ", ".join(f"{cid}({sec}, d={dist:.2f})" for cid, sec, dist in L["candidates"])
                    print(f"    @({cx:.2f},{cy:.2f}) -> {L['leader']} from {L['sector']} | candidates: {cand_str}")

    print(f"\nTest completed! Total steps: {total_steps}")
    # time.sleep(0.5)
    env.close()
    wandb.log({
        "episode/safety/pet_violations_total": pet_violations_total,
        "episode/safety/min_pet_seen_s": (min_pet_seen if np.isfinite(min_pet_seen) else np.nan),
        "episode/fairness/final_priority_respect_fraction": prio_respect_hits / max(1, prio_respect_total),
        "episode/fairness/max_stop_streak_frames": max(stop_streak.values()) if stop_streak else 0,
        "episode/throughput/total_crossings": throughput_total,
        "episode/throughput/mean_dwell_time_s": (
            float(np.mean(dwell_time_samples_s)) if dwell_time_samples_s else np.nan),
        "episode/throughput/median_dwell_time_s": (
            float(np.median(dwell_time_samples_s)) if dwell_time_samples_s else np.nan),
        "episode/time/total_sim_seconds": sim_time_s,
    }, commit = True)

def show_info(actions_dict, env, infos, obs, rewards, total_steps):

    print(f"  Step {total_steps}:")
    for agent_id in env.agent_ids:
        if actions_dict[agent_id] == 0:
            action_name = "STOP"
        elif actions_dict[agent_id] == 1:
            action_name = "GO"
        else:  # action == 2
            action_name = "SLOWDOWN"
        print(f"    {agent_id}: Action={action_name}, Reward={rewards[agent_id]:.2f}, "
              f"Progress={infos[agent_id]['waypoint_progress']['progress_ratio'] * 100:.1f}%, "
              f"Collisions={infos[agent_id]['collisions']}")
        # show robots position and speed
        print(f"    {agent_id} Position: ({obs[agent_id][0]:.2f}, {obs[agent_id][1]:.2f}), "
              f"Speed: {infos[agent_id]['robot_speeds']['linear']:.2f} m/s")


if __name__ == "__main__":
    # Test basic multi-agent functionality
    test_multi_agent_basic()

    print("\nAll multi-agent tests completed!")

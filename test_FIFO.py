"""
Test script for the Multi-Agent PID Road Network Gym Environment.
Demonstrates multi-agent discrete action space {STOP, GO}.
"""
import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from examples.multi_agent_gym_env import make_multi_agent_env
from duckietown_simulator.Model.intersection_manager_V2 import IntersectionManager, find_intersection_centers
from duckietown_simulator.Model.trajectory_conflict import (waypoints_map,
                                                            build_aligned_samples_from_waypoints,
                                                            same_lane,
                                                            same_lane_safety, normalize_waypoints, time_parameterise_waypoints)
from duckietown_simulator.Model.leader_election_manager import LeaderElectionManager
from demos.demo_pid_road_network import create_map_from_json
from duckietown_simulator.Model.post_encroachment_time import PostEncroachmentTimeMonitor
import wandb
from duckietown_simulator.Model.FIFO_leader_election_manager import FIFOLeaderElection, compute_intersection_boundaries

# Initialize W&B
# wandb.init(project="demo-project", name="run-minimal")

def load_trajectory(path: str):
    with open(path) as f:
        raw = json.load(f)
    waypoints = normalize_waypoints(raw)
    return time_parameterise_waypoints(waypoints)

# =====================================================================
# Test: sets up environment, computes intersection zones/centers,
# and runs a long rollout with layered safety/control policies:
#   • Policy 2/3: Winners-only (leader GO, other winners STOP)
#   • Policy 4/10: Same-lane gap propagation (rear STOPs if too close)
#   • Policy 5/7/9: PET-based yielding (later vehicle STOPs if PET small)
#   • FIFO leader election when a zone needs a new leader
#   • Control-zone protection: leaders forced GO inside exit radius
# =====================================================================

def test_multi_agent_basic():
    """Test basic multi-agent functionality."""
    print("Testing Multi-Agent PID Road Network Gym Environment")
    print("=" * 60)

    PET_TOL = 0.20       # seconds
    PET_SAFETY_R = 0.25  # meters
    PET_DT = 0.1
    PET_HORIZON = 1.5

    # Winners policy & propagation
    WINNER_PET_TOL = 0.40  # seconds; rear yields to a stopped winner if PET < this
    WINNER_MIN_GAP = 0.25  # meters; rear yields if too close behind a stopped winner

    # Same-lane safety
    SAME_LANE_LATERAL_TOL = 0.15
    SAME_LANE_MIN_GAP = 0.30

    # Create environment
    traj_dict = {
        "robot1": "../data/looped_trajectory_11.json",
        "robot2": "../data/looped_trajectory_2.json",
        "robot3": "../data/looped_trajectory_3.json",
        "robot4": "../data/looped_trajectory_4.json",
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

    # Load the Duckietown map and compute the intersection centers
    map_object = create_map_from_json(
        "/Users/ruijiang/PycharmProjects/gym-tsl-duckietown/duckietown_simulator/assets/maps/road_network.json")

    intersection_centers_world = find_intersection_centers(map_instance = map_object, x_offset=0.3, y_offset=1.5)
    print(f"[centers] coords: {intersection_centers_world}")

    im = IntersectionManager(map_object, default_radius=0.80)
    lem = LeaderElectionManager(centers=intersection_centers_world, exit_radius=0.60)

    intersection_boundaries, intersection_centers_dict = compute_intersection_boundaries(intersection_centers_world)

    trajectories = {
        rid: load_trajectory(path)
        for rid, path in traj_dict.items()
    }

    fifo = FIFOLeaderElection(intersection_manager = im,
        trajectories = trajectories,
        intersection_boundaries = intersection_boundaries,
        intersection_centers = intersection_centers_dict,
        FIFO_radius= 1.0)

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
    FIFO_RADIUS = fifo.FIFO_radius # 1.5
    num_steps = 15000


    if True:
        print(f"\n--- {num_steps} steps ---")

        for step in range(num_steps):

            # if robots are in (2, 0) or (2, 3), slows down
            for agent_id in env.agent_ids:
                if env.robots[agent_id].current_tile == (0, 3) or env.robots[agent_id].current_tile == (4, 3):
                    actions_dict[agent_id] = 2  # SLOWDOWN

            if env.render_mode == 'human':
                if env.renderer.paused:
                    env.render()
                    continue

            # Safety: Update sample for PET
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

            zones_FIFO_select = im.in_the_intersection_zone(
                env.robots, intersection_centers_world, radius= FIFO_RADIUS
            )
            zones_control = im.in_the_intersection_zone(
                env.robots, intersection_centers_world, radius = CONTROL_RADIUS
            )

            winners_map = im.winners_by_zone(
                env.robots,
                zones_FIFO_select,
                radius = FIFO_RADIUS,
                filter_to_zone_members=True,  # ensures winners ⊆ vehicles
            )
            # Policy 1- Baseline policy: everyone GO by default, override in intersections
            actions_dict = {agent_id: GO for agent_id in env.agent_ids}

            leaders_this_step = []

            # FIFO Leader Election
            for (cx, cy), _, vehicles in zones_control:
                center = (cx, cy)
                if not vehicles:
                    continue  # nothing in this intersection
                winners = winners_map.get(center, [])
                winner_ids = im._extract_ids_from_winners(winners)
                print(f"step{total_steps}: [winners] {winner_ids} @ {center}")

                leader_id = lem.get_leader(center)
                print(f'step{total_steps}: Leader ID: {leader_id} @ {center}')

                # restrict to veh still in the control zone
                vehicles_in_control_zone = set(vehicles)
                winners_in_control_zone = [rid for rid in winner_ids if rid in vehicles_in_control_zone]

                non_leader_winners = [rid for rid in winners_in_control_zone if rid != leader_id]

                print(
                    f"step{total_steps}: [winners] all={winner_ids}  "
                    f"in_control={winners_in_control_zone} "
                    f" non_leader_winners={non_leader_winners} @ {center}")

                # Policy 2 - Winners policy (ControlZone): only the leader goes; non-leader winners IN CONTROL zone STOP
                for rid in non_leader_winners:
                    actions_dict[rid] = STOP
                    print(f"***Policy2***[winners-policy] {rid}=STOP (non-leader winner IN control zone) @ {center}")

                if (leader_id is None) or (leader_id not in vehicles):

                    if not winner_ids:
                        print(f"[vtl] @({cx:.2f},{cy:.2f}) skip: no winners in selection zone")
                        continue

                    # Re-elect leader by FIFO from current winners
                    winner_agents = {aid: env.robots[aid] for aid in winner_ids}
                    fifo.update_spatial_queue(intersection_id=center, agents=env.robots)
                    queue = fifo.priority_queue.get(center, [])
                    if queue:
                        sorted_queue = sorted(queue.items(), key=lambda item: item[1])
                        queue_str = ", ".join(f"{aid}(priority={p})" for aid, p in sorted_queue)
                        print(f"[Queue] @({center[0]:.2f},{center[1]:.2f}): {queue_str}")
                    else:
                        print(f"[Queue] @({center[0]:.2f},{center[1]:.2f}): [empty]")

                    print(f"step{step}: FIFO priority queue @ {center} → {queue}")

                    leader_id = fifo.FIFO_elect_leader(intersection_id = center,
                                                       agents = winner_agents ,
                                                       winner_ids = winner_ids,
                                                       use_candidates=False)

                    lem.set_leader(center, leader_id)
                    # log the pick
                    leader_sector = next((w["sector"] for w in winners if w["id"] == leader_id), "?")
                    leaders_this_step.append({
                        "center": center,
                        "leader": leader_id,
                        "sector": leader_sector,
                        "candidates": [(w["id"], w["sector"], w["dist"]) for w in winners],
                    })

                # Ensure leader always GO inside its control zone
                if leader_id in zones_control:
                    actions_dict[leader_id] = GO

                # Safety: Policy 3 - Winners policy (Selection Zone): leader GO, other winners STOP
                non_leader_winners = [rid for rid in winner_ids if rid != leader_id]
                for rid in non_leader_winners:
                    actions_dict[rid] = STOP
                    print(f"***Policy3***[winners-policy] {rid}=STOP (non-leader winner) @ {center}")

                # Safety: Policy 9-PET overlay: only for cross-approach pairs (skip same-lane),
                free_ids = [aid for aid in env.agent_ids if aid not in vehicles_in_control_zone]
                for i in range(len(free_ids)):
                    for j in range(i + 1, len(free_ids)):
                        a, b = free_ids[i], free_ids[j]

                        try:
                            if same_lane(im, env.robots[a], env.robots[b],
                                         position_tolerance=SAME_LANE_LATERAL_TOL):
                                continue
                        except Exception:
                            continue

                        sa = samples_all.get(a, [])
                        sb = samples_all.get(b, [])
                        res = PET_Monitor.pet_from_samples(sa, sb, safety_radius=PET_SAFETY_R)
                        if res is None:
                            continue
                        pet, ta, tb = res
                        if pet < PET_TOL:
                            later = b if tb > ta else a
                            actions_dict[later] = STOP
                            print(
                                f"***Policy9***[PET-global] {later}=STOP vs {a if later == b else b} (PET={pet:.2f}s)")

            # Safety: Policy8 Global Gap-based (Same-lane spacing)
            ids_all = list(env.agent_ids)
            for i in range(len(ids_all)):
                for j in range(i + 1, len(ids_all)):
                    a, b = ids_all[i], ids_all[j]
                    sa = same_lane_safety(
                        im, a, env.robots[a], b, env.robots[b],
                        position_tolerance =SAME_LANE_LATERAL_TOL,
                        min_gap=SAME_LANE_MIN_GAP
                    )
                    if sa["same_lane"] and sa["should_stop"]:
                        rear = sa["behind_id"]
                        front = sa["ahead_id"]
                        # Ensure leader always GO
                        if any(rear in z and lem.get_leader(c)==rear for (c,_,z) in zones_control): continue
                        actions_dict[rear] = STOP
                        print(f"***Policy8***[lane-global] {rear}=STOP behind {front}; gap={sa['gap']:.2f} m")

            # Safety: Policy 7: Global PET
            for i in range(len(ids_all)):
                for j in range(i + 1, len(ids_all)):
                    a, b = ids_all[i], ids_all[j]
                    sa = samples_all.get(a, [])
                    sb = samples_all.get(b, [])
                    if env.robots[a].linear_velocity < 0.01 or env.robots[b].linear_velocity < 0.01:
                        print(f"***Policy7***[PET-global] SKIP {a} vs {b} — one is stopped.")
                        continue
                    res = PET_Monitor.pet_from_samples(sa, sb, safety_radius=PET_SAFETY_R)
                    if res is None:
                        print(f"***Policy7***[PET-global] SAFE between {a} and {b}")
                        continue
                    pet, ta, tb = res
                    if pet < PET_TOL:
                        later = b if tb > ta else a
                        actions_dict[later] = STOP
                        print(f"***Policy7***[PET-global] {later}=STOP vs {a if later == b else b} PET={pet:.2f}s")

            # Protect leaders and control-zone vehicles
            vehicles_in_control_zone: set[str] = set()
            for (_c, _r, vs) in zones_control:
                vehicles_in_control_zone.update(vs)

            # # Policy 10: Global same-lane spacing (for non-control-zone veh)
            # free_ids = [aid for aid in env.agent_ids if aid not in vehicles_in_control_zone]
            #
            # for i in range(len(free_ids)):
            #     for j in range(i + 1, len(free_ids)):
            #         a, b = free_ids[i], free_ids[j]
            #         sa = same_lane_safety(
            #             im, a, env.robots[a], b, env.robots[b],
            #             position_tolerance=SAME_LANE_LATERAL_TOL,
            #             min_gap=SAME_LANE_MIN_GAP
            #         )
            #         if sa["same_lane"] and sa["should_stop"]:
            #             rear = sa["behind_id"];
            #             front = sa["ahead_id"]
            #             actions_dict[rear] = STOP
            #             print(f"***Policy10***[lane-global] {rear}=STOP behind {front}; gap={sa['gap']:.2f} m")

            # Final check: leader GO override
            for (cx, cy), _, vs in zones_control:
                L = lem.get_leader((cx, cy))
                if L in vs:
                    actions_dict[L] = GO

            # Step the environment
            obs, rewards, terminated, truncated, infos = env.step(actions_dict)

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

                    queue = fifo.priority_queue.get((cx, cy), {})
                    for cid in queue:
                        print(f"  - {cid}: priority={queue[cid]}")

    print(f"\nTest completed! Total steps: {total_steps}")
    # time.sleep(0.5)
    env.close()

def is_done(terminated, truncated):
    return any(terminated.values()) or any(truncated.values())


def show_results(env, terminated, total_steps, truncated):
    print(f"\nEpisode ended at step {total_steps}!")
    for agent_id in env.agent_ids:
        print(f"  {agent_id}: Terminated={terminated[agent_id]}, Truncated={truncated[agent_id]}")


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

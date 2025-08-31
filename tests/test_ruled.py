"""
Test script for the Multi-Agent PID Road Network Gym Environment.
Demonstrates multi-agent discrete action space {STOP, GO}.
"""
import numpy as np
import sys
import os
np.random.seed(42)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from examples.multi_agent_gym_env import make_multi_agent_env
from duckietown_simulator.Model.intersection_manager_V2 import IntersectionManager, find_intersection_centers
from duckietown_simulator.Model.trajectory_conflict import (waypoints_map,
                                                            build_samples_from_waypoints,
                                                            build_aligned_samples_from_waypoints,
                                                            same_lane_safety )
from duckietown_simulator.Model.leader_election_manager import LeaderElectionManager
from demos.demo_pid_road_network import create_map_from_json
from duckietown_simulator.Model.post_encroachment_time import PostEncroachmentTimeMonitor
import wandb

# Weights & Biases configuration
wandb.init(mode="disabled")
# wandb.init(project="demo-project", name="run-minimal")

def debug_vtl_inputs(center, winners, waypoints_map, *, dt = 0.1, horizon = 2.5, zone_radius = 1.5, pad = 0.2):
    """Print why VTL might be False: missing keys, empty waypoints, or no points entering zone."""
    cx, cy = center
    ids = IntersectionManager._extract_ids_from_winners(winners)

    print(f"[debug] winners ids: {ids}")
    missing = [i for i in ids if i not in waypoints_map]
    empties = [i for i in ids if i in waypoints_map and not waypoints_map[i]]
    print(f"[debug] missing in wp_map: {missing}")
    print(f"[debug] empty waypoint lists: {empties}")

    present_ids = [i for i in ids if i in waypoints_map and waypoints_map[i]]
    if not present_ids:
        print("[debug] no valid ids with waypoints -> VTL will be False")
        return

    # Build samples and count points inside the zone
    samples = build_samples_from_waypoints(waypoints_map, present_ids, dt = dt, horizon = horizon)
    r2 = (zone_radius + pad) ** 2

    def first_entry_time(series):
        for (t, x, y) in series:
            if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                return t
        return None

    counts = {rid: sum((x - cx) ** 2 + (y - cy) ** 2 <= r2 for (_, x, y) in samples.get(rid, []))
              for rid in present_ids}
    first_in = {rid: first_entry_time(samples.get(rid, [])) for rid in present_ids}

    print(f"[debug] samples in zone (<= (R+pad)): {counts}")
    print(f"[debug] first entry times (s): {first_in}  (None means never enters within horizon={horizon})")

def test_multi_agent_basic():
    """Test basic multi-agent functionality."""
    print("Testing Multi-Agent PID Road Network Gym Environment")
    print("=" * 60)

    # Create environment
    traj_dict = {
        "robot1": "../data/looped_trajectory_11.json", #Ambulance
        "robot2": "../data/looped_trajectory_2.json", # Bus
        "robot3": "../data/looped_trajectory_3.json",
        "robot4": "../data/looped_trajectory_4.json",
        "robot5": "../data/looped_trajectory_5.json", #Bus
        "robot6": "../data/looped_trajectory_6.json",
        "robot7": "../data/looped_trajectory_7.json", #Ambulance
    }

    env = make_multi_agent_env(
        num_agents=7,
        trajectory_files= traj_dict,
        render_mode='human'  # Use human rendering for visualization
    )

    STOP, GO = 0, 1
    print(f"[mapping] STOP={STOP}, GO={GO}")

    print(f"Agent IDs: {env.agent_ids}")
    print(f"Action spaces: {env.action_space}")
    print(f"Number of agents: {env.num_agents}")

    # Load the Duckietown map and compute the intersection centers
    map_object = create_map_from_json(
        "/Users/ruijiang/PycharmProjects/gym-tsl-duckietown/duckietown_simulator/assets/maps/road_network.json")
    intersection_centers_world = find_intersection_centers(map_instance = map_object, x_offset=0.25, y_offset=1.5)
    print(f"[centers] coords: {intersection_centers_world}")

    im = IntersectionManager(map_object, default_radius=0.8)
    lem = LeaderElectionManager(centers=intersection_centers_world, exit_radius=0.5)


    # Reset environment
    obs, infos = env.reset()
    PET_Monitor = PostEncroachmentTimeMonitor(
        cell_size = 0.25,
        pet_threshold = 0.5,
        max_event_age = 0.3,
        dt = 0.1
    )
    print(f"Initial observations shape: {list(obs.keys())} -> {[obs[k].shape for k in obs.keys()]}")
    print(f"Observation details for robot1: {obs['robot1']}")

    total_steps = 0
    SELECTION_RADIUS = im.default_radius # 0.99
    CONTROL_RADIUS = lem.exit_radius # 0.5
    PET_TOL = 0.20
    PET_SAFETY_R = 0.20  # meters
    SAME_LANE_LATERAL_TOL = 0.20
    SAME_LANE_MIN_GAP = 0.25
    num_steps = 10000

    PET_DT = 0.1
    PET_HORIZON = 2.5

    while True:
        print(f"\n--- {num_steps} steps ---")

        for step in range(num_steps):

            if env.render_mode == 'human':
                if env.renderer.paused:
                    env.render()
                    continue

            # Safety: PET Sample Update
            samples_all = build_aligned_samples_from_waypoints(
                waypoints_map = waypoints_map,
                robots = env.robots,
                ids = env.agent_ids,
                dt = 0.3,
                horizon = 2.5,
            )

            # Clear leaders who left
            exited = lem.update_and_get_exited(env.robots, exit_radius = CONTROL_RADIUS)
            for c in exited:
                print(f"******************** previous leader left @ {c} -> needs reselection")

            zones_select = im.in_the_intersection_zone(env.robots, intersection_centers_world, radius = SELECTION_RADIUS)
            zones_control = im.in_the_intersection_zone(env.robots, intersection_centers_world, radius = CONTROL_RADIUS)

            winners_map = im.winners_by_zone(
                env.robots,
                zones_select,
                radius = SELECTION_RADIUS,
                filter_to_zone_members = True,
            )

            # Policy1: Base policy: everyone GO by default
            actions_dict = {agent_id: GO for agent_id in env.agent_ids}

            # Record leaders
            leaders_this_step = []

            # Safety: Policy2 - Global PET
            ids_all = list(env.agent_ids)

            # PET overlay: stop the later-arriving robot when PET is too small
            for i in range(len(ids_all)):
                for j in range(i + 1, len(ids_all)):
                    a, b = ids_all[i], ids_all[j]
                    sa = samples_all.get(a, [])
                    sb = samples_all.get(b, [])
                    res = PET_Monitor.pet_from_samples(sa, sb, safety_radius=PET_SAFETY_R)
                    if res is None:
                        print(f"***Policy5***res PASS between {a} and {b}")
                        continue
                    pet, ta, tb = res
                    if pet < PET_TOL:
                        later = b if tb > ta else a
                        actions_dict[later] = STOP
                        print(f"***Policy2***[PET-global] {later}=STOP vs {a if later == b else b} PET={pet:.2f}s")

            # Random Leader Election
            for (cx, cy), _, vehicles in zones_control:
                center = (cx, cy)
                if not vehicles:
                    continue

                # Safety: Policy3 Gap-based (control zone)
                veh_list = list(vehicles)
                for i in range(len(veh_list)):
                    for j in range(i + 1, len(veh_list)):
                        ida, idb = veh_list[i], veh_list[j]
                        sa = same_lane_safety(
                            im, ida, env.robots[ida], idb, env.robots[idb],
                            position_tolerance=SAME_LANE_LATERAL_TOL, min_gap=SAME_LANE_MIN_GAP
                        )
                        if not sa["same_lane"] or not sa["should_stop"]:
                            print(f"***Policy3*** robot{ida} and {idb} don't share the same lane ")
                            continue
                        rear = sa["behind_id"];
                        front = sa["ahead_id"]
                        # Leader GO
                        zone_leader = lem.get_leader(center)
                        target = rear if rear != zone_leader else front
                        if target and target in vehicles:
                            actions_dict[target] = STOP
                            print(
                                f"***Policy3*** [lane] @({cx:.2f},{cy:.2f}) {target}=STOP (spacing) behind {front}, gap={sa['gap']:.2f}")
                        else:
                            print(f'***Policy3*** @({cx: .2f},{cy: .2f}) {target} no need to stop')

                # Filter out winners - front candidates (distinct directions), at most 3
                front_lists = im.vehicle_in_the_front(env.robots, centers=[(cx, cy)], radius = SELECTION_RADIUS)
                print(f"step: {total_steps}front lists: {front_lists}")

                winners = winners_map.get(center, [])
                winner_ids = im._extract_ids_from_winners(winners)
                assert all(isinstance(w, str) for w in winner_ids), f"bad winners: {winners}"
                print(f"step: {total_steps} winner_ids: {winner_ids} @ {center}")

                leader_id = lem.get_leader(center)
                print(f"step:{total_steps} leader_id: {leader_id} @ {center}")
                # Safety: Policy4 stopped winner (front) veh behind STOP in the control zone
                for wid in winner_ids:
                    for rid in env.agent_ids:
                        if rid == wid or rid == leader_id:
                            continue
                        try:
                            lane_info = same_lane_safety(im, wid, env.robots[wid], rid, env.robots[rid],
                                                         position_tolerance = PET_SAFETY_R, min_gap = PET_TOL)

                            if lane_info["same_lane"] and lane_info["behind_id"] == rid:
                                actions_dict[rid] = STOP
                                print(
                                    f"***Policy4*** Vehicle {rid} is also stopped because it is behind the stopped winner {wid}.")
                        except Exception as e:
                            print(f"***4*** Error checking same lane for {wid} and {rid}: {e}")

                if (leader_id is None) or (leader_id not in vehicles):
                    lem.clear_leader(center)

                    if len(winner_ids) < 1:
                        debug_vtl_inputs(
                            center, winners, waypoints_map,
                            dt = PET_DT, horizon = PET_HORIZON,
                            zone_radius = SELECTION_RADIUS, pad = 0.2
                        )
                        print(f"[leader] @({cx:.2f},{cy:.2f}) not enough winner available; skipping election")
                        continue
                    else:
                        # Choose among winners
                        leader_id = np.random.choice(winner_ids)
                        lem.set_leader(center, leader_id)
                        print(f"[leader] @({cx:.2f},{cy:.2f}) elected {leader_id} from winners")
                else:
                    for rid in vehicles:
                        actions_dict[rid] = STOP
                        print(f"step: {total_steps} {rid} STOP due to non-leader @ {center}")
                    continue

                if leader_id in vehicles:
                    actions_dict[leader_id] = GO
                    print(f"step: {total_steps} leader{leader_id} @ {center} -> GO")

            # Final leader GO override
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
                print("  Leaders selected (random among front candidates):")
                for L in leaders_this_step:
                    cx, cy = L["center"]
                    cand_str = ", ".join(f"{cid}({sec}, d={dist:.2f})" for cid, sec, dist in L["candidates"])
                    print(f"    @({cx:.2f},{cy:.2f}) -> {L['leader']} from {L['sector']} | candidates: {cand_str}")

    print(f"\nTest completed! Total steps: {total_steps}")
    # time.sleep(0.5)
    env.close()

if __name__ == "__main__":
    # Test basic multi-agent functionality
    test_multi_agent_basic()

    print("\nAll multi-agent tests completed!")

from pathlib import Path
import numpy as np
import imageio
import importlib
import h5py
import os
import argparse
import pickle
import copy
from collections import defaultdict

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.bddl_utils import robosuite_parse_problem   
from libero.libero.envs import OffScreenRenderEnv
from utils.shared_utils import get_libero_dummy_action, is_noop, quat2axisangle
from libero.libero.envs.bddl_base_domain import TASK_MAPPING

task_ids_per_suite = {
    #'libero_10': [9, 0, 1, 2, 7],
    #'libero_90': [21],
    'libero_single': [0, 1, 2, 3, 4, 5, 6, 7, 8]
}

standard_to_low = {

    'put both the alphabet soup and the tomato sauce in the basket': [ 
        'put the alphabet soup in the basket',
        'put the tomato sauce in the basket',
    ],
    'put both the cream cheese box and the butter in the basket': [ 
        'put the cream cheese box in the basket',
        'put the butter in the basket',
    ],
    'turn on the stove and put the moka pot on it': [ 
        'turn on the stove',
        'put the moka pot on the stove',
    ],
    'turn on the stove and put the frying pan on it': [
        'turn on the stove',
        'put the frying pan on the stove',
    ],
    'put the black bowl in the bottom drawer of the cabinet and close it': [ 
        'put the black bowl in the bottom drawer of the cabinet',
        'close the bottom drawer of the cabinet'
    ],
    'put the white mug on the left plate and put the yellow and white mug on the right plate': [ 
        'put the white mug on the left plate',
        'put the yellow and white mug on the right plate'
    ],
    'pick up the book and place it in the back compartment of the caddy': [ 
        'pick up the book and place it in the back compartment of the caddy'
    ],
    'put the white mug on the plate and put the chocolate pudding to the right of the plate': [ 
        'put the white mug on the plate',
        'put the chocolate pudding to the right of the plate'
    ],
    'put both the alphabet soup and the cream cheese box in the basket': [
        'put the alphabet soup in the basket',
        'put the cream cheese box in the basket',
    ],

    'put the yellow and white mug in the microwave and close it': [ 
        'put the yellow and white mug in the microwave',
        'close the microwave'
    ],
    'turn on the stove and put the frying pan on it': [
        'turn on the stove',
        'put the frying pan on the stove'
    ],
    "put the black bowl in the bottom drawer of the cabinet and close it": [
        "close the bottom drawer of the cabinet",
        "put the black bowl in the bottom drawer of the cabinet",
    ]
}

standard_to_high = {
    'put both the alphabet soup and the tomato sauce in the basket': 'put both the alphabet soup and the tomato sauce in the basket',
    'put the yellow and white mug in the microwave and close it': 'microwave the yellow and white mug',
    'turn on the stove and put the moka pot on it': 'heat up the moka pot on the stove',
    'put both the cream cheese box and the butter in the basket': 'put both the cream cheese box and the butter in the basket',
    'turn on the stove and put the frying pan on it': 'heat up the frying pan on the stove',
    "put the black bowl in the bottom drawer of the cabinet and close it": "put away the black bowl in the bottom drawer of the cabinet",
    'put both the alphabet soup and the cream cheese box in the basket': 'put both the alphabet soup and the cream cheese box in the basket'
}

standard_to_conj = {
    'put both the alphabet soup and the tomato sauce in the basket': 'put the alphabet soup in the basket, then put the tomato sauce in the basket',
    'put both the alphabet soup and the cream cheese box in the basket': 'put the alphabet soup in the basket, then put the cream cheese box in the basket',
    'put the yellow and white mug in the microwave and close it': 'put the yellow and white mug in the microwave, then close the microwave',
    'turn on the stove and put the moka pot on it': 'turn on the stove, then put the moka pot on the stove',
    "put the black bowl in the bottom drawer of the cabinet and close it": "put the black bowl in the bottom drawer of the cabinet, then close the bottom drawer of the cabinet",
    'put both the cream cheese box and the butter in the basket': 'put the cream cheese box in the basket, then put the butter in the basket',
    'turn on the stove and put the frying pan on it': 'turn on the stove, then put the frying pan on the stove'
}


def process_libero(args):

    trajs_per_task = 50
    resolution = 256

    hl_ll_dataset = []

    datasets_default_path = get_libero_path("datasets")

    counts_per_task = defaultdict(int)

    for suite_name in task_ids_per_suite.keys():
        suite   = benchmark.get_benchmark_dict()[suite_name]()
        print("suite: ", suite)
        #print(suite.tasks)
        #print(suite.tasks[0])
        #print("task mapping: ", TASK_MAPPING)
        for task_id in task_ids_per_suite[suite_name]:

            task = suite.get_task(task_id)
            if suite_name == "libero_single":
                data_file_folder = task.name.split("SCENE5_")[1]
                data_file_path = os.path.join("/ubc/cs/research/nlp/grigorii/projects/libero_dataset", data_file_folder, "demo.hdf5")
                data_file = h5py.File(data_file_path, "r")
                full_path = os.path.join(args.bddl_files_path, "libero_single",  task.bddl_file)
                
            else:
                data_file = h5py.File(os.path.join("/ubc/cs/research/ubc_ml/gguz/datasets/libero", suite.get_task_demonstration(task_id)), "r")
                full_path = os.path.join(args.bddl_files_path, "libero_high_level_hard",  task.bddl_file)
                print("Full path: ", full_path)

            env_args = {
                "bddl_file_name": full_path, 
                "camera_heights": resolution, 
                "camera_widths": resolution
            }

            env = OffScreenRenderEnv(**env_args)
            demos_collected_for_task = 0
            for traj_id in range(trajs_per_task):
                if suite_name == "libero_single":
                    traj      = data_file["data"][f"demo_{traj_id+1}"]           
                    per_task_subgoal_states = len(env.env.parsed_problem['subgoal_states'])
                else:
                    traj      = data_file["data"][f"demo_{traj_id}"]
                    per_task_subgoal_states = 2   

                start_idx = 0  # index where the *next* segment starts
                orig_actions = traj['actions']

                segments  = []
                static_images = []
                gripper_images = []
                env_states = []
                robot_states = []
                actions = []
                num_noops = 0
                t_step_passed = 0
                subgoal_idx = 0

                env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
                env.reset()
                env.set_init_state(traj['states'][0])

                # Skipping first few timesteps (when the env is initialized, objects are dropping from the ceiling)
                for _ in range(10):
                    obs, reward, done, info = env.step(get_libero_dummy_action())

                subgoal_instructions = []
                current_subgoal_instruction = obs['subgoal_language']

                for t, act in enumerate(orig_actions):
                    prev_action = actions[-1] if len(actions) > 0 else None
                    # If "ungrasp_object", it's ok for root to be still (sim needs to step forward for objects to drop) 
                    if suite_name != "libero_single" or (env.env.current_subgoal_idx < per_task_subgoal_states - 1):

                        if is_noop(act, prev_action):
                            #print(f"\tSkipping no-op action: {act}")
                            num_noops += 1
                            continue

                    env_states.append(env.sim.get_state().flatten())
                    robot_states.append(
                        np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]])
                    )
                    actions.append(act)
                    static_images.append(obs["agentview_image"][::-1, ::-1])
                    gripper_images.append(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    obs, reward, done, info = env.step(act.tolist())
                    if 'inadmissible_task' in info and info['inadmissible_task'] != None:
                        print(f"Inadmissible task: {info['inadmissible_task']}, terminating the episode!")
                        done = False
                        break

                    if info['subgoal_completed']:
                        segments.append((subgoal_idx, start_idx, t_step_passed))
                        subgoal_idx += 1
                        start_idx = t_step_passed
                        subgoal_instructions.append(current_subgoal_instruction)
                        current_subgoal_instruction = obs['subgoal_language']
                    if done:
                        break

                    t_step_passed += 1

                print(f"Demonstration {traj_id} for task {task.name}  – found {len(segments)} sub-tasks")
                print(segments)

                subtask_id = 0
                
                if done and len(segments) >= per_task_subgoal_states:
                    
                    print(f"Saving trajectory {traj_id}. Noop actions skipped: {num_noops}")
                    counts_per_task[task.name] += 1
                    last_tstep = segments[-1][2]
                    print(subgoal_instructions)

                    hl_ll_traj = {
                        'static_images': static_images[:last_tstep],
                        'gripper_images': gripper_images[:last_tstep],
                        'states': env_states[:last_tstep],
                        'actions': actions[:last_tstep],
                        'robot_states': robot_states[:last_tstep],
                        'hl_instruction': task.language, #standard_to_high[task.language],
                        'conj_instruction': ", then ".join(subgoal_instructions), #standard_to_conj[task.language],
                        'll_segments': [],
                        'll_instructions': []
                    }

                    for seg_id, t0, t1 in segments:

                        text = subgoal_instructions[seg_id] #standard_to_low[task.language][seg_id]
                        print(f"  [{t0:>3d} – {t1:>3d}]   {text}")
                        #offset = 15
                        #if seg_id == 0:
                        #    t1 += offset
                        #else:
                        #    t0 += offset

                        hl_ll_traj['ll_segments'].append((t0, t1))
                        hl_ll_traj['ll_instructions'].append(subgoal_instructions[seg_id])#standard_to_low[task.language][seg_id])

                        #if args.video_store_path:

                            #video_writer = imageio.get_writer(args.video_store_path + \
                            #        f"/instr_{standard_to_low[task.language][seg_id]}_traj_{traj_id}_seg_{subtask_id}.mp4", fps=60)
                            #video_writer = imageio.get_writer(args.video_store_path + \
                            #        f"/traj_{traj_id}_instr_{subgoal_instructions[seg_id]}_seg_{subtask_id}.mp4", fps=60)

                            #images = static_images[t0:t1]
                            #for image in images:
                            #    video_writer.append_data(image)
                            #video_writer.close()

                        subtask_id += 1
                    hl_ll_dataset.append(hl_ll_traj)
                elif len(segments) < per_task_subgoal_states:
                    print("Skipping the demonstration, not enough subtasks completed")
                elif not done:
                    print("Skipping the demonstration, task not completed")

                #if args.video_store_path:
                    #video_writer = imageio.get_writer(args.video_store_path + \
                    #        f"/instr_{standard_to_low[task.language][seg_id]}_traj_{traj_id}_seg_{subtask_id}.mp4", fps=60)
                #    video_writer = imageio.get_writer(args.video_store_path + \
                #            f"/traj_{traj_id}_instr_{task.language}.mp4", fps=60)

                #    images = static_images
                #    for image in images:
                #        video_writer.append_data(image)
                #    video_writer.close()

    print("Counts per task:" )
    print(counts_per_task)
    if args.save_dataset:
        print("Saving the dataset")
        with open(args.data_store_path + f'/libero_segmented_dataset.pkl', 'wb') as f:
            pickle.dump(hl_ll_dataset, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl_files_path", type=str, default="/ubc/cs/research/nlp/grigorii/projects/vla_subtask_generalization/LIBERO/libero/libero/bddl_files/")
    parser.add_argument("--data_store_path", type=str, default="/ubc/cs/research/nlp/grigorii/projects/libero_dataset")
    parser.add_argument("--video_store_path", type=str, default="/ubc/cs/research/nlp/grigorii/projects/vla_subtask_generalization/video_save_dir")
    parser.add_argument("--save_dataset", type=bool, default=True)
    args = parser.parse_args()
    print("Begin processing!")
    process_libero(args)



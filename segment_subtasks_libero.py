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
    'libero_10': [0, 1, 2, 7, 9],
    'libero_90': [21]
    #'libero_single': [0, 1, 2, 3, 4]
}

TEMPLATES = {
    "turnon":"turn on the {0}",  
    "TurnOff":"turn off the {0}",
    "Open":"open the {0}",       
    "Close":"close the {0}",
    "on":"put the {0} on the {1}",
    "In":"put the {0} in the {1}",
    "Stack":"stack the {0} on the {1}",
    "in":"put the {0} in the {1}"
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
    'put both the alphabet soup and the tomato sauce in the basket': 'put the alphabet soup in the basket and put the tomato sauce in the basket',
    'put both the alphabet soup and the cream cheese box in the basket': 'put the alphabet soup in the basket and put the cream cheese box in the basket',
    'put the yellow and white mug in the microwave and close it': 'put the yellow and white mug in the microwave and close the microwave',
    'turn on the stove and put the moka pot on it': 'turn on the stove and put the moka pot on the stove',
    "put the black bowl in the bottom drawer of the cabinet and close it": "put the black bowl in the bottom drawer of the cabinet and close the bottom drawer of the cabinet",
    'put both the cream cheese box and the butter in the basket': 'put the cream cheese box in the basket and put the butter in the basket',
    'turn on the stove and put the frying pan on it': 'turn on the stove and put the frying pan on the stove'
}

"""
def object_nice_name(raw):
    if raw in ['flat_stove_1', 'flat_stove_1_cook_region']:
        return 'stove'
    elif raw == 'moka_pot_1':
        return 'moka pot'
    elif raw == 'akita_black_bowl_1':
        return 'black bowl'
    elif raw == 'white_cabinet_1_bottom_region':
        return 'bottom drawer of the cabinet'
    elif raw == 'porcelain_mug_1':
        return 'white mug'
    elif raw == 'white_yellow_mug_1':
        return 'yellow and white mug'
    elif raw == 'plate_1':
        return 'left plate'
    elif raw == 'plate_2':
        return 'right plate'
    elif raw == 'chocolate_pudding_1':
        return 'chocolate pudding'
    elif raw == 'alphabet_soup_1':
        return 'alphabet soup'
    elif raw == 'cream_cheese_1':
        return 'cream cheese'
    elif raw == 'basket_1_contain_region':
        return 'basket'
    elif raw == 'desk_caddy_1_back_contain_region':
        return 'back compartment of the caddy'
    elif raw == 'black_book_1':
        return 'book'
    elif raw == 'living_room_table_plate_right_region':
        return 'to the right of the plate'
    elif raw == 'microwave_1_heating_region':
        return 'microwave'
    return raw.replace('_', ' ').strip()

def instr_for_atom(op, args):
    tpl = TEMPLATES.get(op, None)
    words = [object_nice_name(a) for a in args]
    return tpl.format(*words) if tpl else " ".join(words)

def load_goal_atoms(bddl_file: Path):
    problem = robosuite_parse_problem(str(bddl_file))
    # problem["goal_state"] is a list of S-expr tokens representing the whole (And …) clause
    # we only care about each top-level child
    atoms = []
    
    for expr in problem["goal_state"]:     # skip the initial 'And'
        op, *args = expr
        atoms.append((op, args))                  # e.g. ('Turnon', ['flat_stove_1'])
    atoms = atoms[:2]
    return atoms

def get_sim_obj(domain, name, require=()):
    #Return the wrapper whose *attribute set* satisfies the predicate's needs.

    candidate_tables = [
        getattr(domain, a) for a in dir(domain)
        if (a.endswith("_dict") or a.endswith("_table"))
           and isinstance(getattr(domain, a), dict)
    ]

    preferred = ["region", "contain", "fixture", "object_site", "object"]
    candidate_tables.sort(
        key=lambda tbl: min((preferred.index(k)
                             for k in preferred if k in tbl.__repr__()),
                            default=len(preferred)))

    for tbl in candidate_tables:
        obj = tbl.get(name)
        if obj is not None and all(hasattr(obj, m) for m in require):
            return obj

    need = ", ".join(require) or "—"
    raise KeyError(f"'{name}' with methods [{need}] not found in any *_dict/_table")


def get_libero_evaluators(env, full_path):
        #For a given HL task (specified by full_path to the HL task BDDL file),
        #extract the evaluator predicates for constituent subtasks.
        #For example: 
        #    "turn on the stove and put the pan on it" = TurnOn(stove) and IsClose(pan, stove)
        #    -> We extract  TurnOn(stove), IsClose(pan, stove) as separate evaluators
    dom   = env.env           
    pred_module = importlib.import_module(
                    "libero.libero.envs.predicates.base_predicates")
    atoms   = load_goal_atoms(full_path)   
    evaluators = []

    name_map = {                           # BDDL token  ->  class name
        "turnon": "TurnOn",
        "turnoff": "TurnOff",
        "open": "Open",
        "close": "Close",
        "on": "On",
        "in": "In",
    }

    needs = {
        "On"   : (("check_contact", "check_ontop"),              1),
        "In"   : (("check_contain",),                            1), 
        "Close": (("is_close",),                                 0),
        "Open" : (("is_close",),                                 0),
    }

    for op, names in atoms:               # e.g. ('On', ['moka', 'stove_region'])
        cls_name   = name_map[op]
        # Class 
        pred_cls   = getattr(pred_module, cls_name)

        # TODO: In needs, why is_close = 0 for both Close and Open?
        req, tgt = needs.get(cls_name, ((), None))

        objs = []
        for arg_idx, name in enumerate(names):
            must_have = req if arg_idx == tgt else ()

            objs.append(get_sim_obj(dom, name, require=must_have))

        predicate  = pred_cls()
        evaluators.append(lambda p=predicate, o=objs: p(*o))
    return evaluators
"""

old_libero_bddl_map = {

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
        print(suite.tasks)
        print(suite.tasks[0])
        print("task mapping: ", TASK_MAPPING)
        for task_id in task_ids_per_suite[suite_name]:

            task = suite.get_task(task_id)
            data_file = h5py.File(os.path.join(datasets_default_path, suite.get_task_demonstration(task_id)), 
                                  "r") 
            
            full_path = "/ubc/cs/research/nlp/grigorii/projects/vla_subtask_generalization/LIBERO/libero/libero/bddl_files/" + task.problem_folder + "/" + task.bddl_file
            print("Full path: ", full_path)
            env_args = {
                "bddl_file_name": full_path, 
                "camera_heights": resolution, 
                "camera_widths": resolution
            }
            env     = OffScreenRenderEnv(**env_args)
            #evaluators = get_libero_evaluators(env, full_path)
            #print(evaluators)
            for traj_id in range(trajs_per_task):
                traj      = data_file["data"][f"demo_{traj_id}"]           
                #active    = [False]*len(evaluators)

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
                    if is_noop(act, prev_action):
                        print(f"\tSkipping no-op action: {act}")
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
                    if info['subgoal_completed']:
                        segments.append((subgoal_idx, start_idx, t_step_passed))
                        subgoal_idx += 1
                        start_idx = t_step_passed
                        subgoal_instructions.append(current_subgoal_instruction)
                        current_subgoal_instruction = obs['subgoal_language']
                    if done:
                        break
                    #for i, ev in enumerate(evaluators):
                    #    if not active[i] and ev():
                    #        segments.append((i, start_idx, t_step_passed))
                    #        active[i] = True
                    #        start_idx = t_step_passed

                    t_step_passed += 1

                print(f"Demonstration {traj_id} for task {task.name}  – found {len(segments)} sub-tasks")
                print(segments)

                subtask_id = 0
                if done and len(segments) >= 2:

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

                        if args.video_store_path:

                            #video_writer = imageio.get_writer(args.video_store_path + \
                            #        f"/instr_{standard_to_low[task.language][seg_id]}_traj_{traj_id}_seg_{subtask_id}.mp4", fps=60)
                            video_writer = imageio.get_writer(args.video_store_path + \
                                    f"/traj_{traj_id}_instr_{subgoal_instructions[seg_id]}_seg_{subtask_id}.mp4", fps=60)

                            images = static_images[t0:t1]
                            for image in images:
                                video_writer.append_data(image)
                            video_writer.close()

                        subtask_id += 1
                    hl_ll_dataset.append(hl_ll_traj)
                elif len(segments) < 2:
                    print("Skipping the demonstration, not enough subtasks completed")
                elif not done:
                    print("Skipping the demonstration, task not completed")

                if args.video_store_path:
                    #video_writer = imageio.get_writer(args.video_store_path + \
                    #        f"/instr_{standard_to_low[task.language][seg_id]}_traj_{traj_id}_seg_{subtask_id}.mp4", fps=60)
                    video_writer = imageio.get_writer(args.video_store_path + \
                            f"/traj_{traj_id}_instr_{task.language}.mp4", fps=60)

                    images = static_images
                    for image in images:
                        video_writer.append_data(image)
                    video_writer.close()

        print("Counts per task:" )
        print(counts_per_task)
        if args.save_dataset:
            print("Saving the dataset")
            with open(args.data_store_path + f'/libero_segmented_dataset.pkl', 'wb') as f:
                pickle.dump(hl_ll_dataset, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_store_path", type=str, default="/home/gguz/scratch/datasets")
    parser.add_argument("--video_store_path", type=str, default="/home/gguz/scratch/results/videos_seg")
    parser.add_argument("--save_dataset", type=bool, default=True)
    args = parser.parse_args()
    process_libero(args)



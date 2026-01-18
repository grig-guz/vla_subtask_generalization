import contextlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import functools
from itertools import product
import logging
from operator import add
import random
import numpy as np
import sys
sys.path.append(".")

from utils.shared_utils import tasks, high_to_low_level_mappings, task_categories, temp_seed

logger = logging.getLogger(__name__)


def check_condition(state, condition):
    # 
    for k, v in condition.items():
        if isinstance(v, (str, int)):
            if not state[k] == v:
                return False
        elif isinstance(v, list):
            # Just logical or
            if not state[k] in v:
                return False
        else:
            raise TypeError
    return True


def update_state(state, effect):
    # Override state with task effects
    next_state = deepcopy(state)
    for k, v in effect.items():
        next_state[k] = v
    return next_state


def valid_task(curr_state, task):
    next_states = []
    for i, _task in enumerate(task):
        if check_condition(curr_state, _task["condition"]):
            next_state = update_state(curr_state, _task["effect"])
            next_states.append((i, next_state))
    return next_states



def check_sequence(state, seq):
    # 1) Make sure there exists a unique state satisfying a task
    # 2) No task repeats in the same sequence (may be bad for ungrasping/contacting?)
    task_indices = []
    for task_name in seq:
        states = valid_task(state, tasks[task_name])
        if len(states) != 1:
            return [], False
        task_indices.append(states[0][0])
        state = states[0][1]
    categories = [task_categories[name] for name in seq]
    return task_indices, len(categories) == len(set(categories))


def get_sequences_for_state2(args):
    state, num_sequences, i = args
    np.random.seed(i)
    seq_len = 5
    results = []
    while len(results) < num_sequences:
        seq = np.random.choice(list(tasks.keys()), size=seq_len, replace=False)
        task_indices, passed_check = check_sequence(state, seq)
        if passed_check:
            seq_low_level = convert_to_low_level(state, task_indices, seq)
            results.append(seq_low_level)
    return results

def convert_to_low_level(state, task_indices, seq):
    seq_low_level = []
    block_top = None
    block_bottom = None
    for i, high_level_task in enumerate(seq):
        all_subtask_sequences = high_to_low_level_mappings[high_level_task]
        if len(all_subtask_sequences) == 1:
            seq_low_level.append((high_level_task, all_subtask_sequences[0]))
        else:
            # For stacking and unstacking
            if high_level_task == 'stack_block':
                task_index = task_indices[i]
                pre_post_cond = tasks[high_level_task][task_index]
                stacked_blocks = list(pre_post_cond['effect'].keys())
                top_block_idx = random.sample([0, 1], k=1)[0]
                block_top = stacked_blocks[top_block_idx]
                block_bottom = stacked_blocks[int(not top_block_idx)]
                seq_low_level.append((high_level_task, ["grasp_" + block_top, "lift_grasped_block", 
                            "place_grasped_block_over_" + block_bottom, "ungrasp_block"]))
            elif high_level_task == 'unstack_block':
                task_index = task_indices[i]
                pre_post_cond = tasks[high_level_task][task_index]
                stacked_blocks = list(pre_post_cond['condition'].keys())
                if block_top == None:
                    raise ValueError("No blocks are stacked, can't unstack!")
                seq_low_level.append((high_level_task, ["grasp_" + block_top, 
                    "place_grasped_block_over_table", "ungrasp_block"]))
            else:
                raise ValueError('Unknown ambiguous task!')
            
    return seq_low_level

def flatten(t):
    return [tuple(item) for sublist in t for item in sublist]


@functools.lru_cache
def get_sequences_high2low_chain(num_sequences=1000, num_workers=None):
    possible_conditions = {
        "led": [0, 1],
        "lightbulb": [0, 1],
        "slider": ["right", "left"],
        "drawer": ["closed", "open"],
        "red_block": ["table", "slider_right", "slider_left"],
        "blue_block": ["table", "slider_right", "slider_left"],
        "pink_block": ["table", "slider_right", "slider_left"],
        "grasped": [0]
    }

    f = lambda l: l.count("table") in [1, 2] and l.count("slider_right") < 2 and l.count("slider_left") < 2
    value_combinations = filter(f, product(*possible_conditions.values()))
    
    initial_states = [dict(zip(possible_conditions.keys(), vals)) for vals in value_combinations]

    num_sequences_per_state = list(map(len, np.array_split(range(num_sequences), len(initial_states))))
    logger.info("Start generating evaluation sequences.")
    # set the numpy seed temporarily to 0
    with temp_seed(0):
        num_workers = 6#multiprocessing.cpu_count() if num_workers is None else num_workers
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = flatten(
                executor.map(
                    get_sequences_for_state2, zip(initial_states, num_sequences_per_state, range(len(initial_states)))
                )
            )
        #for state in zip(initial_states, num_sequences_per_state, range(len(initial_states))):
        #    get_sequences_for_state2(state)
        results = list(zip(
            np.repeat(initial_states, num_sequences_per_state), 
            ["" for _ in range(num_sequences)], 
            results)
        )
        np.random.shuffle(results)
    logger.info("Done generating evaluation sequences.")

    return results


def generate_pyhash_seeds():
    import pyhash
    hasher = pyhash.fnv1_32()
    results = get_sequences_high2low_chain(1000)
    seeds = {}

    for initial_state, _, seq in results:
        seed = hasher(str(initial_state.values()))
        init_state_idx = []
        for key, value in initial_state.items():
            init_state_idx.append(key)
            init_state_idx.append(value)

        seeds[tuple(init_state_idx)] = seed
    import pickle
    with open('utils/high2low_seeds', 'wb') as f:
        pickle.dump(seeds, f)

if __name__ == "__main__":
    generate_pyhash_seeds()
    print("getting sequences")
    results = get_sequences_high2low_chain(50)
    print("Done generating!")
    counters = [Counter() for _ in range(5)]  # type: ignore
    for initial_state, _, seq in results:
        for elm in seq:
            if 'stack' in elm[0]:
                print(initial_state)
                print(seq)
                print()
                break
        for i, task in enumerate(seq):
            high_level_task, subtasks = task
            counters[i][high_level_task] += 1

    for i, counter in enumerate(counters):
        print(f"Task {i+1}")
        print()
        for task, freq in sorted(counter.items(), key=lambda x: x[1], reverse=True):
            print(f"{task}: {freq / sum(counter.values()) * 100:.2f}")
        print()
        print()

    print("overall task probability:")
    all_counters = functools.reduce(add, counters)
    for task, freq in sorted(all_counters.items(), key=lambda x: x[1], reverse=True):
        print(f"{task}: {freq / sum(all_counters.values()) * 100:.2f}")

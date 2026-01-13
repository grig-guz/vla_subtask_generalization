import contextlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import functools
from itertools import product
import logging
from operator import add
import numpy as np

from utils.shared_utils import tasks, task_categories, temp_seed

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
    for _task in task:
        if check_condition(curr_state, _task["condition"]):
            next_state = update_state(curr_state, _task["effect"])
            next_states.append(next_state)
    return next_states


def get_sequences_for_state(state, num_sequences=None):
    state = deepcopy(state)

    seq_len = 5
    valid_seqs = [[] for x in range(seq_len)]
    with temp_seed(0):
        for step in range(seq_len):
            for task_name, task in tasks.items():
                if step == 0:
                    for next_state in valid_task(state, task):
                        valid_seqs[0].append([(task_name, next_state)])
                else:
                    for seq in valid_seqs[step - 1]:
                        curr_state = seq[-1][1]
                        for next_state in valid_task(curr_state, task):
                            valid_seqs[step].append([*seq, (task_name, next_state)])

        results = []
        result_set = []
        # set the numpy seed temporarily to 0

        for seq in np.random.permutation(valid_seqs[-1]):
            _seq = list(zip(*seq))[0]
            categories = [task_categories[name] for name in _seq]
            if len(categories) == len(set(categories)) and set(_seq) not in result_set:
                results.append(_seq)
                result_set.append(set(_seq))
    if num_sequences is not None:
        results = results[:num_sequences]
    return results


def check_sequence(state, seq):
    # 1) Make sure there exists a unique state satisfying a task
    # 2) No task repeats in the same sequence (may be bad for ungrasping/contacting?)
    for task_name in seq:
        states = valid_task(state, tasks[task_name])
        if len(states) != 1:
            return False
        state = states[0]
    categories = [task_categories[name] for name in seq]
    return len(categories) == len(set(categories))


def get_sequences_for_state2(args):
    state, num_sequences, i = args
    np.random.seed(i)
    seq_len = 5
    results = []
    while len(results) < num_sequences:
        seq = np.random.choice(list(tasks.keys()), size=seq_len, replace=False)
        #if "stack_block" in seq and state['red_block'] == 'table' and state['blue_block'] == 'table':
        #    breakpoint()
        if check_sequence(state, seq):
            results.append(seq)
    return results


def flatten(t):
    return [tuple(item.tolist()) for sublist in t for item in sublist]


@functools.lru_cache
def get_sequences(num_sequences=1000, num_workers=None):
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
    results = get_sequences(1000)
    seeds = {}

    for initial_state, _, seq in results:
        seed = hasher(str(initial_state.values()))
        init_state_idx = []
        for key, value in initial_state.items():
            init_state_idx.append(key)
            init_state_idx.append(value)

        seeds[tuple(init_state_idx)] = seed
    print(seeds)
    import pickle
    with open('utils/high_sequences_seeds', 'wb') as f:
        pickle.dump(seeds, f)

if __name__ == "__main__":
    generate_pyhash_seeds()
    
    print("getting sequences")
    results = get_sequences(500)
    print("Done generating!")
    counters = [Counter() for _ in range(5)]  # type: ignore
    for initial_state, _, seq in results:
        print(seq)
        for i, task in enumerate(seq):
            counters[i][task] += 1

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

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import functools
from itertools import product
import logging
import sys
sys.path.append(".")

from utils.multistep_sequences import valid_task, flatten, temp_seed
from utils.shared_utils import tasks_low_level

import numpy as np
from utils.calvin_utils import temp_seed

logger = logging.getLogger(__name__)

def check_sequence(state, seq):
    for task_name in seq:
        states = valid_task(state, tasks_low_level[task_name])
        if len(states) != 1:
            return False
        state = states[0]
    return True

def get_sequences_for_state2(args):
    state, num_sequences, i = args
    np.random.seed(i)
    seq_len = 5
    results = []

    while len(results) < num_sequences:
        seq = np.random.choice(list(tasks_low_level.keys()), size=seq_len, replace=False)
        if check_sequence(state, seq):
            results.append(seq)
    return results


@functools.lru_cache
def get_low_level_random_sequences(num_sequences=1000, num_workers=None):
    # An object is never in a drawer initially.
    possible_conditions = {
        "led": [0, 1],
        "lightbulb": [0, 1],
        "slider": ["right", "left"],
        "drawer": ["closed", "open"],
        "red_block": ["table", "slider_right", "slider_left"],
        "blue_block": ["table", "slider_right", "slider_left"],
        "pink_block": ["table", "slider_right", "slider_left"],
        "grasped": [0],
        "contact": [0],
        "red_block_lifted": [0],
        "blue_block_lifted": [0],
        "pink_block_lifted": [0],
    }

    f = lambda l: l.count("table") in [1, 2] and l.count("slider_right") < 2 and l.count("slider_left") < 2
    value_combinations = filter(f, product(*possible_conditions.values()))
    
    initial_states = [dict(zip(possible_conditions.keys(), vals)) for vals in value_combinations]
    num_sequences_per_state = list(map(len, np.array_split(range(num_sequences), len(initial_states))))
    logger.info("Start generating evaluation sequences.")

    with temp_seed(0):
        num_workers = 6#multiprocessing.cpu_count() if num_workers is None else num_workers
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = flatten(
                executor.map(
                    get_sequences_for_state2, zip(initial_states, num_sequences_per_state, range(len(initial_states)))
                )
            )
        results = list(zip(
            np.repeat(initial_states, num_sequences_per_state), 
            ["" for _ in range(num_sequences)], 
            results)
        )

        np.random.shuffle(results)
    logger.info("Done generating evaluation sequences.")
    """
    with temp_seed(0):
        num_workers = 6
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = flatten(
                executor.map(
                    get_sequences_for_state_low_level, zip(initial_states, num_sequences_per_state, range(len(initial_states)))
                )
            )
        results = list(zip(np.repeat(initial_states, num_sequences_per_state), results))
        np.random.shuffle(results)
    """
    logger.info("Done generating evaluation sequences.")

    return results



def generate_pyhash_seeds():
    import pyhash
    hasher = pyhash.fnv1_32()
    results = get_low_level_random_sequences(1000)
    seeds = {}

    for initial_state, _, seq in results:
        seed = hasher(str(initial_state.values()))
        init_state_idx = []
        for key, value in initial_state.items():
            init_state_idx.append(key)
            init_state_idx.append(value)

        seeds[tuple(init_state_idx)] = seed
    import pickle
    with open('utils/low_sequence_seeds', 'wb') as f:
        pickle.dump(seeds, f)

if __name__ == "__main__":
    generate_pyhash_seeds()
    print("getting sequences")
    results = get_low_level_random_sequences(100)
    high_level_counter = Counter()
    low_level_counter = Counter()
    
    for result in results:
        print(result[1])

    for initial_state, task_sequence in results:
        for subtask in task_sequence:
            low_level_counter[subtask] += 1

    #print("overall high level task probability:")
    #for task, freq in sorted(high_level_counter.items(), key=lambda x: x[1], reverse=True):
    #    print(f"{task}: {freq / sum(high_level_counter.values()) * 100:.2f}")
    #print()
    print("overall low level task probability:")
    for task, freq in sorted(low_level_counter.items(), key=lambda x: x[1], reverse=True):
        print(f"{task}: {freq / sum(low_level_counter.values()) * 100:.2f}")


from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import functools

from itertools import product
import logging

import sys
sys.path.append(".")

from utils.multistep_sequences import valid_task, flatten
import numpy as np
from utils.shared_utils import temp_seed

logger = logging.getLogger(__name__)


tasks_low_level = {
    # Description: Initially, the robot should not be holding anything. At the end, it should be holding the corresponding object.
    "grasp_red_block": [
        {"condition": {"red_block": "table", "grasped": 0}, "effect": {"grasped": "red_block"}},
        {"condition": {"red_block": "drawer", "drawer": "open", "grasped": 0}, "effect": {"grasped": "red_block"}},
        {"condition": {"red_block": "slider_left", "slider": "right", "grasped": 0}, "effect": {"grasped": "red_block"}},
        {"condition": {"red_block": "slider_right", "slider": "left", "grasped": 0}, "effect": {"grasped": "red_block"}},
    ],
    "grasp_blue_block": [
        {
            "condition": {"blue_block": "table", "grasped": 0}, 
            "effect": {"grasped": "blue_block"}
         },
        {
            "condition": {"blue_block": "drawer", "drawer": "open", "grasped": 0}, 
            "effect": {"grasped": "blue_block"}
        },
        {
            "condition": {"blue_block": "slider_left", "slider": "right", "grasped": 0}, 
            "effect": {"grasped": "blue_block"}
        },
        {"condition": {"blue_block": "slider_right", "slider": "left", "grasped": 0}, 
         "effect": {"grasped": "blue_block"}},
    ],
    "grasp_pink_block": [
        {
            "condition": {"pink_block": "table", "grasped": 0}, 
            "effect": {"grasped": "pink_block"}
        },
        {
            "condition": {"pink_block": "drawer", "drawer": "open", "grasped": 0}, 
            "effect": {"grasped": "pink_block"}
        },
        {
            "condition": {"pink_block": "slider_left", "slider": "right", "grasped": 0}, 
            "effect": {"grasped": "pink_block"}
        },
        {
            "condition": {"pink_block": "slider_right", "slider": "left", "grasped": 0}, 
            "effect": {"grasped": "pink_block"}
        },
    ],
    
    "grasp_slider": [{"condition": {"grasped": 0}, "effect": {"grasped": "slider"}}],
    "grasp_drawer": [{"condition": {"grasped": 0}, "effect": {"grasped": "drawer"}}],


    # Description: Initially, the robot should be holding an object. At the end, it shouldn't be holding anything.
    "ungrasp_block": [{"condition": {"grasped": ["red_block", "blue_block", "pink_block"]}, 
                       "effect": {"grasped": 0, "red_block_lifted": 0, "blue_block_lifted": 0, "pink_block_lifted": 0}
                       }],
    "ungrasp_slider": [{"condition": {"grasped": "slider"}, "effect": {"grasped": 0}}],
    "ungrasp_drawer": [{"condition": {"grasped": "drawer"}, "effect": {"grasped": 0}}],


    # Gripper placement
    "place_grasped_block_over_red_block": [
        {"condition": {"red_block": "table", 'blue_block_lifted': 1}, "effect": {'blue_block': 'red_block'}},
        {"condition": {"red_block": "table", 'pink_block_lifted': 1}, "effect": {'pink_block': 'red_block'}},
    ],
    "place_grasped_block_over_blue_block": [
        {"condition": {"blue_block": "table", 'red_block_lifted': 1}, "effect": {'red_block': 'blue_block'}},
        {"condition": {"blue_block": "table", 'pink_block_lifted': 1}, "effect": {'pink_block': 'blue_block'}},

    ],
    "place_grasped_block_over_pink_block": [
        {"condition": {"pink_block": "table", 'red_block_lifted': 1}, "effect": {'red_block': 'pink_block'}},
        {"condition": {"pink_block": "table", 'blue_block_lifted': 1}, "effect": {'blue_block': 'pink_block'}},
    ],

    "move_slider_left": [{"condition": {"slider": "right", "grasped": "slider"}, "effect": {"slider": "left"}}],
    "move_slider_right": [{"condition": {"slider": "left", "grasped": "slider"}, "effect": {"slider": "right"}}],

    "open_drawer": [{"condition": {"drawer": "closed", "grasped": "drawer"}, "effect": {"drawer": "open"}}],
    "close_drawer": [{"condition": {"drawer": "open", "grasped": "drawer"}, "effect": {"drawer": "closed"}}],

    # lifting
    "lift_grasped_block": [
        {"condition": {"grasped": "red_block", "red_block_lifted": 0}, "effect": {"red_block_lifted": 1}},
        {"condition": {"grasped": "blue_block", "blue_block_lifted": 0}, "effect": {"blue_block_lifted": 1}},
        {"condition": {"grasped": "pink_block", "pink_block_lifted": 0}, "effect": {"pink_block_lifted": 1}},
    ],

    # rotation
    "rotate_grasped_block_right": [
        {"condition": {"red_block": "table", "grasped": "red_block"}, "effect": {"red_block_lifted": 1}},
        {"condition": {"blue_block": "table", "grasped": "blue_block"}, "effect": {"blue_block_lifted": 1}},
        {"condition": {"pink_block": "table", "grasped": "pink_block"}, "effect": {"pink_block_lifted": 1}},
    ],

    "rotate_grasped_block_left": [
        {"condition": {"red_block": "table", "grasped": "red_block"}, "effect": { "red_block_lifted": 1}},
        {"condition": {"blue_block": "table", "grasped": "blue_block"}, "effect": { "blue_block_lifted": 1}},
        {"condition": {"pink_block": "table", "grasped": "pink_block"}, "effect": { "pink_block_lifted": 1}},
    ],

    # open/close (slider, drawer) -> grasp slider/drawer + 
    "place_grasped_block_over_drawer": [        
        {
            "condition": {"red_block": ["table", "slider_right", "slider_left", "blue_block", "pink_block"], "drawer": "open",  'red_block_lifted': 1},
            "effect": {"red_block": "drawer"},
        },
        {
            "condition": {"blue_block": ["table", "slider_right", "slider_left", "red_block", "pink_block"], "drawer": "open",  'blue_block_lifted': 1},
            "effect": {"blue_block": "drawer"},
        },
        {
            "condition": {"pink_block": ["table", "slider_right", "slider_left", "red_block", "blue_block"], "drawer": "open",  'pink_block_lifted': 1},
            "effect": {"pink_block": "drawer"},
        },
    ],

    "place_grasped_block_over_slider": [
        {
            "condition": {"red_block": ["table", "drawer", "blue_block", "pink_block"], "slider": "right",  'red_block_lifted': 1},
            "effect": {"red_block": "slider_left"},
        },
        {
            "condition": {"red_block": ["table", "drawer", "blue_block", "pink_block"], "slider": "left",  'red_block_lifted': 1},
            "effect": {"red_block": "slider_right"},
        },
        {
            "condition": {"blue_block": ["table", "drawer", "red_block", "pink_block"], "slider": "right", 'blue_block_lifted': 1},
            "effect": {"blue_block": "slider_left"},
        },
        {
            "condition": {"blue_block": ["table", "drawer", "red_block", "pink_block"], "slider": "left",  'blue_block_lifted': 1},
            "effect": {"blue_block": "slider_right"},
        },
        {
            "condition": {"pink_block": ["table", "drawer", "red_block", "blue_block"], "slider": "right",  'pink_block_lifted': 1},
            "effect": {"pink_block": "slider_left"},
        },
        {
            "condition": {"pink_block": ["table", "drawer", "red_block", "blue_block"], "slider": "left", 'pink_block_lifted': 1},
            "effect": {"pink_block": "slider_right"},
        },
    ],

    "place_grasped_block_over_table": [
        {
            "condition": {"red_block": ["drawer", "slider_right", "slider_left", "blue_block", "pink_block"],  'red_block_lifted': 1},
            "effect": {"red_block": "table"},
        },
        {
            "condition": {"blue_block": ["drawer", "slider_right", "slider_left", "red_block", "pink_block"],  'blue_block_lifted': 1},
            "effect": {"blue_block": "table"},
        },
        {
            "condition": {"pink_block": ["drawer", "slider_right", "slider_left", "red_block", "blue_block"],  'pink_block_lifted': 1},
            "effect": {"pink_block": "table"},
        },
    ],
}

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
        "red_block_lifted": [0],
        "blue_block_lifted": [0],
        "pink_block_lifted": [0],
    }

    f = lambda l: l.count("table") in [1, 2] and l.count("slider_right") < 2 and l.count("slider_left") < 2
    value_combinations = filter(f, product(*possible_conditions.values()))
    
    initial_states = [dict(zip(possible_conditions.keys(), vals)) for vals in value_combinations]
    num_sequences_per_state = list(map(len, np.array_split(range(num_sequences), len(initial_states))))
    print("Start generating evaluation sequences")

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
    print("getting sequences")
    results = get_low_level_random_sequences(1000)
    high_level_counter = Counter()
    low_level_counter = Counter()
    for result in results:
        print(result[2])

    for initial_state, _, task_sequence in results:
        for subtask in task_sequence:
            low_level_counter[subtask] += 1

    #print("overall high level task probability:")
    #for task, freq in sorted(high_level_counter.items(), key=lambda x: x[1], reverse=True):
    #    print(f"{task}: {freq / sum(high_level_counter.values()) * 100:.2f}")
    #print()
    num_tasks_total = sum(low_level_counter.values())
    print(f"overall low level task probability: (total task count: {num_tasks_total}")
    for task, freq in sorted(low_level_counter.items(), key=lambda x: x[1], reverse=True):
        print(f"{task}: {freq / num_tasks_total * 100:.2f}")


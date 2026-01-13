from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import functools

from itertools import product
import logging

import sys
sys.path.append(".")

from utils.multistep_sequences import valid_task, flatten
from copy import deepcopy
import numpy as np
from utils.shared_utils import temp_seed
from random import sample

logger = logging.getLogger(__name__)

"""
class RigidObject:

    def __init__(self, name, location):
        self.name = name
        self.location = location


class ArticulatedObject:
    def __init__(self, name, state):
        self.name = name
        self.state = state


class Drawer(ArticulatedObject):

    def __init__(self, name, state):
        self.name = name
        self.state = state
        if self.state not in ['open', 'closed']:
            raise Exception("Invalid state for drawer!")

class Slider(ArticulatedObject):

    def __init__(self, name, state):
        self.name = name
        self.state = state
        if self.state not in ['left', 'right']:
            raise Exception("Invalid state for slider!")
"""

class CalvinTask:

    def __init__(self, target_obj=None, target_loc=None, all_rigid_objs=[], all_art_objs=[], all_objs=[], all_locations=[]):
        self.target_obj = target_obj
        self.target_loc = target_loc
        self.all_rigid_objs = all_rigid_objs
        self.all_art_objs = all_art_objs
        self.all_objs = all_objs
        self.all_locations = all_locations

        self.and_preconds = {}
        self.or_preconds = []
        self.post_conds = {}

    def check_preconditions(self, state):
        for pred, val in self.and_preconds.items():
            if state[pred] != val:
                return False
        if self.or_preconds == []:
            return True
        else:
            for or_precond in self.or_preconds:
                all_true = True
                for pred, val in or_precond.items():
                    if state[pred] != val:
                        all_true = False
                        break
                if all_true:
                    return True
            return False

    def update_state(self, state):
        for pred, val in self.post_conds.items():
            state[pred] = val
        return state

    def __str__(self):
        raise Exception("Implement object naming!")


class GraspBlock(CalvinTask):

    def __init__(self, target_obj=None, target_loc=None, all_rigid_objs=[], all_art_objs=[], all_objs=[], all_locations=[]):
        super().__init__(target_obj, target_loc, all_rigid_objs, all_art_objs, all_objs, all_locations)

        if not self.target_obj:
            self.target_obj = np.random.choice(self.all_rigid_objs, size=1)[0]
        
        self.and_preconds = {
            "grasped": 0, 
        }

        self.or_preconds = [
            {
                self.target_obj: "drawer",
                "drawer": "open"
            },
            {
                self.target_obj: "slider_left",
                "slider": "right"
            },
            {
                self.target_obj: "slider_right",
                "slider": "left"
            }
        ]
        self.post_conds = {
            "grasped": self.target_obj
        }

    def check_preconditions(self, state):
        first_check = super().check_preconditions(state)
        if not first_check:
            # check that there's no other block on top 
            if state[self.target_obj] == 'table' and state['grasped'] == 0:
                for other_block in self.all_rigid_objs:
                    if other_block != self.target_obj and state[other_block] == self.target_obj:
                        return False
                return True

        return False
        

    def update_state(self, state):
        state = super().update_state(state)
        for other_block in self.all_rigid_objs:
            if other_block != self.target_obj and state[self.target_obj] == other_block:
                state[self.target_obj + "_lifted"] = 1
        return state
    

    def __str__(self):
        return "grasp_" + self.target_obj


class GraspArticulated(CalvinTask):

    def __init__(self, target_obj=None, target_loc=None, all_rigid_objs=[], all_art_objs=[], all_objs=[], all_locations=[]):
        super().__init__(target_obj, target_loc, all_rigid_objs, all_art_objs, all_objs, all_locations)

        if not target_obj:
            self.target_obj = np.random.choice(self.all_art_objs, size=1)[0]

        self.and_preconds = {
            "grasped": 0, 
        }

        self.or_preconds = []
        self.post_conds = {
            "grasped": self.target_obj
        }

    def __str__(self):
        return "grasp_" + self.target_obj


class Ungrasp(CalvinTask):

    def __init__(self, target_obj=None, target_loc=None, all_rigid_objs=[], all_art_objs=[], all_objs=[], all_locations=[]):
        super().__init__(target_obj, target_loc, all_rigid_objs, all_art_objs, all_objs, all_locations)

        self.and_preconds = {}
        self.or_preconds = [
            {"grasped": obj} for obj in self.all_objs
        ]
        self.post_conds = {
            "grasped": 0,
            "red_block_lifted": 0,
            "blue_block_lifted": 0,
            "pink_block_lifted": 0,
        }

    def check_preconditions(self, state):
        for obj in self.all_objs:
            if state['grasped'] == obj:
                return True
        return False

    def update_state(self, state):
        for obj in self.all_objs:
            if state['grasped'] == obj:
                state['grasped'] = 0
                self.target_obj = obj
                if obj in self.all_rigid_objs:
                    state[obj + "_lifted"] = 0
                return state


    def __str__(self):
        if self.target_obj in self.all_rigid_objs:
            return "ungrasp_block"
        else:
            return "ungrasp_" + self.target_obj


class PlaceGraspedBlockOver(CalvinTask):

    def __init__(self, target_obj=None, target_loc=None, all_rigid_objs=[], all_art_objs=[], all_objs=[], all_locations=[]):
        super().__init__(target_obj, target_loc, all_rigid_objs, all_art_objs, all_objs, all_locations)
        if not self.target_loc:
            self.target_loc = np.random.choice(self.all_locations, size=1)[0]
        
    def check_preconditions(self, state):
        for block in self.all_rigid_objs:
            if state[block + "_lifted"] == 1 and state[block] != self.target_loc:
                if self.target_loc in self.all_rigid_objs:
                    if state[self.target_loc] == 'table':
                        return True
                elif self.target_loc == "drawer":
                    if state['drawer'] == 'open':
                        return True
                elif self.target_loc in ["slider_left", "slider_right"]:
                    if (self.target_loc == "slider_left" and state["slider"] == "right") or (self.target_loc == "slider_right" and state["slider"] == "left"):
                        return True
                elif self.target_loc == 'table':
                    for other_block in self.all_rigid_objs:
                        if block != other_block and state[block] == other_block:
                            return True
                else:
                    raise Exception("Unknown placement location!")
        return False

    def update_state(self, state):
        for block in self.all_rigid_objs:
            if state[block + "_lifted"] == 1:
                state[block] = self.target_loc
                return state

    def __str__(self):
        if self.target_loc in ["slider_left", "slider_right"]:
            return "place_grasped_block_over_slider"
        return "place_grasped_block_over_" + self.target_loc


class LiftGraspedBlock(CalvinTask):

    def __init__(self, target_obj=None, target_loc=None, all_rigid_objs=[], all_art_objs=[], all_objs=[], all_locations=[]):
        super().__init__(target_obj, target_loc, all_rigid_objs, all_art_objs, all_objs, all_locations)
        # Just go over state, check if a block is grasped, then check if it is lifted.
        # That's it.

    def check_preconditions(self, state):
        for block in self.all_rigid_objs:
            if state['grasped'] == block and state[block + "_lifted"] == 0:
                return True
        return False
    
    def update_state(self, state):
        for block in self.all_rigid_objs:
            if state['grasped'] == block and state[block + "_lifted"] == 0:
                state[block + "_lifted"] = 1
                return state
            
    def __str__(self):
        return "lift_grasped_block"


class RotateGraspedBlock(CalvinTask):

    def __init__(self, target_obj=None, target_loc=None, all_rigid_objs=[], all_art_objs=[], all_objs=[], all_locations=[]):
        super().__init__(target_obj, target_loc, all_rigid_objs, all_art_objs, all_objs, all_locations)
        self.direction = np.random.choice(["left", "right"], size=1)[0]

    def check_preconditions(self, state):
        for block in self.all_rigid_objs:
            if state['grasped'] == block and state[block] == 'table':
                return True
        return False
    
    def update_state(self, state):
        for block in self.all_rigid_objs:
            if state['grasped'] == block:
                state[block + "_lifted"] = 1
                return state

    def __str__(self):
        return "rotate_grasped_block_" + self.direction


class MoveSlider(CalvinTask):

    def __init__(self, target_obj=None, target_loc=None, all_rigid_objs=[], all_art_objs=[], all_objs=[], all_locations=[]):
        super().__init__(target_obj, target_loc, all_rigid_objs, all_art_objs, all_objs, all_locations)
        self.direction = np.random.choice(["left", "right"], size=1)[0]

    def check_preconditions(self, state):
        if state["slider"] == self.direction or state["grasped"] != "slider":
            return False
        return True

    def update_state(self, state):
        state[self.direction] = self.direction
        return state

    def __str__(self):
        return "move_slider_" + self.direction


class MoveDrawer(CalvinTask):

    def __init__(self, target_obj=None, target_loc=None, all_rigid_objs=[], all_art_objs=[], all_objs=[], all_locations=[]):
        super().__init__(target_obj, target_loc, all_rigid_objs, all_art_objs, all_objs, all_locations)
        self.direction = np.random.choice(["open", "closed"], size=1)[0]

    def check_preconditions(self, state):
        if state["drawer"] == self.direction or state["grasped"] != "drawer":
            return False
        return True

    def update_state(self, state):
        state[self.direction] = self.direction
        return state

    def __str__(self):
        if self.direction == "open":
            return "open_drawer"
        else:
            return "close_drawer"


def get_sequences_for_state2(args):
    state, num_sequences, i = args
    print(f"num sequences: {num_sequences}")
    np.random.seed(i)
    seq_len = 5
    results = []
    # PlaceGraspedBlockOver for task diversity
    all_tasks = [GraspBlock, GraspArticulated, Ungrasp, PlaceGraspedBlockOver, PlaceGraspedBlockOver, LiftGraspedBlock, RotateGraspedBlock, MoveSlider, MoveSlider, MoveDrawer, MoveDrawer]
    all_rigid_objs = ["red_block", "blue_block", "pink_block"]
    all_articulated_objects = ["drawer", "slider"]
    all_objects = all_rigid_objs + all_articulated_objects
    all_locations = ["slider_left", "slider_right", "drawer", "table"] + all_rigid_objs

    while len(results) < num_sequences:
        seq = np.random.choice(all_tasks, size=seq_len, replace=False)
        seq = [cls(all_rigid_objs=all_rigid_objs, all_art_objs=all_articulated_objects, all_objs=all_objects, all_locations=all_locations) for cls in seq]
        if check_sequence(state, seq):
            new_seq = tuple([str(task) for task in seq])
            if 'place_grasped_block_over_table' in new_seq:
                print(new_seq)
            results.append(new_seq)
    return results


def check_sequence(state, seq, log_res=False):
    state_copy = deepcopy(state)
    for task in seq:
        if not task.check_preconditions(state_copy):
            return False
        #if log_res:
        #    print(str(task), state_copy)
        #    print(state_copy['grasped'], task.and_preconds)
        state_copy = task.update_state(state_copy)     
        #if log_res:
        #    print(state_copy)
    return True


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
    print("Num init states: ", len(initial_states))
    num_sequences_per_state = list(map(len, np.array_split(range(num_sequences), len(initial_states))))
    print(f"Start generating evaluation sequences, sequences per state: {num_sequences_per_state}, size: {len(num_sequences_per_state)}")

    with temp_seed(0):
        num_workers = 6#multiprocessing.cpu_count() if num_workers is None else num_workers
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(
                    get_sequences_for_state2, zip(initial_states, num_sequences_per_state, range(len(initial_states)))
                )
        
        res_flat = []
        for res in results:
            res_flat.extend(res)
        results = res_flat
        
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
    print("printing res")
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
    print(f"overall low level task probability: (total task count: {num_tasks_total})")
    for task, freq in sorted(low_level_counter.items(), key=lambda x: x[1], reverse=True):
        print(f"{task}: {freq / num_tasks_total * 100:.2f}")


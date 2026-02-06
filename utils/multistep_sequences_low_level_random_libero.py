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
import time

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

class LiberoTask:

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


class GraspRigid(LiberoTask):

    def __init__(self, target_obj=None, target_loc=None, all_rigid_objs=[], all_art_objs=[], all_objs=[], all_locations=[]):
        super().__init__(target_obj, target_loc, all_rigid_objs, all_art_objs, all_objs, all_locations)

        if not self.target_obj:
            self.target_obj = np.random.choice(self.all_rigid_objs, size=1)[0]
        
        self.and_preconds = {
            "grasped": 0, 
        }

        self.or_preconds = [
            {
                self.target_obj: "top_drawer",
                "top_drawer": "open"
            },
            {
                self.target_obj: "floor"
            }, 
            {
                self.target_obj: "cabinet"
            }
            #{
            #    self.target_obj: "middle_drawer",
            #    "middle_drawer": "open",
            #    "top_drawer": "closed"
            #},
        ]
        self.post_conds = {
            "grasped": self.target_obj
        }

    def check_preconditions(self, state):
        if state[self.target_obj] == "top_drawer":
            return False

        first_check = super().check_preconditions(state)
        
        if first_check:
            # check that there's no other objects on top 
            for other_obj in self.all_rigid_objs:
                if other_obj != self.target_obj and state[other_obj] == self.target_obj:
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


class GraspArticulated(LiberoTask):

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


class Ungrasp(LiberoTask):

    def __init__(self, target_obj=None, target_loc=None, all_rigid_objs=[], all_art_objs=[], all_objs=[], all_locations=[]):
        super().__init__(target_obj, target_loc, all_rigid_objs, all_art_objs, all_objs, all_locations)

        self.and_preconds = {}
        self.or_preconds = [
            {"grasped": obj} for obj in self.all_objs
        ]
        self.post_conds = {
            "grasped": 0,
        }
        for obj in self.all_rigid_objs:
            self.post_conds[obj + "_lifted"] = 0
            
        if not target_obj:
            self.target_obj = np.random.choice(self.all_objs, size=1)[0]

    def check_preconditions(self, state):
        return state['grasped'] == self.target_obj

    def update_state(self, state):
        state['grasped'] = 0
        for obj in self.all_rigid_objs:
            state[obj + "_lifted"] = 0
        return state

    def __str__(self):
        return "ungrasp_" + self.target_obj


class PlaceGraspedObjOver(LiberoTask):

    def __init__(self, target_obj=None, target_loc=None, all_rigid_objs=[], all_art_objs=[], all_objs=[], all_locations=[]):
        super().__init__(target_obj, target_loc, all_rigid_objs, all_art_objs, all_objs, all_locations)
        if not self.target_obj:
            self.target_obj = np.random.choice(self.all_rigid_objs, size=1)[0]

        if not self.target_loc:
            self.target_loc = np.random.choice(self.all_locations, size=1)[0]
        

    def check_preconditions(self, state):
        if state[self.target_obj + "_lifted"] == 1 and self.target_obj != self.target_loc and state[self.target_obj] != self.target_loc:
            if self.target_loc in ["plate", "bowl"]:
                if state[self.target_loc] in ["floor", "cabinet"]:
                    return True
            elif self.target_loc == "top_drawer":
                if state['top_drawer'] == 'open':
                    return True
            #elif self.target_loc == "middle_drawer":
            #    if state['top_drawer'] == 'closed' and state['middle_drawer'] == 'open':
            #        return True
            elif self.target_loc == 'cabinet':
                return True
            else:
                raise Exception("Unknown placement location!")
        return False

    def update_state(self, state):
        state[self.target_obj] = self.target_loc
        return state

    def __str__(self):
        return "place_" + self.target_obj + "_over_" + self.target_loc


class LiftGraspedObj(LiberoTask):

    def __init__(self, target_obj=None, target_loc=None, all_rigid_objs=[], all_art_objs=[], all_objs=[], all_locations=[]):
        super().__init__(target_obj, target_loc, all_rigid_objs, all_art_objs, all_objs, all_locations)
        # Just go over state, check if a block is grasped, then check if it is lifted.
        # That's it.
        if not self.target_obj:
            self.target_obj = np.random.choice(self.all_rigid_objs, size=1)[0]

    def check_preconditions(self, state):
        if state['grasped'] == self.target_obj and state[self.target_obj + "_lifted"] == 0:
            return True
        return False
    
    def update_state(self, state):
        state[self.target_obj + "_lifted"] = 1
        return state
            
    def __str__(self):
        return "lift_" + self.target_obj

class RotateGraspedObj(LiberoTask):

    def __init__(self, target_obj=None, target_loc=None, all_rigid_objs=[], all_art_objs=[], all_objs=[], all_locations=[]):
        super().__init__(target_obj, target_loc, all_rigid_objs, all_art_objs, all_objs, all_locations)
        # Just go over state, check if a block is grasped, then check if it is lifted.
        # That's it.
        if not self.target_obj:
            self.target_obj = np.random.choice(self.all_rigid_objs, size=1)[0]
        self.direction = np.random.choice(["left", "right"], size=1)[0]


    def check_preconditions(self, state):
        if state['grasped'] == self.target_obj:
            return True
        return False
    
    def update_state(self, state):
        state[self.target_obj + "_lifted"] = 1
        return state
            
    def __str__(self):
        return "rotated_low_" + self.direction + "_" + self.target_obj


class MoveDrawer(LiberoTask):

    def __init__(self, target_obj=None, target_loc=None, all_rigid_objs=[], all_art_objs=[], all_objs=[], all_locations=[]):
        super().__init__(target_obj, target_loc, all_rigid_objs, all_art_objs, all_objs, all_locations)
        self.direction = np.random.choice(["open", "closed"], size=1)[0]
        if target_obj == None:
            self.target_obj = np.random.choice(["top_drawer"], size=1)[0]

    def check_preconditions(self, state):
        if state["ketchup"] == "top_drawer":
            return False
        
        if state[self.target_obj] == self.direction or state["grasped"] != self.target_obj:
            return False
        return True

    def update_state(self, state):
        state[self.target_obj] = self.direction
        return state

    def __str__(self):
        if self.direction == "open":
            return "open_low_" + self.target_obj
        else:
            return "close_low_" + self.target_obj


def get_sequences_for_state2(args):
    state, num_sequences, i = args
    print(f"num sequences: {num_sequences}")
    np.random.seed(i)
    seq_len = 5
    results = []
    # PlaceGraspedBlockOver for task diversity
    all_tasks = [GraspRigid, GraspRigid, GraspArticulated, RotateGraspedObj, Ungrasp, Ungrasp, PlaceGraspedObjOver, PlaceGraspedObjOver, PlaceGraspedObjOver, PlaceGraspedObjOver, PlaceGraspedObjOver, LiftGraspedObj, MoveDrawer, MoveDrawer]
    all_rigid_objs = ["bowl", "ketchup"]
    all_articulated_objects = ["top_drawer"]#, "middle_drawer"]
    all_objects = all_rigid_objs + all_articulated_objects
    all_locations = ["cabinet", "top_drawer", "plate", "bowl"]#, "middle_drawer"]

    while len(results) < num_sequences:
        seq = np.random.choice(all_tasks, size=seq_len, replace=False)
        seq = [cls(all_rigid_objs=all_rigid_objs, all_art_objs=all_articulated_objects, all_objs=all_objects, all_locations=all_locations) for cls in seq]
        
        if check_sequence(state, seq):
            new_seq = tuple([str(task) for task in seq])
            results.append(new_seq)
    return results


def check_sequence(state, seq, log_res=False):
    state_copy = deepcopy(state)
    #print(state_copy)
    #print([str(elm) for elm in seq])
    for task in seq:
        if not task.check_preconditions(state_copy):
            #print("Fail at: ", task)
            #print()
            #print()
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
        "top_drawer": ["closed", "open"],
        #"middle_drawer": ["closed", "open"],
        "bowl": ["floor"],
        "plate": ["floor"],
        "ketchup": ["floor"],
        "grasped": [0],
        "bowl_lifted": [0],
        "ketchup_lifted": [0],
    }

    f = lambda l: l.count("open") in [0, 1]
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

def store_sequences_init_states(store_path, results):
    from libero.libero.envs import OffScreenRenderEnv
    

    env_args = {"bddl_file_name": "/home/gguz/projects/aip-vshwartz/gguz/vla_subtask_generalization/LIBERO/libero/libero/bddl_files/libero_single/KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet.bddl", 
                    "camera_heights": 256, 
                    "camera_widths": 256}
    env_drawer_open = OffScreenRenderEnv(**env_args)
    env_drawer_open.seed(0)

    env_args = {"bddl_file_name": "/home/gguz/projects/aip-vshwartz/gguz/vla_subtask_generalization/LIBERO/libero/libero/bddl_files/libero_single/KITCHEN_SCENE5_open_the_top_drawer_of_the_cabinet.bddl", 
                    "camera_heights": 256, 
                    "camera_widths": 256}
    env_drawer_closed = OffScreenRenderEnv(**env_args)
    env_drawer_closed.seed(0)

    results_states = []    
    for initial_state, sub, seq in results:

        if initial_state["top_drawer"] == "open":
            env = env_drawer_open
        else:
            env = env_drawer_closed
        #before_reset = time.time()
        env.reset()
        #print(f"time to reset the env: {time.time() - before_reset}")
        state = env.env.sim.get_state().flatten()
        #print(f"time to reset the env and get the state: {time.time() - before_reset}")
        results_states.append((initial_state, sub, seq, state))
    
    import pickle
    with open(store_path, 'wb') as f:
        pickle.dump(results_states, f)


if __name__ == "__main__":
    print("getting sequences")
    results = get_low_level_random_sequences(1050)
    store_path = "utils/libero_low_sequences_init_states"
    #store_sequences_init_states(store_path, results)

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


import numpy as np
import os
import robosuite.utils.transform_utils as T

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs.robots import *
from libero.libero.envs.utils import *
from libero.libero.envs.object_states import *
from libero.libero.envs.objects import *
from libero.libero.envs.regions import *
from libero.libero.envs.arenas import *
from libero.libero.envs.bddl_base_domain import BDDLBaseDomain

class BDDLSequentialBaseDomain(BDDLBaseDomain):
    """
    A base domain for parsing bddl files.
    """

    def __init__(
        self,
        bddl_file_name,
        *args,
        **kwargs,
    ):
        super().__init__(
            bddl_file_name=bddl_file_name,
            *args,
            **kwargs
        )
        assert 'subgoal_states' in self.parsed_problem and len(self.parsed_problem['subgoal_states']) > 0
        self.current_subgoal_idx = 0
        

    def parse_bddl(self):
        self.parsed_problem = BDDLUtils.robosuite_parse_problem(self.bddl_file_name)
        self._assert_problem_name()

    def _check_success_seq(self):
        """
        Check if the goal is achieved. Consider conjunction goals at the moment
        """
        goal_state = self.parsed_problem["subgoal_states"][self.current_subgoal_idx]
        result = True
        for state in goal_state:
            print("state: ", state)
            result = self._eval_predicate(state) and result
        return result

    def reset(self):
        self.current_subgoal_idx = 0
        return super().reset()

    def step(self, action):

        obs, reward, done, info = super().step(action)
        done = self._check_success()
        done_subgoal = self._check_success_seq()
        

        if done_subgoal:
            info['subgoal_completed'] = True
            self.current_subgoal_idx += 1
        else:
            info['subgoal_completed'] = False

        all_subgoals_done = self.current_subgoal_idx >= len(self.parsed_problem['subgoal_states'])
        if all_subgoals_done:
            obs['subgoal_language'] = ''
        else:
            obs['subgoal_language'] = self.parsed_problem['subgoal_instructions'][self.current_subgoal_idx]

        return obs, reward, done and all_subgoals_done, info


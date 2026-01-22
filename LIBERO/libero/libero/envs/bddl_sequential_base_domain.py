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
        self.t_step = 0
        self.task_to_inadm = {
            # LL task mappings
            "grasp_bowl": ["grasp_ketchup", "grasp_top_drawer"],
            "grasp_ketchup": ["grasp_bowl", "grasp_top_drawer"],
            "grasp_top_drawer": ["grasp_bowl", "grasp_ketchup",  "open_top_drawer", "close_top_drawer"],

            "ungrasp_bowl": [],
            "ungrasp_ketchup": [],
            "ungrasp_top_drawer": ["open_top_drawer", "close_top_drawer"],

            "lift_bowl": ["ungrasp_bowl"],
            "lift_ketchup": ["ungrasp_ketchup"],

            "push_top_drawer": ["grasp_ketchup", "grasp_bowl"],
            "pull_top_drawer": ["grasp_ketchup", "grasp_bowl"],

            "place_ketchup_over_plate": ["ungrasp_ketchup"],
            "place_ketchup_over_bowl": ["ungrasp_ketchup"],
            "place_ketchup_over_top_drawer": ["ungrasp_ketchup"],
            "place_ketchup_over_cabinet": ["ungrasp_ketchup"],

            "place_bowl_over_plate": ["ungrasp_bowl"],
            "place_bowl_over_top_drawer": ["ungrasp_bowl"],
            "place_bowl_over_cabinet": ["ungrasp_bowl"],

            # HL task mappings
            "close_top_drawer": ["grasp_bowl", "grasp_ketchup"],
            "open_top_drawer": ["grasp_bowl", "grasp_ketchup"],

            "put_bowl_in_top_drawer": ["grasp_ketchup", "grasp_top_drawer"], 
            "put_bowl_on_plate": ["grasp_ketchup", "grasp_top_drawer", "grasp_top_drawer"], 
            "put_bowl_on_cabinet": ["grasp_ketchup", "grasp_top_drawer", "grasp_top_drawer"], 

            "put_ketchup_in_top_drawer": ["grasp_bowl", "grasp_top_drawer"], 
            "put_ketchup_on_plate": ["grasp_bowl", "grasp_top_drawer"], 
            "put_ketchup_on_bowl": ["grasp_bowl", "grasp_top_drawer"], 
            "put_bowl_on_cabinet": ["grasp_ketchup", "grasp_top_drawer", "grasp_top_drawer"], 

            # Old experiment
            "turn_on_stove_3": ["grasp_pan", "grasp_moka_pot"],
            "put_pan_on_stove_3": ["grasp_moka_pot", "turn_on_stove_3"],
            "put_moka_pot_on_stove_3": ["grasp_pan", "turn_on_stove_3"],

            "put_yellow_white_mug_in_microwave_6": ["grasp_porcelain_mug", "close_microwave_6"],
            "close_microwave_6": ["grasp_porcelain_mug", "grasp_yellow_white_mug"],

            "put_right_moka_pot_on_stove_8": ["grasp_left_moka_pot"],
            "put_left_moka_pot_on_stove_8": ["grasp_moka_pot"],

            "put_alphabet_soup_in_basket_1": ["grasp_cream_cheese", "grasp_tomato_sauce", "grasp_ketchup"],
            "put_cream_cheese_in_basket_1": ["grasp_alphabet_soup", "grasp_tomato_sauce", "grasp_ketchup"],

            "put_alphabet_soup_in_basket_2": ["grasp_cream_cheese", "grasp_tomato_sauce", "grasp_ketchup", "grasp_milk", "grasp_orange_juice", "grasp_butter"],
            "put_tomato_sauce_in_basket_2": ["grasp_cream_cheese", "grasp_alphabet_soup", "grasp_ketchup", "grasp_milk", "grasp_orange_juice", "grasp_butter"],

            "put_cream_cheese_in_basket_2": ["grasp_alphabet_soup", "grasp_tomato_sauce", "grasp_ketchup", "grasp_milk", "grasp_orange_juice", "grasp_butter"],
            "put_butter_in_basket_2": ["grasp_cream_cheese", "grasp_alphabet_soup", "grasp_tomato_sauce", "grasp_ketchup", "grasp_milk", "grasp_orange_juice"],

        }

        self.tasks_to_inadm_validation = {
            
        }

        self.task_to_predicate = {

            # Old tasks
            "turn_on_stove_3": ["turnon", "flat_stove_1"],
            "close_microwave_6": ["close", "microwave_1"],
            "put_pan_on_stove_3": ["on", "chefmate_8_frypan_1", "flat_stove_1_cook_region"],
            "put_moka_pot_on_stove_3": ["on", "moka_pot_1", "flat_stove_1_cook_region"],

            "put_right_moka_pot_on_stove_8": ["on", "moka_pot_1", "flat_stove_1_cook_region"],
            "put_left_moka_pot_on_stove_8": ["on", "moka_pot_2", "flat_stove_1_cook_region"],

            "put_alphabet_soup_in_basket_1": ["in", "alphabet_soup_1", "basket_1_contain_region"],
            "put_cream_cheese_in_basket_1": ["in", "cream_cheese_1", "basket_1_contain_region"],

            "put_alphabet_soup_in_basket_2": ["in", "alphabet_soup_1", "basket_1_contain_region"],
            "put_tomato_sauce_in_basket_2": ["in", "tomato_sauce_1", "basket_1_contain_region"],
            
            "put_cream_cheese_in_basket_2": ["in", "cream_cheese_1", "basket_1_contain_region"],
            "put_butter_in_basket_2": ["in", "butter_1", "basket_1_contain_region"],

            # LL tasks:
            "grasp_bowl": ["grasped", "akita_black_bowl_1"],
            "grasp_ketchup": ["grasped", "ketchup_1"],
            "grasp_top_drawer": ["grasped", "white_cabinet_1_top_region"],

            "ungrasp_bowl": ["ungrasped", "akita_black_bowl_1"],
            "ungrasp_ketchup": ["ungrasped", "ketchup_1"],
            "ungrasp_top_drawer": ["ungrasped", "white_cabinet_1_top_region"],

            "lift_bowl": ["lifted", "akita_black_bowl_1"],
            "lift_ketchup": ["lifted", "ketchup_1"],

            "place_ketchup_over_plate": ["over", "ketchup_1", "plate_1"],
            "place_ketchup_over_bowl": ["over", "ketchup_1", "akita_black_bowl_1"],
            "place_ketchup_over_top_drawer": ["over", "ketchup_1", "white_cabinet_1_top_region"],
            "place_ketchup_over_cabinet": ["over", "ketchup_1", "white_cabinet_1_top_side"],

            "place_bowl_over_plate": ["over", "akita_black_bowl_1", "plate_1"],
            "place_bowl_over_top_drawer": ["over", "akita_black_bowl_1", "white_cabinet_1_top_region"],
            "place_bowl_over_cabinet": ["over", "akita_black_bowl_1", "white_cabinet_1_top_side"],

            "push_top_drawer": ["close", "white_cabinet_1_top_region"],
            "pull_top_drawer": ["open", "white_cabinet_1_top_region"],

            # HL tasks:
            "close_top_drawer": ["close", "white_cabinet_1_top_region"],
            "open_top_drawer": ["open", "white_cabinet_1_top_region"],

            "put_bowl_in_top_drawer": ["on", "akita_black_bowl_1", "white_cabinet_1_top_region"], 
            "put_bowl_on_plate": ["on", "akita_black_bowl_1", "plate_1"], 
            "put_bowl_on_cabinet": ["on", "akita_black_bowl_1", "white_cabinet_1_top_side"], 

            "put_ketchup_in_top_drawer": ["on", "ketchup_1", "white_cabinet_1_top_region"], 
            "put_ketchup_on_plate": ["on", "ketchup_1",  "plate_1"], 
            "put_ketchup_on_bowl": ["on", "ketchup_1",  "akita_black_bowl_1"], 
            "put_ketchup_on_cabinet": ["on", "ketchup_1", "white_cabinet_1_top_side"], 


            "grasp_pan": ["grasped", "chefmate_8_frypan_1"],
            "grasp_moka_pot": ["grasped", "moka_pot_1"],
            "grasp_left_moka_pot": ["grasped", "moka_pot_2"],
            "grasp_yellow_white_mug": ["grasped", "white_yellow_mug_1"],
            "grasp_porcelain_mug": ["grasped", "porcelain_mug_1"],
            "grasp_cream_cheese": ["grasped", "cream_cheese_1"],
            "grasp_tomato_sauce": ["grasped", "tomato_sauce_1"],
            "grasp_milk": ["grasped", "milk_1"],
            "grasp_orange_juice": ["grasped", "orange_juice_1"],
            "grasp_butter": ["grasped", "butter_1"],
            "grasp_stove": ["grasped", "flat_stove_1"],
            "grasp_alphabet_soup": ["grasped", "alphabet_soup_1"],
        }


    def predicate_to_task(self, predicate):
        if predicate[0] == "grasped":
            if predicate[1] == "akita_black_bowl_1":
                return "grasp_bowl"
            elif predicate[1] == "alphabet_soup_1":
                return "grasp_alphabet_soup"
            elif predicate[1] == "ketchup_1":
                return "grasp_ketchup"
            elif predicate[1] == "white_cabinet_1_top_region":
                return "grasp_top_drawer"
            elif predicate[1] == "chefmate_8_frypan_1":
                return "grasp_pan"
            elif predicate[1] == "moka_pot_1":
                return "grasp_moka_pot"
            elif predicate[1] == "moka_pot_2":
                return "grasp_left_moka_pot"
            elif predicate[1] == "white_yellow_mug_1":
                return "grasp_yellow_white_mug"
            elif predicate[1] == "porcelain_mug_1":
                return "grasp_porcelain_mug"
            elif predicate[1] == "cream_cheese_1":
                return "grasp_cream_cheese"
            elif predicate[1] == "tomato_sauce_1":
                return "grasp_tomato_sauce"
            elif predicate[1] == "milk_1":
                return "grasp_milk"
            elif predicate[1] == "orange_juice_1":
                return "grasp_orange_juice"
            elif predicate[1] == "butter_1":
                return "grasp_butter"
            elif predicate[1] == "flat_stove_1":
                return "grasp_stove"
            else:
                raise Exception(f"Grasping unknown object: {predicate[1]}")
        elif predicate[0] == "ungrasped":
            if predicate[1] == "akita_black_bowl_1":
                return "ungrasp_bowl"
            elif predicate[1] == "ketchup_1":
                return "ungrasp_ketchup"
            elif predicate[1] == "white_cabinet_1_top_region":
                return "ungrasp_top_drawer"
            else:
                raise Exception(f"Ungrasping unknown object: {predicate[1]}")
        elif predicate[0] == "lifted":
            if predicate[1] == "akita_black_bowl_1":
                return "lift_bowl"
            elif predicate[1] == "ketchup_1":
                return "lift_ketchup"
            else:
                raise Exception(f"Lifting unknown object: {predicate[1]}")
        elif predicate[0] == "open":
            return "open_top_drawer"
        elif predicate[0] == "close":
            if predicate[1] == "white_cabinet_1_top_region":
                return "close_top_drawer"
            elif predicate[1] == "microwave_1":
                return "close_microwave_6"
        elif predicate[0] == "over":
            if predicate[1] == "ketchup_1":
                if predicate[2] == "plate_1":
                    return "place_ketchup_over_plate"
                elif predicate[2] == "akita_black_bowl_1":
                    return "place_ketchup_over_bowl"
                elif predicate[2] == "white_cabinet_1_top_region":
                    return "place_ketchup_over_top_drawer"
                elif predicate[2] == "white_cabinet_1_top_side":
                    return "place_ketchup_over_cabinet"
                else:
                    raise Exception(f"Placing object: {predicate[1]} over unknown location {predicate[2]}")
            elif predicate[1] == "akita_black_bowl_1":
                if predicate[2] == "plate_1":
                    return "place_bowl_over_plate"
                elif predicate[2] == "white_cabinet_1_top_region":
                    return "place_bowl_over_top_drawer"
                elif predicate[2] == "white_cabinet_1_top_side":
                    return "place_bowl_over_cabinet"
                else:
                    raise Exception(f"Placing object: {predicate[1]} over unknown location {predicate[2]}")
            else:
                raise Exception(f"Placing unknown object: {predicate[1]}")
        elif predicate[0] == "on":
            if predicate[1] == "ketchup_1":
                if predicate[2] == "plate_1":
                    return "put_ketchup_on_plate"
                elif predicate[2] == "akita_black_bowl_1":
                    return "put_ketchup_on_bowl"
                elif predicate[2] == "white_cabinet_1_top_region":
                    return "put_ketchup_in_top_drawer"
                elif predicate[2] == "white_cabinet_1_top_side":
                    return "put_ketchup_on_cabinet"
                else:
                    raise Exception(f"Placing object: {predicate[1]} over unknown location {predicate[2]}")
            elif predicate[1] == "chefmate_8_frypan_1":
                if predicate[2] == "flat_stove_1_cook_region":
                    return "put_pan_on_stove_3"
            elif predicate[1] == "akita_black_bowl_1":
                if predicate[2] == "plate_1":
                    return "put_bowl_on_plate"
                elif predicate[2] == "white_cabinet_1_top_region":
                    return "put_bowl_in_top_drawer"
                elif predicate[2] == "white_cabinet_1_top_side":
                    return "put_bowl_on_cabinet"
                else:
                    raise Exception(f"Placing object: {predicate[1]} over unknown location {predicate[2]}")
            elif predicate[1] == "moka_pot_1":
                if predicate[2] == "flat_stove_1_cook_region":
                    return "put_moka_pot_on_stove_3"
            elif predicate[1] == "moka_pot_2":
                if predicate[2] == "flat_stove_1_cook_region":
                    return "put_left_moka_pot_on_stove"
            else:
                raise Exception(f"Placing unknown object: {predicate[1]}")
        elif predicate[0] == "in":
            if predicate[1] == "cream_cheese_1":
                if predicate[2] == "basket_1_contain_region":
                    return "put_cream_cheese_in_basket_2"
            elif predicate[1] == "butter_1":
                if predicate[2] == "basket_1_contain_region":
                    return "put_butter_in_basket_2"
            elif predicate[1] == "alphabet_soup_1":
                if predicate[2] == "basket_1_contain_region":
                    return "put_alphabet_soup_in_basket_2"
            elif predicate[1] == "tomato_sauce_1":
                if predicate[2] == "basket_1_contain_region":
                    return "put_tomato_sauce_in_basket_2"
            elif predicate[1] == "white_yellow_mug_1":
                if predicate[2] == "microwave_1_heating_region":
                    return "put_yellow_white_mug_in_microwave_6"

        elif predicate[0] == "turnon":
            if predicate[1] == "flat_stove_1":

                return "turn_on_stove_3"
            else:
                return Exception(f"Turning on unknown object: {predicate[1]}")
        else:
            raise Exception(f"Unknown task: {predicate[0]}")

        raise Exception("Something's wrong!")


    def parse_bddl(self):
        self.parsed_problem = BDDLUtils.robosuite_parse_problem(self.bddl_file_name)
        self._assert_problem_name()


    def _check_success(self):
        """
        Check if the goal is achieved. Consider conjunction goals at the moment
        """
        goal_state = self.parsed_problem["goal_state"]
        result = True
        for state in goal_state:
            result = self._eval_predicate(state) and result
        return result
    

    def _check_success_seq(self):
        """
        Check if the goal is achieved. 
        """
        goal_state = self.parsed_problem["subgoal_states"][self.current_subgoal_idx]
        all_true = True
        for state in goal_state:
            #print(state)
            this_result = self._eval_predicate(state)
            all_true = this_result and all_true

        return all_true
    

    def reset(self):
        self.current_subgoal_idx = 0
        self.t_step = 0
        return super().reset()


    def _pass_hard_eval(self):
        #print("Cur subtask: ", self.parsed_problem["subgoal_states"][self.current_subgoal_idx])
        task = self.predicate_to_task(self.parsed_problem["subgoal_states"][self.current_subgoal_idx][0])
        
        inadm_tasks = self.task_to_inadm[task]
        #print("Checking inadm tasks ", inadm_tasks, " for task ", task)
        for inadm in inadm_tasks:
            pred = self.task_to_predicate[inadm]
            if pred[1] not in self.object_states_dict:
                continue
            this_result = self._eval_predicate(pred)
            #print(pred, this_result)
            if this_result:
                return False, self.predicate_to_task(pred)

        return True, None


    def step(self, action, do_hard_validation=False):

        obs, reward, done, info = super().step(action)
        self.t_step += 1
        if self.t_step == 8:
            self.update_init_obj_poses()

        done = self._check_success()
        
        all_subgoals_done = self.current_subgoal_idx >= len(self.parsed_problem['subgoal_states'])


        if all_subgoals_done:
            obs['subgoal_language'] = ''
            info['subgoal_completed'] = False
            info["hard_eval_passed"] = True
            info["inadmissible_task"] = None
        else:
            done_subgoal = self._check_success_seq()
            hard_eval_passed, inadm_task = self._pass_hard_eval()
            info["hard_eval_passed"] = hard_eval_passed
            if hard_eval_passed:
                info["inadmissible_task"] = None
            else:
                info["inadmissible_task"] = inadm_task
                print("HARD EVAL FAILED, INADMISSIBLE TASK: ", inadm_task)

            if done_subgoal:
                info['subgoal_completed'] = True
                self.current_subgoal_idx += 1
                self.update_init_obj_poses()
            else:
                info['subgoal_completed'] = False
            

            if self.current_subgoal_idx >= len(self.parsed_problem['subgoal_states']):
                obs['subgoal_language'] = None
                all_subgoals_done = True
            else:
                obs['subgoal_language'] = self.parsed_problem['subgoal_instructions'][self.current_subgoal_idx]

        #print("Final result: ", done, all_subgoals_done)

        return obs, reward, done and all_subgoals_done, info


    def update_init_obj_poses(self):
        for obj in self.object_states_dict.values():
            if isinstance(obj, ObjectState):
                obj.set_init_pos()
                    
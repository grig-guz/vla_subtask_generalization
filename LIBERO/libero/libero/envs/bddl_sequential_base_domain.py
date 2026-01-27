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

GRASP_TOP_DRAWER = "grasp_top_drawer"
GRASP_KETCHUP = "grasp_ketchup"
GRASP_CREAM_CHEESE = "grasp_cream_cheese"
GRASP_BOWL = "grasp_bowl"

LIFT_BOWL = "lift_bowl"
LIFT_KETCHUP = "lift_ketchup"

CLOSE_LOW_TOP_DRAWER = "close_low_top_drawer"
OPEN_LOW_TOP_DRAWER = "open_low_top_drawer"
CLOSE_HIGH_TOP_DRAWER = "close_high_top_drawer"
OPEN_HIGH_TOP_DRAWER = "open_high_top_drawer"

CLOSE_STATE_TOP_DRAWER = "close_state_top_drawer"
OPEN_STATE_TOP_DRAWER = "open_state_top_drawer"

UNGRASP_BOWL = "ungrasp_bowl"
UNGRASP_TOP_DRAWER = "ungrasp_top_drawer"
UNGRASP_KETCHUP = "ungrasp_ketchup"
UNGRASP_CREAM_CHEESE = "ungrasp_cream_cheese"

PLACE_KETCHUP_OVER_PLATE = "place_ketchup_over_plate"
PLACE_KETCHUP_OVER_BOWL = "place_ketchup_over_bowl"
PLACE_KETCHUP_OVER_CABINET = "place_ketchup_over_cabinet"
PLACE_KETCHUP_OVER_TOP_DRAWER = "place_ketchup_over_top_drawer"

PLACE_BOWL_OVER_PLATE = "place_bowl_over_plate"
PLACE_BOWL_OVER_CABINET = "place_bowl_over_cabinet"
PLACE_BOWL_OVER_TOP_DRAWER = "place_bowl_over_top_drawer"

PUT_KETCHUP_ON_PLATE = "put_ketchup_on_plate"
PUT_KETCHUP_ON_BOWL = "put_ketchup_on_bowl"
PUT_KETCHUP_IN_TOP_DRAWER = "put_ketchup_in_top_drawer"
PUT_KETCHUP_ON_CABINET = "put_ketchup_on_cabinet"

PUT_BOWL_IN_TOP_DRAWER = "put_bowl_in_top_drawer"
PUT_BOWL_ON_PLATE = "put_bowl_on_plate"
PUT_BOWL_ON_CABINET = "put_bowl_on_cabinet"


GRASP_TOMATO_SAUCE = "grasp_tomato_sauce"
GRASP_MOKA_POT = "grasp_moka_pot"
GRASP_LEFT_MOKA_POT = "grasp_left_moka_pot"
GRASP_MILK = "grasp_milk"
GRASP_PAN = "grasp_pan"
GRASP_ALPHABET_SOUP = "grasp_alphabet_soup"
GRASP_ORANGE_JUICE = "grasp_orange_juice"
GRASP_BUTTER = "grasp_butter"
GRASP_PORCELAIN_MUG = "grasp_porcelain_mug"
GRASP_YELLOW_WHITE_MUG = "grasp_yellow_white_mug"
GRASP_STOVE = "grasp_stove"

TURN_ON_STOVE = "turn_on_stove_3"
CLOSE_MICROWAVE = "close_state_microwave_6"




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
        #assert 'subgoal_states' in self.parsed_problem and len(self.parsed_problem['subgoal_states']) > 0
        self.current_subgoal_idx = 0
        self.do_sequential_evaluation = True
        self.t_step = 0


        self.task_to_lang = {
            GRASP_TOP_DRAWER: "grasp the handle of the top drawer",
            GRASP_KETCHUP: "grasp the ketchup",
            GRASP_CREAM_CHEESE: "grasp the cream cheese",
            GRASP_BOWL: "grasp the black bowl",

            LIFT_BOWL: "lift the grasped object",
            LIFT_KETCHUP: "lift the grasped object",
            CLOSE_LOW_TOP_DRAWER: "pull the drawer out",
            OPEN_LOW_TOP_DRAWER: "push the drawer in",

            UNGRASP_TOP_DRAWER: "ungrasp the object",
            UNGRASP_BOWL: "ungrasp the object",
            UNGRASP_KETCHUP: "ungrasp the object",

            PLACE_KETCHUP_OVER_PLATE: "place the object over the plate",
            PLACE_KETCHUP_OVER_BOWL: "place the object over the black bowl",
            PLACE_KETCHUP_OVER_CABINET: "place the object over the top of the cabinet",
            PLACE_KETCHUP_OVER_TOP_DRAWER: "place the object over the top drawer of the cabinet",

            PLACE_BOWL_OVER_PLATE: "place the object over the plate",
            PLACE_BOWL_OVER_CABINET: "place the object over the top of the cabinet",
            PLACE_BOWL_OVER_TOP_DRAWER: "place the object over the top drawer of the cabinet",

        }

        self.task_to_inadm = {
            # LL task mappings
            GRASP_BOWL: [GRASP_KETCHUP, CLOSE_LOW_TOP_DRAWER, OPEN_LOW_TOP_DRAWER, GRASP_CREAM_CHEESE],
            GRASP_KETCHUP: [GRASP_CREAM_CHEESE, CLOSE_LOW_TOP_DRAWER, OPEN_LOW_TOP_DRAWER, GRASP_CREAM_CHEESE],
            GRASP_CREAM_CHEESE: [GRASP_BOWL, GRASP_KETCHUP, CLOSE_LOW_TOP_DRAWER, OPEN_LOW_TOP_DRAWER],
            UNGRASP_BOWL: [],
            UNGRASP_KETCHUP: [],
            UNGRASP_CREAM_CHEESE: [],

            GRASP_TOP_DRAWER: [GRASP_CREAM_CHEESE, GRASP_KETCHUP, GRASP_BOWL,  OPEN_LOW_TOP_DRAWER, CLOSE_LOW_TOP_DRAWER],
            UNGRASP_TOP_DRAWER: [OPEN_LOW_TOP_DRAWER, CLOSE_LOW_TOP_DRAWER],
            
            CLOSE_STATE_TOP_DRAWER: [],
            OPEN_STATE_TOP_DRAWER: [],
            CLOSE_LOW_TOP_DRAWER: [OPEN_LOW_TOP_DRAWER],
            OPEN_LOW_TOP_DRAWER: [CLOSE_LOW_TOP_DRAWER],

            CLOSE_HIGH_TOP_DRAWER: [GRASP_CREAM_CHEESE, GRASP_KETCHUP, GRASP_BOWL],
            OPEN_HIGH_TOP_DRAWER: [GRASP_CREAM_CHEESE, GRASP_KETCHUP, GRASP_BOWL],

            LIFT_BOWL: [UNGRASP_BOWL],
            LIFT_KETCHUP: [UNGRASP_KETCHUP],

            PLACE_KETCHUP_OVER_PLATE: [UNGRASP_KETCHUP],
            PLACE_KETCHUP_OVER_BOWL: [UNGRASP_KETCHUP],
            PLACE_KETCHUP_OVER_TOP_DRAWER: [UNGRASP_KETCHUP],
            PLACE_KETCHUP_OVER_CABINET: [UNGRASP_KETCHUP],

            "place_cream_cheese_over_plate": [UNGRASP_CREAM_CHEESE],
            "place_cream_cheese_over_bowl": [UNGRASP_CREAM_CHEESE],
            "place_cream_cheese_over_top_drawer": [UNGRASP_CREAM_CHEESE],
            "place_cream_cheese_over_cabinet": [UNGRASP_CREAM_CHEESE],

            PLACE_BOWL_OVER_PLATE: [UNGRASP_BOWL],
            PLACE_BOWL_OVER_TOP_DRAWER: [UNGRASP_BOWL],
            PLACE_BOWL_OVER_CABINET: [UNGRASP_BOWL],

            # HL task mappings
            PUT_BOWL_IN_TOP_DRAWER: [GRASP_KETCHUP, CLOSE_LOW_TOP_DRAWER, OPEN_LOW_TOP_DRAWER,], 
            PUT_BOWL_ON_PLATE: [GRASP_KETCHUP, CLOSE_LOW_TOP_DRAWER, OPEN_LOW_TOP_DRAWER,], 
            PUT_BOWL_ON_CABINET: [GRASP_CREAM_CHEESE, GRASP_KETCHUP, CLOSE_LOW_TOP_DRAWER, OPEN_LOW_TOP_DRAWER,], 

            PUT_KETCHUP_IN_TOP_DRAWER: [GRASP_CREAM_CHEESE, CLOSE_LOW_TOP_DRAWER, OPEN_LOW_TOP_DRAWER,], 
            PUT_KETCHUP_ON_PLATE: [GRASP_CREAM_CHEESE, CLOSE_LOW_TOP_DRAWER, OPEN_LOW_TOP_DRAWER,], 
            PUT_KETCHUP_ON_BOWL: [GRASP_CREAM_CHEESE, CLOSE_LOW_TOP_DRAWER, OPEN_LOW_TOP_DRAWER,], 

            # Old experiment
            TURN_ON_STOVE: [GRASP_PAN, GRASP_MOKA_POT],
            "put_pan_on_stove_3": [GRASP_MOKA_POT, TURN_ON_STOVE],
            "put_moka_pot_on_stove_3": [GRASP_PAN, TURN_ON_STOVE],

            "put_yellow_white_mug_in_microwave_6": [GRASP_PORCELAIN_MUG, CLOSE_MICROWAVE],
            CLOSE_MICROWAVE: [GRASP_PORCELAIN_MUG, GRASP_YELLOW_WHITE_MUG],

            "put_right_moka_pot_on_stove_8": [GRASP_LEFT_MOKA_POT],
            "put_left_moka_pot_on_stove_8": [GRASP_MOKA_POT],

            "put_alphabet_soup_in_basket_1": [GRASP_CREAM_CHEESE, GRASP_TOMATO_SAUCE, GRASP_KETCHUP],
            "put_cream_cheese_in_basket_1": [GRASP_ALPHABET_SOUP, GRASP_TOMATO_SAUCE, GRASP_KETCHUP],

            "put_alphabet_soup_in_basket_2": [GRASP_CREAM_CHEESE, GRASP_TOMATO_SAUCE, GRASP_KETCHUP, GRASP_MILK, GRASP_ORANGE_JUICE, GRASP_BUTTER],
            "put_tomato_sauce_in_basket_2": [GRASP_CREAM_CHEESE, GRASP_ALPHABET_SOUP, GRASP_KETCHUP, GRASP_MILK, GRASP_ORANGE_JUICE, GRASP_BUTTER],

            "put_cream_cheese_in_basket_2": [GRASP_ALPHABET_SOUP, GRASP_TOMATO_SAUCE, GRASP_KETCHUP, GRASP_MILK, GRASP_ORANGE_JUICE, GRASP_BUTTER],
            "put_butter_in_basket_2": [GRASP_CREAM_CHEESE, GRASP_ALPHABET_SOUP, GRASP_TOMATO_SAUCE, GRASP_KETCHUP, GRASP_MILK, GRASP_ORANGE_JUICE],

        }

        self.task_to_predicate = {
            CLOSE_STATE_TOP_DRAWER: ["close", "white_cabinet_1_top_region"],
            OPEN_STATE_TOP_DRAWER: ["open", "white_cabinet_1_top_region"],

            CLOSE_LOW_TOP_DRAWER: ["closelow", "white_cabinet_1_top_region"],
            OPEN_LOW_TOP_DRAWER: ["openlow", "white_cabinet_1_top_region"],

            # HL tasks:
            CLOSE_HIGH_TOP_DRAWER: ["closehigh", "white_cabinet_1_top_region"],
            OPEN_HIGH_TOP_DRAWER: ["openhigh", "white_cabinet_1_top_region"],

            # Old tasks
            TURN_ON_STOVE: ["turnon", "flat_stove_1"],
            CLOSE_MICROWAVE: ["close", "microwave_1"],
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
            GRASP_BOWL: ["grasped", "akita_black_bowl_1"],
            GRASP_KETCHUP: ["grasped", "ketchup_1"],
            GRASP_TOP_DRAWER: ["grasped", "white_cabinet_1_top_region"],

            UNGRASP_BOWL: ["ungrasped", "akita_black_bowl_1"],
            UNGRASP_KETCHUP: ["ungrasped", "ketchup_1"],
            UNGRASP_CREAM_CHEESE: ["ungrasped", "cream_cheese_1"],
            UNGRASP_TOP_DRAWER: ["ungrasped", "white_cabinet_1_top_region"],

            LIFT_BOWL: ["lifted", "akita_black_bowl_1"],
            LIFT_KETCHUP: ["lifted", "ketchup_1"],

            PLACE_KETCHUP_OVER_PLATE: ["over", "ketchup_1", "plate_1"],
            PLACE_KETCHUP_OVER_BOWL: ["over", "ketchup_1", "akita_black_bowl_1"],
            PLACE_KETCHUP_OVER_TOP_DRAWER: ["over", "ketchup_1", "white_cabinet_1_top_region"],
            PLACE_KETCHUP_OVER_CABINET: ["over", "ketchup_1", "white_cabinet_1_top_side"],

            PLACE_BOWL_OVER_PLATE: ["over", "akita_black_bowl_1", "plate_1"],
            PLACE_BOWL_OVER_TOP_DRAWER: ["over", "akita_black_bowl_1", "white_cabinet_1_top_region"],
            PLACE_BOWL_OVER_CABINET: ["over", "akita_black_bowl_1", "white_cabinet_1_top_side"],


            PUT_BOWL_IN_TOP_DRAWER: ["on", "akita_black_bowl_1", "white_cabinet_1_top_region"], 
            PUT_BOWL_ON_PLATE: ["on", "akita_black_bowl_1", "plate_1"], 
            PUT_BOWL_ON_CABINET: ["on", "akita_black_bowl_1", "white_cabinet_1_top_side"], 

            PUT_KETCHUP_IN_TOP_DRAWER: ["on", "ketchup_1", "white_cabinet_1_top_region"], 
            PUT_KETCHUP_ON_PLATE: ["on", "ketchup_1",  "plate_1"], 
            PUT_KETCHUP_ON_BOWL: ["on", "ketchup_1",  "akita_black_bowl_1"], 
            PUT_KETCHUP_ON_CABINET: ["on", "ketchup_1", "white_cabinet_1_top_side"], 


            GRASP_PAN: ["grasped", "chefmate_8_frypan_1"],
            GRASP_MOKA_POT: ["grasped", "moka_pot_1"],
            GRASP_LEFT_MOKA_POT: ["grasped", "moka_pot_2"],
            GRASP_YELLOW_WHITE_MUG: ["grasped", "white_yellow_mug_1"],
            GRASP_PORCELAIN_MUG: ["grasped", "porcelain_mug_1"],
            GRASP_CREAM_CHEESE: ["grasped", "cream_cheese_1"],
            GRASP_TOMATO_SAUCE: ["grasped", "tomato_sauce_1"],
            GRASP_MILK: ["grasped", "milk_1"],
            GRASP_ORANGE_JUICE: ["grasped", "orange_juice_1"],
            GRASP_BUTTER: ["grasped", "butter_1"],
            GRASP_STOVE: ["grasped", "flat_stove_1"],
            GRASP_ALPHABET_SOUP: ["grasped", "alphabet_soup_1"],
        }

    def set_seq_evaluation(self, do_sequential_evaluation):
        self.do_sequential_evaluation = do_sequential_evaluation

    def predicate_to_task(self, predicate):
        if predicate[0] == "grasped":
            if predicate[1] == "akita_black_bowl_1":
                return GRASP_BOWL
            elif predicate[1] == "alphabet_soup_1":
                return GRASP_ALPHABET_SOUP
            elif predicate[1] == "ketchup_1":
                return GRASP_KETCHUP
            elif predicate[1] == "white_cabinet_1_top_region":
                return GRASP_TOP_DRAWER
            elif predicate[1] == "chefmate_8_frypan_1":
                return GRASP_PAN
            elif predicate[1] == "moka_pot_1":
                return GRASP_MOKA_POT
            elif predicate[1] == "moka_pot_2":
                return GRASP_LEFT_MOKA_POT
            elif predicate[1] == "white_yellow_mug_1":
                return GRASP_YELLOW_WHITE_MUG
            elif predicate[1] == "porcelain_mug_1":
                return GRASP_PORCELAIN_MUG
            elif predicate[1] == "cream_cheese_1":
                return GRASP_CREAM_CHEESE
            elif predicate[1] == "tomato_sauce_1":
                return GRASP_TOMATO_SAUCE
            elif predicate[1] == "milk_1":
                return GRASP_MILK
            elif predicate[1] == "orange_juice_1":
                return GRASP_ORANGE_JUICE
            elif predicate[1] == "butter_1":
                return GRASP_BUTTER
            elif predicate[1] == "flat_stove_1":
                return GRASP_STOVE
            else:
                raise Exception(f"Grasping unknown object: {predicate[1]}")
        elif predicate[0] == "ungrasped":
            if predicate[1] == "akita_black_bowl_1":
                return UNGRASP_BOWL
            elif predicate[1] == "ketchup_1":
                return UNGRASP_KETCHUP
            elif predicate[1] == "cream_cheese_1":
                return UNGRASP_CREAM_CHEESE
            elif predicate[1] == "white_cabinet_1_top_region":
                return UNGRASP_TOP_DRAWER
            else:
                raise Exception(f"Ungrasping unknown object: {predicate[1]}")
        elif predicate[0] == "lifted":
            if predicate[1] == "akita_black_bowl_1":
                return GRASP_CREAM_CHEESE
            elif predicate[1] == "ketchup_1":
                return LIFT_KETCHUP
            elif predicate[1] == "cream_cheese_1":
                return GRASP_CREAM_CHEESE
            else:
                raise Exception(f"Lifting unknown object: {predicate[1]}")
        elif predicate[0] == "open":
            if predicate[1] == "white_cabinet_1_top_region":
                return OPEN_STATE_TOP_DRAWER
            else:
                raise Exception(f"Unknown object to open: {predicate[1]}")
        elif predicate[0] == "openlow":
            if predicate[1] == "white_cabinet_1_top_region":
                return OPEN_LOW_TOP_DRAWER
            else:
                raise Exception(f"Unknown object to open: {predicate[1]}")
        elif predicate[0] == "openhigh":
            if predicate[1] == "white_cabinet_1_top_region":
                return OPEN_HIGH_TOP_DRAWER
            else:
                raise Exception(f"Unknown object to open: {predicate[1]}")
        elif predicate[0] == "close":
            if predicate[1] == "white_cabinet_1_top_region":
                return CLOSE_STATE_TOP_DRAWER
            elif predicate[1] == "microwave_1":
                return CLOSE_MICROWAVE
            else:
                raise Exception(f"Unknown object to close: {predicate[1]}")
        elif predicate[0] == "closelow":
            if predicate[1] == "white_cabinet_1_top_region":
                return CLOSE_LOW_TOP_DRAWER
            else:
                raise Exception(f"Unknown object to close: {predicate[1]}")
        elif predicate[0] == "closehigh":
            if predicate[1] == "white_cabinet_1_top_region":
                return CLOSE_HIGH_TOP_DRAWER
            else:
                raise Exception(f"Unknown object to close: {predicate[1]}")
        elif predicate[0] == "over":
            if predicate[1] == "ketchup_1":
                if predicate[2] == "plate_1":
                    return PLACE_KETCHUP_OVER_PLATE
                elif predicate[2] == "akita_black_bowl_1":
                    return PLACE_KETCHUP_OVER_BOWL
                elif predicate[2] == "white_cabinet_1_top_region":
                    return PLACE_KETCHUP_OVER_TOP_DRAWER
                elif predicate[2] == "white_cabinet_1_top_side":
                    return PLACE_KETCHUP_OVER_CABINET
                else:
                    raise Exception(f"Placing object: {predicate[1]} over unknown location {predicate[2]}")
            elif predicate[1] == "akita_black_bowl_1":
                if predicate[2] == "plate_1":
                    return PLACE_BOWL_OVER_PLATE
                elif predicate[2] == "white_cabinet_1_top_region":
                    return PLACE_BOWL_OVER_TOP_DRAWER
                elif predicate[2] == "white_cabinet_1_top_side":
                    return PLACE_BOWL_OVER_CABINET
                else:
                    raise Exception(f"Placing object: {predicate[1]} over unknown location {predicate[2]}")
            elif predicate[1] == "cream_cheese_1":
                if predicate[2] == "plate_1":
                    return "place_cream_cheese_over_plate"
                elif predicate[2] == "akita_black_bowl_1":
                    return "place_cream_cheese_over_bowl"
                elif predicate[2] == "white_cabinet_1_top_region":
                    return "place_cream_cheese_over_top_drawer"
                elif predicate[2] == "white_cabinet_1_top_side":
                    return "place_cream_cheese_over_cabinet"
                else:
                    raise Exception(f"Placing object: {predicate[1]} over unknown location {predicate[2]}")
            else:
                raise Exception(f"Placing unknown object: {predicate[1]}")
        elif predicate[0] == "on":
            if predicate[1] == "ketchup_1":
                if predicate[2] == "plate_1":
                    return PUT_KETCHUP_ON_PLATE
                elif predicate[2] == "akita_black_bowl_1":
                    return PUT_KETCHUP_ON_BOWL
                elif predicate[2] == "white_cabinet_1_top_region":
                    return PUT_KETCHUP_IN_TOP_DRAWER
                elif predicate[2] == "white_cabinet_1_top_side":
                    return PUT_KETCHUP_ON_CABINET
                else:
                    raise Exception(f"Placing object: {predicate[1]} over unknown location {predicate[2]}")
            elif predicate[1] == "chefmate_8_frypan_1":
                if predicate[2] == "flat_stove_1_cook_region":
                    return "put_pan_on_stove_3"
            elif predicate[1] == "akita_black_bowl_1":
                if predicate[2] == "plate_1":
                    return PUT_BOWL_ON_PLATE
                elif predicate[2] == "white_cabinet_1_top_region":
                    return PUT_BOWL_IN_TOP_DRAWER
                elif predicate[2] == "white_cabinet_1_top_side":
                    return PUT_BOWL_ON_CABINET
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

                return TURN_ON_STOVE
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
            this_result = self._eval_predicate(state)
            all_true = this_result and all_true

        return all_true
    

    def reset(self):
        self.current_subgoal_idx = 0
        self.t_step = 0
        return super().reset()


    def _pass_hard_eval(self):
        """
            This should be redefined w.r.t. a task and a scene, not separately as in 
            _pass_hard_eval and _pass_hard_eval_validation
        """

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
                return False, inadm

        return True, None

    def _pass_hard_eval_validation(self):
        """
            This should be redefined w.r.t. a task and a scene, not separately as in 
            _pass_hard_eval and _pass_hard_eval_validation
        """

        if "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it" in self.bddl_file_name:
            inadm_tasks = [GRASP_MOKA_POT]
        elif "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it" in self.bddl_file_name:
            inadm_tasks = [GRASP_PAN]
        elif "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it" in self.bddl_file_name:
            inadm_tasks = [GRASP_PORCELAIN_MUG]
        elif "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove" in self.bddl_file_name:
            inadm_tasks = [TURN_ON_STOVE]
        elif "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket" in self.bddl_file_name:
            inadm_tasks = [GRASP_TOMATO_SAUCE, GRASP_KETCHUP, GRASP_MILK, GRASP_ORANGE_JUICE, GRASP_BUTTER]
        elif "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket" in self.bddl_file_name:
            inadm_tasks = [GRASP_CREAM_CHEESE, GRASP_KETCHUP, GRASP_MILK, GRASP_ORANGE_JUICE, GRASP_BUTTER]
        elif "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket" in self.bddl_file_name:
            inadm_tasks = [GRASP_TOMATO_SAUCE, GRASP_ALPHABET_SOUP, GRASP_KETCHUP, GRASP_MILK, GRASP_ORANGE_JUICE]
        elif "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove" in self.bddl_file_name:
            inadm_tasks = [TURN_ON_STOVE, GRASP_MOKA_POT]
        elif "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove" in self.bddl_file_name:
            inadm_tasks = [TURN_ON_STOVE, GRASP_PAN]
        elif "KITCHEN_SCENE3_turn_on_the_stove" in self.bddl_file_name:
            inadm_tasks = [GRASP_MOKA_POT, GRASP_PAN]
        elif "KITCHEN_SCENE6_close_the_microwave" in self.bddl_file_name:
            inadm_tasks = [GRASP_PORCELAIN_MUG, GRASP_YELLOW_WHITE_MUG]
        elif "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave" in self.bddl_file_name:
            inadm_tasks = [GRASP_PORCELAIN_MUG, CLOSE_MICROWAVE]
        elif "KITCHEN_SCENE8_put_the_left_moka_pot_on_the_stove" in self.bddl_file_name:
            inadm_tasks = [GRASP_MOKA_POT]
        elif "KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove" in self.bddl_file_name:
            inadm_tasks = [GRASP_LEFT_MOKA_POT]
        elif "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket" in self.bddl_file_name:
            inadm_tasks = [GRASP_TOMATO_SAUCE, GRASP_CREAM_CHEESE, GRASP_KETCHUP, GRASP_MILK, GRASP_ORANGE_JUICE, GRASP_BUTTER]
        elif "LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket" in self.bddl_file_name:
            inadm_tasks = [GRASP_TOMATO_SAUCE, GRASP_ALPHABET_SOUP, GRASP_KETCHUP, GRASP_MILK, GRASP_ORANGE_JUICE, GRASP_BUTTER]
        elif "LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket" in self.bddl_file_name:
            inadm_tasks = [GRASP_TOMATO_SAUCE, GRASP_CREAM_CHEESE, GRASP_KETCHUP, GRASP_MILK, GRASP_ORANGE_JUICE, GRASP_BUTTER]
        elif "LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket" in self.bddl_file_name:
            inadm_tasks = [GRASP_TOMATO_SAUCE, GRASP_CREAM_CHEESE, GRASP_KETCHUP, GRASP_MILK, GRASP_ORANGE_JUICE, GRASP_ALPHABET_SOUP]
        elif "LIVING_ROOM_SCENE2_pick_up_the_cream_cheese_box_and_put_it_in_the_basket" in self.bddl_file_name:
            inadm_tasks = [GRASP_TOMATO_SAUCE, GRASP_ALPHABET_SOUP, GRASP_KETCHUP, GRASP_MILK, GRASP_ORANGE_JUICE, GRASP_BUTTER]
        elif "LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket" in self.bddl_file_name:
            inadm_tasks = [GRASP_CREAM_CHEESE, GRASP_ALPHABET_SOUP, GRASP_KETCHUP, GRASP_MILK, GRASP_ORANGE_JUICE, GRASP_BUTTER]
        elif "KITCHEN_SCENE5" in self.bddl_file_name:
            goal_state = self.parsed_problem["goal_state"]
            if isinstance(goal_state[0], list):
                goal_state = goal_state[0]
            task = self.predicate_to_task(goal_state)
            inadm_tasks = self.task_to_inadm[task]
        else:
            raise Exception("Unknown bddl file")

        for inadm in inadm_tasks:
            pred = self.task_to_predicate[inadm]
            if pred[1] not in self.object_states_dict:
                continue
            this_result = self._eval_predicate(pred)
            if this_result:
                return False, inadm

        return True, None

    def all_subgoals_completed(self):
        return self.current_subgoal_idx >= len(self.parsed_problem['subgoal_states'])

    def step(self, action):            

        obs, reward, done, info = super().step(action)
        self.t_step += 1
        if self.t_step == 8:
            self.update_init_obj_poses()

        done = self._check_success()

        if not self.do_sequential_evaluation:
            all_subgoals_done = True
            info["hard_eval_passed"], info["inadmissible_task"] = self._pass_hard_eval_validation()
            return obs, reward, done, info
        else:
            obs['subgoal_language'] = None
            info['subgoal_completed'] = False
            info["hard_eval_passed"] = True
            info["inadmissible_task"] = None

            all_subgoals_done = self.all_subgoals_completed()
            
            if not all_subgoals_done:

                info["hard_eval_passed"], info["inadmissible_task"] = self._pass_hard_eval()
                if not info["hard_eval_passed"]:
                    print("HARD EVAL FAILED, INADMISSIBLE TASK: ", info["inadmissible_task"], " for task ", self.parsed_problem["subgoal_states"][self.current_subgoal_idx])

                info['subgoal_completed'] = self._check_success_seq()
                
                if info['subgoal_completed']:
                    print("Completed subgoal! ", self.parsed_problem["subgoal_states"][self.current_subgoal_idx], " done? ", done)
                    self.current_subgoal_idx += 1
                    self.update_init_obj_poses()                    
                
                all_subgoals_done = self.all_subgoals_completed()
                if not all_subgoals_done:
                    obs['subgoal_language'] = self.parsed_problem['subgoal_instructions'][self.current_subgoal_idx]

            return obs, reward, done and all_subgoals_done, info


    def update_init_obj_poses(self):
        for obj in self.object_states_dict.values():
            if isinstance(obj, ObjectState) or isinstance(obj, SiteObjectState):
                obj.set_init_pos()
                    
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
LIFT_CREAM_CHEESE = "lift_cream_cheese"

ROTATED_HIGH_LEFT_KETCHUP = "rotated_high_left_ketchup"
ROTATED_HIGH_LEFT_BOWL = "rotated_high_left_bowl"

ROTATED_LOW_LEFT_KETCHUP = "rotated_low_left_ketchup"
ROTATED_LOW_LEFT_BOWL = "rotated_low_left_bowl"


ROTATED_HIGH_RIGHT_KETCHUP = "rotated_high_right_ketchup"
ROTATED_HIGH_RIGHT_BOWL = "rotated_high_right_bowl"

ROTATED_LOW_RIGHT_KETCHUP = "rotated_low_right_ketchup"
ROTATED_LOW_RIGHT_BOWL = "rotated_low_right_bowl"


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
UNGRASP_STATE_BOWL = "ungrasp_state_bowl"
UNGRASP_STATE_KETCHUP = "ungrasp_state_ketchup"


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


MILK = "milk_1"
BOWL = "akita_black_bowl_1"
CABINET = "white_cabinet_1_top_side"
BASKET = "basket_1_contain_region"
STOVE_REGION = "flat_stove_1_cook_region"


STOVE = "flat_stove_1"
MICROWAVE = "microwave_1"
CREAM_CHEESE = "cream_cheese_1"
ORANGE_JUICE = "orange_juice_1"
TOP_DRAWER = "white_cabinet_1_top_region"
MOKA_POT_1 = "moka_pot_1"
MOKA_POT_2 = "moka_pot_2"
PLATE = "plate_1"
KETCHUP = "ketchup_1"
BUTTER = "butter_1"
SOUP = "alphabet_soup_1"
SAUCE = "tomato_sauce_1"
FRYING_PAN = "chefmate_8_frypan_1"
WHITE_YELLOW_MUG = "white_yellow_mug_1"
PORCELAIN_MUG = "porcelain_mug_1"

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

        self.task_to_subtasks = {
            CLOSE_HIGH_TOP_DRAWER:          [GRASP_TOP_DRAWER, CLOSE_LOW_TOP_DRAWER, UNGRASP_TOP_DRAWER],
            OPEN_HIGH_TOP_DRAWER:           [GRASP_TOP_DRAWER, OPEN_LOW_TOP_DRAWER, UNGRASP_TOP_DRAWER],
            PUT_KETCHUP_ON_PLATE:           [GRASP_KETCHUP, LIFT_KETCHUP, PLACE_KETCHUP_OVER_PLATE, UNGRASP_KETCHUP],
            PUT_KETCHUP_ON_CABINET:         [GRASP_KETCHUP, LIFT_KETCHUP, PLACE_KETCHUP_OVER_CABINET, UNGRASP_KETCHUP],
            PUT_KETCHUP_ON_BOWL:            [GRASP_KETCHUP, LIFT_KETCHUP, PLACE_KETCHUP_OVER_BOWL, UNGRASP_KETCHUP],
            PUT_KETCHUP_IN_TOP_DRAWER:      [GRASP_KETCHUP, LIFT_KETCHUP, PLACE_KETCHUP_OVER_TOP_DRAWER, UNGRASP_KETCHUP],
            PUT_BOWL_ON_CABINET:            [GRASP_BOWL, LIFT_BOWL, "place_bowl_over_cabinet", "ungrasp_bowl"],
            PUT_BOWL_ON_PLATE:              [GRASP_BOWL, LIFT_BOWL, "place_bowl_over_plate", "ungrasp_bowl"],
            PUT_BOWL_IN_TOP_DRAWER:         [GRASP_BOWL, LIFT_BOWL, "place_bowl_over_top_drawer", "ungrasp_bowl"],
            ROTATED_HIGH_LEFT_BOWL:         [GRASP_BOWL, ROTATED_LOW_LEFT_BOWL, UNGRASP_BOWL],
            ROTATED_HIGH_RIGHT_BOWL:        [GRASP_BOWL, ROTATED_LOW_RIGHT_BOWL, UNGRASP_BOWL],
            ROTATED_HIGH_LEFT_KETCHUP:      [GRASP_BOWL, ROTATED_LOW_LEFT_KETCHUP, UNGRASP_BOWL],
            ROTATED_HIGH_RIGHT_KETCHUP:     [GRASP_BOWL, ROTATED_LOW_RIGHT_KETCHUP, UNGRASP_BOWL],
        }

        self.task_to_lang = {
            GRASP_TOP_DRAWER:               "grasp the handle of the top drawer",
            GRASP_KETCHUP:                  "grasp the ketchup",
            GRASP_CREAM_CHEESE:             "grasp the cream cheese",
            GRASP_BOWL:                     "grasp the black bowl",
            LIFT_BOWL:                      "lift the grasped object",
            LIFT_KETCHUP:                   "lift the grasped object",
            CLOSE_LOW_TOP_DRAWER:           "push the drawer in",
            OPEN_LOW_TOP_DRAWER:            "pull the drawer out",

            ROTATED_LOW_LEFT_BOWL:          "rotate the grasped object 60 degrees to the left",
            ROTATED_LOW_LEFT_KETCHUP:       "rotate the grasped object 60 degrees to the left",
            ROTATED_LOW_RIGHT_BOWL:         "rotate the grasped object 60 degrees to the right",
            ROTATED_LOW_RIGHT_KETCHUP:      "rotate the grasped object 60 degrees to the right",

            ROTATED_HIGH_LEFT_BOWL:         "rotate the black bowl 60 degrees to the left",
            ROTATED_HIGH_RIGHT_BOWL:        "rotate the black bowl 60 degrees to the right",

            ROTATED_HIGH_LEFT_KETCHUP:      "rotate the ketchup 60 degrees to the left",
            ROTATED_HIGH_RIGHT_KETCHUP:     "rotate the ketchup 60 degrees to the right",

            UNGRASP_TOP_DRAWER:             "ungrasp the object",
            UNGRASP_BOWL:                   "ungrasp the object",
            UNGRASP_KETCHUP:                "ungrasp the object",

            PLACE_KETCHUP_OVER_PLATE:       "place the grasped object over the plate",
            PLACE_KETCHUP_OVER_BOWL:        "place the grasped object over the black bowl",
            PLACE_KETCHUP_OVER_CABINET:     "place the grasped object over the top of the cabinet",
            PLACE_KETCHUP_OVER_TOP_DRAWER:  "place the grasped object over the top drawer of the cabinet",

            PLACE_BOWL_OVER_PLATE:          "place the grasped object over the plate",
            PLACE_BOWL_OVER_CABINET:        "place the grasped object over the top of the cabinet",
            PLACE_BOWL_OVER_TOP_DRAWER:     "place the grasped object over the top drawer of the cabinet",
            

            PUT_BOWL_IN_TOP_DRAWER:         "put the black bowl in the top drawer of the cabinet", 
            "put_bowl_on_top_drawer":       "put the black bowl in the top drawer of the cabinet", 

            PUT_BOWL_ON_PLATE:              "put the black bowl on the plate", 
            PUT_BOWL_ON_CABINET:            "put the black bowl on top of the cabinet", 


            PUT_KETCHUP_IN_TOP_DRAWER:      "put the ketchup in the top drawer of the cabinet", 
            "put_ketchup_on_top_drawer":    "put the ketchup in the top drawer of the cabinet", 
            PUT_KETCHUP_ON_PLATE:           "put the ketchup on the plate",
            PUT_KETCHUP_ON_BOWL:            "put the ketchup on the bowl", 
            PUT_KETCHUP_ON_CABINET:         "put the ketchup on top of the cabinet",

            CLOSE_HIGH_TOP_DRAWER:          "close the top drawer of the cabinet",
            OPEN_HIGH_TOP_DRAWER:           "open the top drawer of the cabinet",

        }

        self.task_to_inadm = {
            # LL task mappings
            GRASP_BOWL:                     [GRASP_KETCHUP, CLOSE_LOW_TOP_DRAWER, OPEN_LOW_TOP_DRAWER, GRASP_CREAM_CHEESE],
            GRASP_KETCHUP:                  [GRASP_CREAM_CHEESE, GRASP_BOWL, CLOSE_LOW_TOP_DRAWER, OPEN_LOW_TOP_DRAWER],
            
            GRASP_CREAM_CHEESE:             [GRASP_BOWL, GRASP_KETCHUP, CLOSE_LOW_TOP_DRAWER, OPEN_LOW_TOP_DRAWER],
            UNGRASP_BOWL:                   [LIFT_BOWL, ROTATED_LOW_LEFT_BOWL, ROTATED_LOW_RIGHT_BOWL],
            UNGRASP_KETCHUP:                [LIFT_KETCHUP, ROTATED_LOW_LEFT_KETCHUP, ROTATED_LOW_RIGHT_KETCHUP],
            UNGRASP_CREAM_CHEESE:           [],
            
            ROTATED_LOW_LEFT_KETCHUP:       [PUT_KETCHUP_IN_TOP_DRAWER, PUT_KETCHUP_ON_BOWL, PUT_KETCHUP_ON_CABINET, ROTATED_LOW_RIGHT_KETCHUP],
            ROTATED_LOW_RIGHT_KETCHUP:      [PUT_KETCHUP_IN_TOP_DRAWER, PUT_KETCHUP_ON_BOWL, PUT_KETCHUP_ON_CABINET, ROTATED_LOW_LEFT_KETCHUP],
            ROTATED_LOW_LEFT_BOWL:          [PUT_BOWL_ON_CABINET, PUT_BOWL_ON_PLATE, ROTATED_LOW_RIGHT_BOWL],
            ROTATED_LOW_RIGHT_BOWL:         [PUT_BOWL_ON_CABINET, PUT_BOWL_ON_PLATE, ROTATED_LOW_LEFT_BOWL],

            ROTATED_HIGH_LEFT_BOWL:         [GRASP_KETCHUP, GRASP_TOP_DRAWER],
            ROTATED_HIGH_RIGHT_BOWL:        [GRASP_KETCHUP, GRASP_TOP_DRAWER],
            ROTATED_HIGH_LEFT_KETCHUP:      [GRASP_BOWL, GRASP_TOP_DRAWER],
            ROTATED_HIGH_RIGHT_KETCHUP:     [GRASP_BOWL, GRASP_TOP_DRAWER],

            GRASP_TOP_DRAWER:               [GRASP_CREAM_CHEESE, GRASP_KETCHUP, GRASP_BOWL],
            UNGRASP_TOP_DRAWER:             [OPEN_LOW_TOP_DRAWER, CLOSE_LOW_TOP_DRAWER],
            
            CLOSE_STATE_TOP_DRAWER:         [],
            OPEN_STATE_TOP_DRAWER:          [],
            CLOSE_LOW_TOP_DRAWER:           [OPEN_LOW_TOP_DRAWER, GRASP_KETCHUP, GRASP_BOWL],
            OPEN_LOW_TOP_DRAWER:            [CLOSE_LOW_TOP_DRAWER, GRASP_KETCHUP, GRASP_BOWL],

            CLOSE_HIGH_TOP_DRAWER:          [GRASP_CREAM_CHEESE, GRASP_KETCHUP, GRASP_BOWL],
            OPEN_HIGH_TOP_DRAWER:           [GRASP_CREAM_CHEESE, GRASP_KETCHUP, GRASP_BOWL],

            LIFT_BOWL:                      [UNGRASP_BOWL],
            LIFT_KETCHUP:                   [UNGRASP_KETCHUP],

            PLACE_KETCHUP_OVER_PLATE:       [PUT_KETCHUP_IN_TOP_DRAWER, PUT_KETCHUP_ON_BOWL, PUT_KETCHUP_ON_CABINET, GRASP_BOWL, GRASP_TOP_DRAWER],
            PLACE_KETCHUP_OVER_BOWL:        [PUT_KETCHUP_IN_TOP_DRAWER, PUT_KETCHUP_ON_PLATE, PUT_KETCHUP_ON_CABINET, GRASP_BOWL, GRASP_TOP_DRAWER],
            PLACE_KETCHUP_OVER_TOP_DRAWER:  [PUT_KETCHUP_ON_BOWL, PUT_KETCHUP_ON_PLATE, PUT_KETCHUP_ON_CABINET, GRASP_BOWL, GRASP_TOP_DRAWER],
            PLACE_KETCHUP_OVER_CABINET:     [PUT_KETCHUP_ON_BOWL, PUT_KETCHUP_ON_PLATE, PUT_KETCHUP_ON_CABINET, GRASP_BOWL, GRASP_TOP_DRAWER],
            PLACE_BOWL_OVER_PLATE:          [PUT_BOWL_ON_CABINET, PUT_BOWL_IN_TOP_DRAWER, GRASP_KETCHUP, GRASP_TOP_DRAWER],
            PLACE_BOWL_OVER_TOP_DRAWER:     [PUT_BOWL_ON_CABINET, PUT_BOWL_ON_PLATE, GRASP_KETCHUP, GRASP_TOP_DRAWER],
            PLACE_BOWL_OVER_CABINET:        [PUT_BOWL_IN_TOP_DRAWER, PUT_BOWL_ON_PLATE, GRASP_KETCHUP, GRASP_TOP_DRAWER],

            "place_cream_cheese_over_plate": [UNGRASP_CREAM_CHEESE],
            "place_cream_cheese_over_bowl": [UNGRASP_CREAM_CHEESE],
            "place_cream_cheese_over_top_drawer": [UNGRASP_CREAM_CHEESE],
            "place_cream_cheese_over_cabinet": [UNGRASP_CREAM_CHEESE],

        
            # HL task mappings
            PUT_BOWL_IN_TOP_DRAWER:         [GRASP_KETCHUP, CLOSE_LOW_TOP_DRAWER, OPEN_LOW_TOP_DRAWER,], 
            "put_bowl_on_top_drawer":       [GRASP_KETCHUP, CLOSE_LOW_TOP_DRAWER, OPEN_LOW_TOP_DRAWER,], 

            PUT_BOWL_ON_PLATE:              [GRASP_KETCHUP, CLOSE_LOW_TOP_DRAWER, OPEN_LOW_TOP_DRAWER,], 
            PUT_BOWL_ON_CABINET:            [GRASP_CREAM_CHEESE, GRASP_KETCHUP, CLOSE_LOW_TOP_DRAWER, OPEN_LOW_TOP_DRAWER,], 


            PUT_KETCHUP_IN_TOP_DRAWER:      [GRASP_CREAM_CHEESE, CLOSE_LOW_TOP_DRAWER, OPEN_LOW_TOP_DRAWER, GRASP_BOWL], 
            "put_ketchup_on_top_drawer":    [GRASP_CREAM_CHEESE, CLOSE_LOW_TOP_DRAWER, OPEN_LOW_TOP_DRAWER, GRASP_BOWL], 
            PUT_KETCHUP_ON_PLATE:           [GRASP_CREAM_CHEESE, CLOSE_LOW_TOP_DRAWER, OPEN_LOW_TOP_DRAWER, GRASP_BOWL], 
            PUT_KETCHUP_ON_BOWL:            [GRASP_CREAM_CHEESE, CLOSE_LOW_TOP_DRAWER, OPEN_LOW_TOP_DRAWER, GRASP_BOWL], 
            PUT_KETCHUP_ON_CABINET:         [GRASP_CREAM_CHEESE, CLOSE_LOW_TOP_DRAWER, OPEN_LOW_TOP_DRAWER, GRASP_BOWL],
            
            # Old experiment
            TURN_ON_STOVE:                  [GRASP_PAN, GRASP_MOKA_POT],
            "put_pan_on_stove_3":           [GRASP_MOKA_POT, TURN_ON_STOVE],
            "put_moka_pot_on_stove_3":      [GRASP_PAN, TURN_ON_STOVE],

            "put_yellow_white_mug_in_microwave_6": [GRASP_PORCELAIN_MUG, CLOSE_MICROWAVE],
            CLOSE_MICROWAVE:                [GRASP_PORCELAIN_MUG, GRASP_YELLOW_WHITE_MUG],

            "put_right_moka_pot_on_stove_8":[GRASP_LEFT_MOKA_POT],
            "put_left_moka_pot_on_stove_8": [GRASP_MOKA_POT],

            "put_alphabet_soup_in_basket_1":[GRASP_CREAM_CHEESE, GRASP_TOMATO_SAUCE, GRASP_KETCHUP],
            "put_cream_cheese_in_basket_1": [GRASP_ALPHABET_SOUP, GRASP_TOMATO_SAUCE, GRASP_KETCHUP],

            "put_alphabet_soup_in_basket_2": [GRASP_CREAM_CHEESE, GRASP_TOMATO_SAUCE, GRASP_KETCHUP, GRASP_MILK, GRASP_ORANGE_JUICE, GRASP_BUTTER],
            "put_tomato_sauce_in_basket_2": [GRASP_CREAM_CHEESE, GRASP_ALPHABET_SOUP, GRASP_KETCHUP, GRASP_MILK, GRASP_ORANGE_JUICE, GRASP_BUTTER],

            "put_cream_cheese_in_basket_2": [GRASP_ALPHABET_SOUP, GRASP_TOMATO_SAUCE, GRASP_KETCHUP, GRASP_MILK, GRASP_ORANGE_JUICE, GRASP_BUTTER],
            "put_butter_in_basket_2":       [GRASP_CREAM_CHEESE, GRASP_ALPHABET_SOUP, GRASP_TOMATO_SAUCE, GRASP_KETCHUP, GRASP_MILK, GRASP_ORANGE_JUICE],
        }

        self.task_to_predicate = {
            CLOSE_STATE_TOP_DRAWER:         ["close", TOP_DRAWER],
            OPEN_STATE_TOP_DRAWER:          ["open", TOP_DRAWER],

            ROTATED_LOW_RIGHT_KETCHUP:      ["rotatedrightlow", KETCHUP],
            ROTATED_LOW_LEFT_KETCHUP:       ["rotatedleftlow", KETCHUP],
            ROTATED_LOW_LEFT_BOWL:          ["rotatedleftlow", BOWL],
            ROTATED_LOW_RIGHT_BOWL:         ["rotatedrightlow", BOWL],

            ROTATED_HIGH_RIGHT_KETCHUP:     ["rotatedrighthigh", KETCHUP],
            ROTATED_HIGH_LEFT_KETCHUP:      ["rotatedlefthigh", KETCHUP],
            ROTATED_HIGH_LEFT_BOWL:         ["rotatedlefthigh", BOWL],
            ROTATED_HIGH_RIGHT_BOWL:        ["rotatedrighthigh", BOWL],


            CLOSE_LOW_TOP_DRAWER:           ["closelow", TOP_DRAWER],
            OPEN_LOW_TOP_DRAWER:            ["openlow", TOP_DRAWER],

            # HL tasks:
            CLOSE_HIGH_TOP_DRAWER:          ["closehigh", TOP_DRAWER],
            OPEN_HIGH_TOP_DRAWER:           ["openhigh", TOP_DRAWER],

            # Old tasks
            TURN_ON_STOVE:                  ["turnon", STOVE],
            CLOSE_MICROWAVE:                ["close", MICROWAVE],
            "put_pan_on_stove_3":           ["on", FRYING_PAN, STOVE_REGION],
            "put_moka_pot_on_stove_3":      ["on", MOKA_POT_1, STOVE_REGION],

            "put_right_moka_pot_on_stove_8": ["on", MOKA_POT_1, STOVE_REGION],
            "put_left_moka_pot_on_stove_8": ["on", MOKA_POT_2, STOVE_REGION],

            "put_alphabet_soup_in_basket_1": ["in", SOUP, BASKET],
            "put_cream_cheese_in_basket_1": ["in", CREAM_CHEESE, BASKET],

            "put_alphabet_soup_in_basket_2": ["in", SOUP, BASKET],
            "put_tomato_sauce_in_basket_2": ["in", SAUCE, BASKET],
            
            "put_cream_cheese_in_basket_2": ["in", CREAM_CHEESE, BASKET],
            "put_butter_in_basket_2": ["in", BUTTER, BASKET],

            # LL tasks:
            GRASP_BOWL:                     ["grasped", BOWL],
            GRASP_KETCHUP:                  ["grasped", KETCHUP],
            GRASP_TOP_DRAWER:               ["grasped", TOP_DRAWER],

            UNGRASP_BOWL:                   ["ungrasped", BOWL],
            UNGRASP_KETCHUP:                ["ungrasped", KETCHUP],
            UNGRASP_CREAM_CHEESE:           ["ungrasped", CREAM_CHEESE],
            UNGRASP_TOP_DRAWER:             ["ungrasped", TOP_DRAWER],
            UNGRASP_STATE_BOWL:             ["ungraspedstate", BOWL],
            UNGRASP_STATE_KETCHUP:          ["ungraspedstate", KETCHUP],


            LIFT_BOWL:                      ["lifted", BOWL],
            LIFT_KETCHUP:                   ["lifted", KETCHUP],

            PLACE_KETCHUP_OVER_PLATE:       ["over", KETCHUP, PLATE],
            PLACE_KETCHUP_OVER_BOWL:        ["over", KETCHUP, BOWL],
            PLACE_KETCHUP_OVER_TOP_DRAWER:  ["over", KETCHUP, TOP_DRAWER],
            PLACE_KETCHUP_OVER_CABINET:     ["over", KETCHUP, CABINET],

            PLACE_BOWL_OVER_PLATE:          ["over", BOWL, PLATE],
            PLACE_BOWL_OVER_TOP_DRAWER:     ["over", BOWL, TOP_DRAWER],
            PLACE_BOWL_OVER_CABINET:        ["over", BOWL, CABINET],

            PUT_BOWL_IN_TOP_DRAWER:         ["on", BOWL, TOP_DRAWER], 
            "put_bowl_on_top_drawer":       ["on", BOWL, TOP_DRAWER], 

            PUT_BOWL_ON_PLATE:              ["on", BOWL, PLATE], 
            PUT_BOWL_ON_CABINET:            ["on", BOWL, CABINET], 

            PUT_KETCHUP_IN_TOP_DRAWER:      ["on", KETCHUP, TOP_DRAWER], 
            "put_ketchup_on_top_drawer":    ["on", KETCHUP, TOP_DRAWER], 
            PUT_KETCHUP_ON_PLATE:           ["on", KETCHUP,  PLATE], 
            PUT_KETCHUP_ON_BOWL:            ["on", KETCHUP,  BOWL], 
            PUT_KETCHUP_ON_CABINET:         ["on", KETCHUP, CABINET], 


            GRASP_PAN:                      ["grasped", FRYING_PAN],
            GRASP_MOKA_POT:                 ["grasped", MOKA_POT_1],
            GRASP_LEFT_MOKA_POT:            ["grasped", MOKA_POT_2],
            GRASP_YELLOW_WHITE_MUG:         ["grasped", WHITE_YELLOW_MUG],
            GRASP_PORCELAIN_MUG:            ["grasped", PORCELAIN_MUG],
            GRASP_CREAM_CHEESE:             ["grasped", CREAM_CHEESE],
            GRASP_TOMATO_SAUCE:             ["grasped", SAUCE],
            GRASP_MILK:                     ["grasped", MILK],
            GRASP_ORANGE_JUICE:             ["grasped", ORANGE_JUICE],
            GRASP_BUTTER:                   ["grasped", BUTTER],
            GRASP_STOVE:                    ["grasped", STOVE],
            GRASP_ALPHABET_SOUP:            ["grasped", SOUP],
        }

    def set_seq_evaluation(self, do_sequential_evaluation):
        self.do_sequential_evaluation = do_sequential_evaluation

    def predicate_to_task(self, predicate):
        if predicate[0] == "grasped":
            if predicate[1] == BOWL:
                return GRASP_BOWL
            elif predicate[1] == SOUP:
                return GRASP_ALPHABET_SOUP
            elif predicate[1] == KETCHUP:
                return GRASP_KETCHUP
            elif predicate[1] == TOP_DRAWER:
                return GRASP_TOP_DRAWER
            elif predicate[1] == FRYING_PAN:
                return GRASP_PAN
            elif predicate[1] == MOKA_POT_1:
                return GRASP_MOKA_POT
            elif predicate[1] == MOKA_POT_2:
                return GRASP_LEFT_MOKA_POT
            elif predicate[1] == WHITE_YELLOW_MUG:
                return GRASP_YELLOW_WHITE_MUG
            elif predicate[1] == PORCELAIN_MUG:
                return GRASP_PORCELAIN_MUG
            elif predicate[1] == CREAM_CHEESE:
                return GRASP_CREAM_CHEESE
            elif predicate[1] == SAUCE:
                return GRASP_TOMATO_SAUCE
            elif predicate[1] == MILK:
                return GRASP_MILK
            elif predicate[1] == ORANGE_JUICE:
                return GRASP_ORANGE_JUICE
            elif predicate[1] == BUTTER:
                return GRASP_BUTTER
            elif predicate[1] == STOVE:
                return GRASP_STOVE
            else:
                raise Exception(f"Grasping unknown object: {predicate[1]}")
        elif predicate[0] == "rotatedleftlow":
            if predicate[1] == KETCHUP:
                return ROTATED_LOW_LEFT_KETCHUP
            elif predicate[1] == BOWL:
                return ROTATED_LOW_LEFT_BOWL
            else:
                raise Exception(f"Rotating left unknown object: {predicate[1]}")
        elif predicate[0] == "rotatedrightlow":
            if predicate[1] == KETCHUP:
                return ROTATED_LOW_RIGHT_KETCHUP
            elif predicate[1] == BOWL:
                return ROTATED_LOW_RIGHT_BOWL
            else:
                raise Exception(f"Rotating right unknown object: {predicate[1]}")
        elif predicate[0] == "rotatedlefthigh":
            if predicate[1] == KETCHUP:
                return ROTATED_HIGH_LEFT_KETCHUP
            elif predicate[1] == BOWL:
                return ROTATED_HIGH_LEFT_BOWL
            else:
                raise Exception(f"Rotating left unknown object: {predicate[1]}")
        elif predicate[0] == "rotatedrighthigh":
            if predicate[1] == KETCHUP:
                return ROTATED_HIGH_RIGHT_KETCHUP
            elif predicate[1] == BOWL:
                return ROTATED_HIGH_RIGHT_BOWL
            else:
                raise Exception(f"Rotating right unknown object: {predicate[1]}")
        elif predicate[0] == "ungrasped":
            if predicate[1] == BOWL:
                return UNGRASP_BOWL
            elif predicate[1] == KETCHUP:
                return UNGRASP_KETCHUP
            elif predicate[1] == CREAM_CHEESE:
                return UNGRASP_CREAM_CHEESE
            elif predicate[1] == TOP_DRAWER:
                return UNGRASP_TOP_DRAWER
            else:
                raise Exception(f"Ungrasping unknown object: {predicate[1]}")
        elif predicate[0] == "lifted":
            if predicate[1] == BOWL:
                return LIFT_BOWL
            elif predicate[1] == KETCHUP:
                return LIFT_KETCHUP
            elif predicate[1] == CREAM_CHEESE:
                return LIFT_CREAM_CHEESE
            else:
                raise Exception(f"Lifting unknown object: {predicate[1]}")
        elif predicate[0] == "open":
            if predicate[1] == TOP_DRAWER:
                return OPEN_STATE_TOP_DRAWER
            else:
                raise Exception(f"Unknown object to open: {predicate[1]}")
        elif predicate[0] == "openlow":
            if predicate[1] == TOP_DRAWER:
                return OPEN_LOW_TOP_DRAWER
            else:
                raise Exception(f"Unknown object to open: {predicate[1]}")
        elif predicate[0] == "openhigh":
            if predicate[1] == TOP_DRAWER:
                return OPEN_HIGH_TOP_DRAWER
            else:
                raise Exception(f"Unknown object to open: {predicate[1]}")
        elif predicate[0] == "close":
            if predicate[1] == TOP_DRAWER:
                return CLOSE_STATE_TOP_DRAWER
            elif predicate[1] == MICROWAVE:
                return CLOSE_MICROWAVE
            else:
                raise Exception(f"Unknown object to close: {predicate[1]}")
        elif predicate[0] == "closelow":
            if predicate[1] == TOP_DRAWER:
                return CLOSE_LOW_TOP_DRAWER
            else:
                raise Exception(f"Unknown object to close: {predicate[1]}")
        elif predicate[0] == "closehigh":
            if predicate[1] == TOP_DRAWER:
                return CLOSE_HIGH_TOP_DRAWER
            else:
                raise Exception(f"Unknown object to close: {predicate[1]}")
        elif predicate[0] == "over":
            if predicate[1] == KETCHUP:
                if predicate[2] == PLATE:
                    return PLACE_KETCHUP_OVER_PLATE
                elif predicate[2] == BOWL:
                    return PLACE_KETCHUP_OVER_BOWL
                elif predicate[2] == TOP_DRAWER:
                    return PLACE_KETCHUP_OVER_TOP_DRAWER
                elif predicate[2] == CABINET:
                    return PLACE_KETCHUP_OVER_CABINET
                else:
                    raise Exception(f"Placing object: {predicate[1]} over unknown location {predicate[2]}")
            elif predicate[1] == BOWL:
                if predicate[2] == PLATE:
                    return PLACE_BOWL_OVER_PLATE
                elif predicate[2] == TOP_DRAWER:
                    return PLACE_BOWL_OVER_TOP_DRAWER
                elif predicate[2] == CABINET:
                    return PLACE_BOWL_OVER_CABINET
                else:
                    raise Exception(f"Placing object: {predicate[1]} over unknown location {predicate[2]}")
            elif predicate[1] == CREAM_CHEESE:
                if predicate[2] == PLATE:
                    return "place_cream_cheese_over_plate"
                elif predicate[2] == BOWL:
                    return "place_cream_cheese_over_bowl"
                elif predicate[2] == TOP_DRAWER:
                    return "place_cream_cheese_over_top_drawer"
                elif predicate[2] == CABINET:
                    return "place_cream_cheese_over_cabinet"
                else:
                    raise Exception(f"Placing object: {predicate[1]} over unknown location {predicate[2]}")
            else:
                raise Exception(f"Placing unknown object: {predicate[1]}")
        elif predicate[0] == "on":
            if predicate[1] == KETCHUP:
                if predicate[2] == PLATE:
                    return PUT_KETCHUP_ON_PLATE
                elif predicate[2] == BOWL:
                    return PUT_KETCHUP_ON_BOWL
                elif predicate[2] == TOP_DRAWER:
                    return PUT_KETCHUP_IN_TOP_DRAWER
                elif predicate[2] == CABINET:
                    return PUT_KETCHUP_ON_CABINET
                else:
                    raise Exception(f"Placing object: {predicate[1]} over unknown location {predicate[2]}")
            elif predicate[1] == FRYING_PAN:
                if predicate[2] == STOVE_REGION:
                    return "put_pan_on_stove_3"
            elif predicate[1] == BOWL:
                if predicate[2] == PLATE:
                    return PUT_BOWL_ON_PLATE
                elif predicate[2] == TOP_DRAWER:
                    return PUT_BOWL_IN_TOP_DRAWER
                elif predicate[2] == CABINET:
                    return PUT_BOWL_ON_CABINET
                else:
                    raise Exception(f"Placing object: {predicate[1]} over unknown location {predicate[2]}")
            elif predicate[1] == MOKA_POT_1:
                if predicate[2] == STOVE_REGION:
                    return "put_moka_pot_on_stove_3"
            elif predicate[1] == MOKA_POT_2:
                if predicate[2] == STOVE_REGION:
                    return "put_left_moka_pot_on_stove"
            else:
                raise Exception(f"Placing unknown object: {predicate[1]}")
        elif predicate[0] == "in":
            if predicate[1] == CREAM_CHEESE:
                if predicate[2] == BASKET:
                    return "put_cream_cheese_in_basket_2"
            elif predicate[1] == BUTTER:
                if predicate[2] == BASKET:
                    return "put_butter_in_basket_2"
            elif predicate[1] == SOUP:
                if predicate[2] == BASKET:
                    return "put_alphabet_soup_in_basket_2"
            elif predicate[1] == SAUCE:
                if predicate[2] == BASKET:
                    return "put_tomato_sauce_in_basket_2"
            elif predicate[1] == WHITE_YELLOW_MUG:
                if predicate[2] == "microwave_1_heating_region":
                    return "put_yellow_white_mug_in_microwave_6"
        elif predicate[0] == "turnon":
            if predicate[1] == STOVE:

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
            info["hard_eval_passed"], info["inadmissible_task"] = self._pass_hard_eval_validation()
            if done:
                self.update_init_obj_poses()
            
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
            
            prob = self.parsed_problem["goal_state"][0][0]
            if 'rotate' in prob and all_subgoals_done:
                return obs, reward, all_subgoals_done, info
            else:
                return obs, reward, done and all_subgoals_done, info


    def update_init_obj_poses(self):
        for obj in self.object_states_dict.values():
            if isinstance(obj, ObjectState) or isinstance(obj, SiteObjectState):
                obj.set_init_pos()
                    
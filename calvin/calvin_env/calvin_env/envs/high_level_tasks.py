from functools import partial
from typing import Dict, Set

import numpy as np
from omegaconf import ListConfig
from scipy.spatial.transform import Rotation as R

REL_EPS = 0.03

class HighLevelTasks:
    def __init__(self, tasks):
        """
        A task is defined as a specific change between the start_info and end_info dictionaries.
        Use config file in conf/tasks/ to define tasks using the base task functions defined in this class
        """
        # register task functions from config file
        self.tasks = {name: partial(getattr(self, args[0]), *args[1:]) for name, args in dict(tasks).items()}
        # dictionary mapping from task name to task id
        self.task_to_id = {name: i for i, name in enumerate(self.tasks.keys())}
        # dictionary mapping from task id to task name
        self.id_to_task = {i: name for i, name in enumerate(self.tasks.keys())}

    def get_task_info(self, start_info: Dict, end_info: Dict) -> Set:
        """
        start_info: dict with scene info and robot info
        end_info: dict with scene info and robot info
        returns set with achieved tasks
        """
        # call functions that are registered in self.tasks
        return {
            task_name
            for task_name, function in self.tasks.items()
            if function(start_info=start_info, end_info=end_info)
        }

    def get_task_info_for_set(self, start_info: Dict, end_info: Dict, task_filter: Set) -> Set:
        """
        start_info: dict with scene info and robot info
        end_info: dict with scene info and robot info
        task_filter: set with task names to check
        returns set with achieved tasks
        """
        # call functions that are registered in self.tasks
        return {
            task_name
            for task_name, function in self.tasks.items()
            if task_name in task_filter and function(start_info=start_info, end_info=end_info)
        }

    @property
    def num_tasks(self):
        return len(self.tasks)
    
    # =========================== HIGH LEVEL ===================================
    @staticmethod
    def toggle_both_lights(start_state, end_state, start_info, end_info):
        light_1_name = "lightbulb"
        light_2_name = "led"
        
        light_1_start_state = start_info["scene_info"]["lights"][light_1_name]["logical_state"]
        light_2_start_state = start_info["scene_info"]["lights"][light_2_name]["logical_state"]

        light_1_end_state = end_info["scene_info"]["lights"][light_1_name]["logical_state"]
        light_2_end_state = end_info["scene_info"]["lights"][light_2_name]["logical_state"]

        return (
            light_1_start_state == start_state and light_2_start_state == start_state
            and light_1_end_state == end_state and light_2_end_state == end_state
        )

    @staticmethod
    def place_object_into_closed_drawer(obj_name, dest_body, dest_link, start_info=None, end_info=None):
        # push object but first open the drawer
        return HighLevelTasks.place_object(obj_name, dest_body, dest_link, start_info=start_info, end_info=end_info) and \
            HighLevelTasks.move_door_rel('base__drawer', 0.12, start_info, end_info)

    @staticmethod
    def place_object(obj_name, dest_body, dest_link, start_info=None, end_info=None):
        """
        (on 'dest_link' if provided).
        The robot may not touch the object after placing.
        """

        robot_uid = start_info["robot_info"]["uid"]
        robot_contacts_end = set(c[2] for c in end_info["robot_info"]["contacts"])
        robot_contacts_end = set(c[2] for c in end_info["robot_info"]["contacts"])

        # Make sure robot isnt touching the object at the end
        _obj_uid = [v["uid"] for k, v in start_info["scene_info"]["movable_objects"].items() if k == obj_name]

        if _obj_uid[0] in robot_contacts_end:
            return False

        dest_uid = end_info["scene_info"]["fixed_objects"][dest_body]["uid"]
        dest_link_id = end_info["scene_info"]["fixed_objects"][dest_body]["links"][dest_link]

        object_contacts_start = set((c[2], c[4]) for c in start_info["scene_info"]["movable_objects"][obj_name]["contacts"])
        object_contacts_end = set((c[2], c[4]) for c in end_info["scene_info"]["movable_objects"][obj_name]["contacts"])
        
        return (
            len(object_contacts_start) == 1 and 
            len(object_contacts_end) == 1 and
            (dest_uid, dest_link_id) in object_contacts_end and
            (dest_uid, dest_link_id) not in object_contacts_start
        )
    
        # =========================== MEDIUM LEVEL ===================================

    @staticmethod
    def push_object_into(obj_name, src_body, src_link, dest_body, dest_link, start_info=None, end_info=None):
        """
        obj_name is either a list of object names or a string
        Returns True if the object / any of the objects changes contact from src_body to dest_body.
        The robot may neither touch the object at start nor end.
        """
        if isinstance(obj_name, (list, ListConfig)):
            return any(
                HighLevelTasks.push_object_into(ob, src_body, src_link, dest_body, dest_link, start_info, end_info)
                for ob in obj_name
            )
        
        robot_uid = start_info["robot_info"]["uid"]

        src_uid = start_info["scene_info"]["fixed_objects"][src_body]["uid"]
        src_link_id = start_info["scene_info"]["fixed_objects"][src_body]["links"][src_link]
        dest_uid = end_info["scene_info"]["fixed_objects"][dest_body]["uid"]
        dest_link_id = end_info["scene_info"]["fixed_objects"][dest_body]["links"][dest_link]

        start_contacts = set((c[2], c[4]) for c in start_info["scene_info"]["movable_objects"][obj_name]["contacts"])
        end_contacts = set((c[2], c[4]) for c in end_info["scene_info"]["movable_objects"][obj_name]["contacts"])
        return (
            robot_uid not in start_contacts | end_contacts
            and len(start_contacts) == 1
            and (src_uid, src_link_id) in start_contacts
            and (dest_uid, dest_link_id) in end_contacts
        )
    
    @staticmethod
    def move_door_rel(joint_name, threshold, start_info, end_info):
        """
        Returns True if the joint specified by 'obj_name' and 'joint_name' (e.g. a door or drawer)
        is moved from at least 'start_threshold' to 'end_threshold'.
        """
        start_joint_state = start_info["scene_info"]["doors"][joint_name]["current_state"]
        end_joint_state = end_info["scene_info"]["doors"][joint_name]["current_state"]

        return (
            0 < threshold < end_joint_state - start_joint_state or 0 > threshold > end_joint_state - start_joint_state
        )

    @staticmethod
    def stack_objects(max_vel=1, start_info=None, end_info=None):

        # TODO: ASSERT THE OBJECT IS GRASPED ALREADY?
        obj_uids = set(obj["uid"] for obj in start_info["scene_info"]["movable_objects"].values())

        for obj_name in start_info["scene_info"]["movable_objects"]:
            obj_start_info = start_info["scene_info"]["movable_objects"][obj_name]
            obj_end_info = end_info["scene_info"]["movable_objects"][obj_name]
            obj_start_contacts = set(c[2] for c in obj_start_info["contacts"])
            obj_end_contacts = set(c[2] for c in obj_end_info["contacts"])

            if (
                not len(obj_uids & obj_start_contacts)
                and len(obj_uids & obj_end_contacts)
                and not len(obj_end_contacts - obj_uids)
            ):
                # object velocity may not exceed max_vel for successful stack
                if np.all(np.abs(obj_end_info["current_lin_vel"]) < max_vel) and np.all(
                    np.abs(obj_end_info["current_ang_vel"]) < max_vel
                ):
                    return True
        return False

    @staticmethod
    def unstack_objects(max_vel=1, start_info=None, end_info=None):
        obj_uids = set(obj["uid"] for obj in start_info["scene_info"]["movable_objects"].values())

        for obj_name in start_info["scene_info"]["movable_objects"]:
            obj_start_info = start_info["scene_info"]["movable_objects"][obj_name]
            obj_end_info = end_info["scene_info"]["movable_objects"][obj_name]
            obj_start_contacts = set(c[2] for c in obj_start_info["contacts"])
            obj_end_contacts = set(c[2] for c in obj_end_info["contacts"])

            if (
                len(obj_uids & obj_start_contacts)
                and not len(obj_start_contacts - obj_uids)
                and not len(obj_uids & obj_end_contacts)
            ):
                # object velocity may not exceed max_vel for successful stack
                if np.all(np.abs(obj_start_info["current_lin_vel"]) < max_vel) and np.all(
                    np.abs(obj_start_info["current_ang_vel"]) < max_vel
                ):
                    return True
        return False

    

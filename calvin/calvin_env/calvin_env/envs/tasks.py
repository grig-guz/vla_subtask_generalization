from functools import partial
from typing import Dict, Set

import numpy as np
from omegaconf import ListConfig
from scipy.spatial.transform import Rotation as R

MAX_VEL = 1

class Tasks:
    
    def __init__(self, tasks, admissible_constraints, tasks_to_ignore):
        """
        A task is defined as a specific change between the start_info and end_info dictionaries.
        Use config file in conf/tasks/ to define tasks using the base task functions defined in this class
        """

        # register task functions from config file
        self.tasks = {name: partial(getattr(self, args[0]), *args[1:]) for name, args in dict(tasks).items()}
        self.tasks_to_ignore = {name: partial(getattr(self, args[0]), *args[1:]) for name, args in dict(tasks_to_ignore).items()}
        # dictionary mapping from task name to task ida
        self.task_to_id = {name: i for i, name in enumerate(self.tasks.keys())}
        # dictionary mapping from task id to task name
        self.id_to_task = {i: name for i, name in enumerate(self.tasks.keys())}
        self.admissible_constraints = {name: set(args) for name, args in dict(admissible_constraints).items()}

    def get_task_info(self, start_info: Dict, end_info: Dict) -> Set:
        """
        start_info: dict with scene info and robot info
        end_info: dict with scene info and robot info
        returns set with achieved tasks
        """
        # call functions that are registered in self.tasks
        completed_tasks = {
            task_name
            for task_name, function in self.tasks.items()
            if function(start_info=start_info, end_info=end_info)
        }

        ignored_tasks_completed = {
            task_name
            for task_name, function in self.tasks_to_ignore.items()
            if function(start_info=start_info, end_info=end_info)
        }

        if len(ignored_tasks_completed) > 0:
            return set()
        else:
            return completed_tasks

    def get_task_info_for_set(self, start_info: Dict, end_info: Dict, task_filter: Set) -> Set:
        """
        start_info: dict with scene info and robot info
        end_info: dict with scene info and robot info
        task_filter: set with task names to check
        returns set with achieved tasks
        """
        # call functions that are registered in self.tasks
        completed_tasks = {
            task_name
            for task_name, function in self.tasks.items()
            if task_name in task_filter and function(start_info=start_info, end_info=end_info)
        }

        ignored_tasks_completed = {
            task_name
            for task_name, function in self.tasks_to_ignore.items()
            if function(start_info=start_info, end_info=end_info)
        }
        if len(ignored_tasks_completed) > 0:
            return set()
        else:
            return completed_tasks

    def get_task_info_with_criteria(self, start_info: Dict, end_info: Dict, task: str) -> Set:
        """
        start_info: dict with scene info and robot info
        end_info: dict with scene info and robot info
        task: which task to check
        returns two boolean values:
            done: true if the given task is completed
            terminate: true if some constraint is violated    
        """

        task_function = self.tasks[task]
        done = task_function(start_info=start_info, end_info=end_info)
        if done:
            return done, False

        terminate = False

        to_log = True
        other_tasks_completed = self.get_task_info(start_info=start_info, end_info=end_info)
        if len(other_tasks_completed - self.admissible_constraints[task]) > 0:
            terminate = True
            if to_log:
                print(f"Terminating {task}, found {other_tasks_completed}")
        elif len(other_tasks_completed) > 0:
            if to_log:
                print(f"For {task}, found {other_tasks_completed}")

        return done, terminate


    @property
    def num_tasks(self):
        return len(self.tasks)

    @staticmethod
    def rotate_object(
        target_obj_name, z_degrees, x_y_threshold=30, z_threshold=180, movement_threshold=0.1, start_info=None, end_info=None
    ):
        """
        Preconditions:
            - The robot is not in contact with anything
            - All objects are stationary and in contact with something.
        Postconditions:
            - The object was rotated by z_degrees while not being moved by movement_threshold
            - All objects are stationary and in contact with something.
            - The robot is not in contact with anything

        Returns True if the object with obj_name was rotated more than z_degrees degrees around the z-axis while not
        being rotated more than x_y_threshold degrees around the x or y axis.
        z_degrees is negative for clockwise rotations and positive for counter-clockwise rotations.
        """

        if not stationary_objects_no_robot_contact(start_info=start_info, end_info=end_info):
            return False


        target_obj_start_info = start_info["scene_info"]["movable_objects"][target_obj_name]
        target_obj_end_info = end_info["scene_info"]["movable_objects"][target_obj_name]

        start_pos = np.array(target_obj_start_info["current_pos"])
        end_pos = np.array(target_obj_end_info["current_pos"])

        pos_diff = end_pos - start_pos
        if np.linalg.norm(pos_diff) > movement_threshold:
            return False

        start_orn = R.from_quat(target_obj_start_info["current_orn"])
        end_orn = R.from_quat(target_obj_end_info["current_orn"])
        rotation = end_orn * start_orn.inv()
        x, y, z = rotation.as_euler("xyz", degrees=True)

        if z_degrees > 0:
            return z_degrees < z < z_threshold and abs(x) < x_y_threshold and abs(y) < x_y_threshold
        else:
            return z_degrees > z > -z_threshold and abs(x) < x_y_threshold and abs(y) < x_y_threshold

    @staticmethod
    def push_object(obj_name, x_direction, y_direction, start_info, end_info):
        """
        Preconditions:
            - The obj_name is in contact with some surface
            - All objects are stationary and in contact with something.
            - The robot is not in contact with anything
        Postconditions:
            - The obj_name was moved in a given direction while staying on the same surface
                (TODO the object can be lifted)
            - All objects are stationary and in contact with something.
            - The robot is not in contact with anything

        Returns True if the object with 'obj_name' was moved more than 'x_direction' meters in x direction
        (or 'y_direction' meters in y direction analogously).
        Note that currently x and y pushes are mutually exclusive, meaning that one of the arguments has to be 0.
        The sign matters, e.g. pushing an object to the right when facing the table coincides with a movement in
        positive x-direction.
        """
        assert x_direction * y_direction == 0 and x_direction + y_direction != 0

        if not stationary_objects_no_robot_contact(start_info=start_info, end_info=end_info):
            return False


        robot_uid = start_info["robot_info"]["uid"]

        obj_start_info = start_info["scene_info"]["movable_objects"][obj_name]
        obj_end_info = end_info["scene_info"]["movable_objects"][obj_name]

        # contacts excluding robot
        start_contacts = set((c[2], c[4]) for c in obj_start_info["contacts"] if c[2] != robot_uid)
        end_contacts = set((c[2], c[4]) for c in obj_end_info["contacts"] if c[2] != robot_uid)

        # computing set difference to check if object had surface contact (excluding robot) at both times
        surface_contact = len(start_contacts) > 0 and len(end_contacts) > 0 and start_contacts <= end_contacts
        if not surface_contact:
            return False

        start_pos = np.array(obj_start_info["current_pos"])
        end_pos = np.array(obj_end_info["current_pos"])
        pos_diff = end_pos - start_pos

        if x_direction > 0:
            return pos_diff[0] > x_direction
        elif x_direction < 0:
            return pos_diff[0] < x_direction

        if y_direction > 0:
            return pos_diff[1] > y_direction
        elif y_direction < 0:
            return pos_diff[1] < y_direction


    @staticmethod
    def lift_object(obj_name, z_direction, surface_body=None, surface_link=None, start_info=None, end_info=None):
        """
        Preconditions:
            - The obj_name is on the surface surface_link
            - All objects are stationary and in contact with something.
            - The robot is not in contact with anything
        Postconditions:
            - The robot is in contact with obj_name
            - obj_name is lifted up by z_direction
        """
        assert z_direction > 0

        if not stationary_objects_no_robot_contact_for_info(info=start_info):
            return False

        robot_uid = start_info["robot_info"]["uid"]
        obj_start_info = start_info["scene_info"]["movable_objects"][obj_name]
        obj_end_info = end_info["scene_info"]["movable_objects"][obj_name]

        start_orn = R.from_quat(obj_start_info["current_orn"])
        end_orn = R.from_quat(obj_end_info["current_orn"])
        rotation = end_orn * start_orn.inv()
        x, y, z = rotation.as_euler("xyz", degrees=True)
        if abs(x) > 30 or abs(y) > 30 or abs(z) > 30:
            return False


        start_contacts = set(c[2] for c in obj_start_info["contacts"])
        end_contacts = set(c[2] for c in obj_end_info["contacts"])


        if surface_body and surface_link is None:
            surface_uid = start_info["scene_info"]["fixed_objects"][surface_body]["uid"]
            surface_criterion = surface_uid in start_contacts
        elif surface_body and surface_link:
            surface_uid = start_info["scene_info"]["fixed_objects"][surface_body]["uid"]
            surface_link_id = start_info["scene_info"]["fixed_objects"][surface_body]["links"][surface_link]
            start_contacts_links = set((c[2], c[4]) for c in obj_start_info["contacts"])
            surface_criterion = (surface_uid, surface_link_id) in start_contacts_links

        start_pos = np.array(obj_start_info["current_pos"])
        end_pos = np.array(obj_end_info["current_pos"])
        pos_diff = end_pos - start_pos
        z_diff = pos_diff[2]

        return (
            z_diff > z_direction
            # and robot_uid not in start_contacts
            and robot_uid in end_contacts
            and len(end_contacts) == 1
            and surface_criterion
        )

    @staticmethod
    def place_object_whole(obj_name, dest_body, dest_link=None, start_info=None, end_info=None):
        """
        Semantics:
        Preconditions: 
            The obj_name is not in contact with dest_link
            All objects are stationary and in contact with something.
            The robot is not in contact with anything.
        Post-conditions: 
            The obj_name is in contact with dest_link
            All objects are stationary and in contact with something.
            The robot is not in contact with anything.
        """
        robot_uid = start_info["robot_info"]["uid"]

        if not stationary_objects_no_robot_contact(start_info=start_info, end_info=end_info):
            return False

        dest_uid = end_info["scene_info"]["fixed_objects"][dest_body]["uid"]
        object_contacts_start = set(c[2] for c in start_info["scene_info"]["movable_objects"][obj_name]["contacts"])
        object_contacts_end = set(c[2] for c in end_info["scene_info"]["movable_objects"][obj_name]["contacts"])

        if dest_link is None:
            object_contacts_end = set(c[2] for c in end_info["scene_info"]["movable_objects"][obj_name]["contacts"])
            return (
                robot_uid in object_contacts_start
                and len(object_contacts_start) == 1
                and dest_uid in object_contacts_end
            )
        else:
            dest_link_id = end_info["scene_info"]["fixed_objects"][dest_body]["links"][dest_link]
            end_contacts_links = set(
                (c[2], c[4]) for c in end_info["scene_info"]["movable_objects"][obj_name]["contacts"]
            )
            start_contacts_links = set(
                (c[2], c[4]) for c in start_info["scene_info"]["movable_objects"][obj_name]["contacts"]
            )
            table_object_id = 5
            return (
                table_object_id in object_contacts_start
                and (dest_uid, dest_link_id) in end_contacts_links 
                and not (dest_uid, dest_link_id) in start_contacts_links 
            )


    @staticmethod
    def move_door_rel(joint_name, threshold, start_info, end_info):
        """
        Semantics:
        Preconditions: 
            The robot is not in contact with anything.
            All objects are stationary and in contact with something.
        Post-conditions: 
            The door was moved by a specified amount.
            All objects are stationary and in contact with something.
            The robot is not in contact with anything.

        Returns True if the joint specified by 'obj_name' and 'joint_name' (e.g. a door or drawer)
        is moved from at least 'start_threshold' to 'end_threshold'.
        """

        if not stationary_objects_no_robot_contact(
                start_info=start_info, 
                end_info=end_info
            ):
            return False

        start_joint_state = start_info["scene_info"]["doors"][joint_name]["current_state"]
        end_joint_state = end_info["scene_info"]["doors"][joint_name]["current_state"]

        return (
            0 < threshold < end_joint_state - start_joint_state or 0 > threshold > end_joint_state - start_joint_state
        )



    @staticmethod
    def stack_objects_whole(start_info=None, end_info=None):
        """
        Semantics:
        Preconditions: 
            The robot is not in contact with anything.
            All objects are stationary and in contact with something.
        Post-conditions: 
            One of the objects is on top of another object.
            All objects are stationary and in contact with something.
            The robot is not in contact with anything.
        """

        if not stationary_objects_no_robot_contact(start_info=start_info, end_info=end_info):
            return False

        obj_uids = set(obj["uid"] for obj in start_info["scene_info"]["movable_objects"].values())


        for obj_name in start_info["scene_info"]["movable_objects"]:
            obj_start_info = start_info["scene_info"]["movable_objects"][obj_name]
            obj_end_info = end_info["scene_info"]["movable_objects"][obj_name]
            obj_start_contacts = set(c[2] for c in obj_start_info["contacts"])
            obj_end_contacts = set(c[2] for c in obj_end_info["contacts"])

            if (
                not len(obj_uids & obj_start_contacts) # obj_name IS NOT in contact with other objects initially 
                and len(obj_uids & obj_end_contacts) # obj_name IS in contact with other objects initially 
                and not len(obj_end_contacts - obj_uids) # obj_name is only in contact with one other object
            ):
                #("Stacking:", np.all(np.abs(obj_start_info["current_lin_vel"])), max_vel)
                # object velocity may not exceed max_vel for successful stack
                if np.all(np.abs(obj_end_info["current_lin_vel"]) < MAX_VEL) and np.all(
                    np.abs(obj_end_info["current_ang_vel"]) < MAX_VEL
                ):
                    return True
        return False
    
    @staticmethod
    def unstack_objects_whole(max_vel=1, start_info=None, end_info=None):
        obj_uids = set(obj["uid"] for obj in start_info["scene_info"]["movable_objects"].values())

        if not stationary_objects_no_robot_contact(start_info=start_info, end_info=end_info):
            return False

        for obj_name in start_info["scene_info"]["movable_objects"]:
            obj_start_info = start_info["scene_info"]["movable_objects"][obj_name]
            obj_end_info = end_info["scene_info"]["movable_objects"][obj_name]
            obj_start_contacts = set(c[2] for c in obj_start_info["contacts"])
            obj_end_contacts = set(c[2] for c in obj_end_info["contacts"])

            if (
                len(obj_uids & obj_start_contacts) # obj_name IS in contact with some other object initially
                and not len(obj_start_contacts - obj_uids) # obj_name IS ONLY in contact with 1 object initially
                and not len(obj_uids & obj_end_contacts) # obj_name IS NOT in contact with any objects in the end
            ):
                #print("Unstacking:", np.all(np.abs(obj_start_info["current_lin_vel"])), max_vel)
                # object velocity may not exceed max_vel for successful stack
                if np.all(np.abs(obj_start_info["current_lin_vel"]) < max_vel) and \
                    np.all(np.abs(obj_start_info["current_ang_vel"]) < max_vel):
                    return True
        return False

    @staticmethod
    def toggle_light(light_name, start_state, end_state, start_info, end_info):
        return (
            start_info["scene_info"]["lights"][light_name]["logical_state"] == start_state
            and end_info["scene_info"]["lights"][light_name]["logical_state"] == end_state
        )
    
def stationary_objects_no_robot_contact(start_info, end_info):
    return stationary_objects_no_robot_contact_for_info(info=start_info) and \
        stationary_objects_no_robot_contact_for_info(info=end_info)

def stationary_objects_no_robot_contact_for_info(info):
    return stationary_objects_for_info(info) and no_robot_contacts_for_info(info)

def stationary_objects_for_info(info):

    for obj in info["scene_info"]["movable_objects"].keys():
        obj_info = info["scene_info"]["movable_objects"][obj]
        obj_contacts = set(c[2] for c in obj_info["contacts"])


        if len(obj_contacts) == 0 or np.any(np.abs(obj_info["current_lin_vel"]) > MAX_VEL) or \
            np.any(np.abs(obj_info["current_ang_vel"]) > MAX_VEL):
            return False
        
    return True

def no_robot_contacts_for_info(info):
    robot_contacts = set(c[2] for c in info["robot_info"]["contacts"])
    if len(robot_contacts) > 0:
        return False
    return True

"""
    @staticmethod
    def place_object(obj_name_arg, dest_body, dest_link=None, start_info=None, end_info=None):
        
        #Returns True if the object that the robot has currently lifted is placed on the body 'dest_body'
        #(on 'dest_link' if provided).
        #The robot may not touch the object after placing.
        
        robot_uid = start_info["robot_info"]["uid"]

        robot_contacts_start = set(c[2] for c in start_info["robot_info"]["contacts"])
        robot_contacts_end = set(c[2] for c in end_info["robot_info"]["contacts"])
        if not len(robot_contacts_start) == 1:
            return False
        obj_uid = list(robot_contacts_start)[0]

        if obj_uid in robot_contacts_end:
            return False
        _obj_name = [k for k, v in start_info["scene_info"]["movable_objects"].items() if v["uid"] == obj_uid]
        if not len(_obj_name) == 1:
            return False
        obj_name = _obj_name[0]

        dest_uid = end_info["scene_info"]["fixed_objects"][dest_body]["uid"]

        object_contacts_start = set(c[2] for c in start_info["scene_info"]["movable_objects"][obj_name]["contacts"])
        if dest_link is None:
            object_contacts_end = set(c[2] for c in end_info["scene_info"]["movable_objects"][obj_name]["contacts"])
            return (
                robot_uid in object_contacts_start
                and len(object_contacts_start) == 1
                and dest_uid in object_contacts_end
            )
        else:
            dest_link_id = end_info["scene_info"]["fixed_objects"][dest_body]["links"][dest_link]
            end_contacts_links = set(
                (c[2], c[4]) for c in end_info["scene_info"]["movable_objects"][obj_name]["contacts"]
            )
            return (
                robot_uid in object_contacts_start
                and len(object_contacts_start) == 1
                and (dest_uid, dest_link_id) in end_contacts_links
            )


    @staticmethod
    def stack_objects(max_vel=1, start_info=None, end_info=None):
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

    @staticmethod
    def push_object_into(obj_name, src_body, src_link, dest_body, dest_link, start_info=None, end_info=None):
        
        #obj_name is either a list of object names or a string
        #Returns True if the object / any of the objects changes contact from src_body to dest_body.
        #The robot may neither touch the object at start nor end.
        
        if isinstance(obj_name, (list, ListConfig)):
            return any(
                Tasks.push_object_into(ob, src_body, src_link, dest_body, dest_link, start_info, end_info)
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

"""
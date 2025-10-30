from functools import partial
from typing import Dict, Set

import numpy as np
from omegaconf import ListConfig
from scipy.spatial.transform import Rotation as R

REL_EPS = 0.03
EPSILON_ROTATION_DEGREES = 45
EPSILON_SLIDER_DRAWER =  0.05
EPSILON_BLOCK_POS = 0.2
EPSILON_BLOCK_TOUCH = 0.05
EPSILON_GRASP_LIFT = 0.001


class LowLevelTasks:
    def __init__(self, tasks, admissible_constraints):
        """
        A task is defined as a specific change between the start_info and end_info dictionaries.
        Use config file in conf/tasks/ to define tasks using the base task functions defined in this class
        """

        # register task functions from config file
        self.tasks = {name: partial(getattr(self, args[0]), *args[1:]) for name, args in dict(tasks).items()}
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
        obj_names, z_degrees, x_y_threshold=30, z_threshold=180, movement_threshold=0.1, start_info=None, end_info=None
    ):
        """
        Preconditions:
            - The robot is grasping some object
        Postconditions:
            - The object was rotated by z_degrees
            - The robot is still grasping some object

        Returns True if the object with obj_name was rotated more than z_degrees degrees around the z-axis while not
        being rotated more than x_y_threshold degrees around the x or y axis.
        z_degrees is negative for clockwise rotations and positive for counter-clockwise rotations.
        """


        found_grasped = False
        for name in obj_names:
            if is_block_grasped(obj_name=name, state_info=start_info):
                obj_name = name
                found_grasped = True
                break
        
        if not found_grasped:
            return False
        
        obj_start_info = start_info["scene_info"]["movable_objects"][obj_name]
        obj_end_info = end_info["scene_info"]["movable_objects"][obj_name]
        #print(f"Found {obj_name} grasped in the beginning!")
        if not is_block_grasped(obj_name=obj_name, state_info=end_info):
            return False
        #print(f"{obj_name} is grasped in the end still!")


        start_orn = R.from_quat(obj_start_info["current_orn"])
        end_orn = R.from_quat(obj_end_info["current_orn"])
        rotation = end_orn * start_orn.inv()
        x, y, z = rotation.as_euler("xyz", degrees=True)

        #print("Rotated by ", z)
        start_pos = np.array(obj_start_info["current_pos"])
        end_pos = np.array(obj_end_info["current_pos"])
        pos_diff = end_pos - start_pos
        if np.linalg.norm(pos_diff) > movement_threshold:
            return False

        #end_contacts = set(c[2] for c in obj_end_info["contacts"])
        #robot_uid = {start_info["robot_info"]["uid"]}

        # object should be in contact with ground
        #if len(end_contacts - robot_uid) == 0:
        #    return False

        if z_degrees > 0:
            return z_degrees < z < z_threshold and abs(x) < x_y_threshold and abs(y) < x_y_threshold
        else:
            return z_degrees > z > -z_threshold and abs(x) < x_y_threshold and abs(y) < x_y_threshold

    @staticmethod
    def push_object(x_direction, y_direction, start_info, end_info):
        """
        Preconditions:
            - The robot in contact with some object
        Postconditions:
            - The object was moved in x_direction
            - The robot is not in contact with anything

        Returns True if the object with 'obj_name' was moved more than 'x_direction' meters in x direction
        (or 'y_direction' meters in y direction analogously).
        Note that currently x and y pushes are mutually exclusive, meaning that one of the arguments has to be 0.
        The sign matters, e.g. pushing an object to the right when facing the table coincides with a movement in
        positive x-direction.
        """
        assert x_direction * y_direction == 0 and x_direction + y_direction != 0

        robot_contacts_end = [c[2] for c in end_info["robot_info"]["contacts"]]

        if len(robot_contacts_end) > 0:
            return False


        # Find the block which is in contact with the gripper
        found_block = False
        robot_uid = start_info["robot_info"]["uid"]

        for block_name, block_info in start_info["scene_info"]["movable_objects"].items():
            for c in block_info['contacts']:
                if c[2] == robot_uid:
                    obj_name = block_name
                    found_block = True
                    break
        
        if not found_block:
            return False

        obj_start_info = start_info["scene_info"]["movable_objects"][obj_name]
        obj_end_info = end_info["scene_info"]["movable_objects"][obj_name]
        start_pos = np.array(obj_start_info["current_pos"])
        end_pos = np.array(obj_end_info["current_pos"])
        pos_diff = end_pos - start_pos

        # contacts excluding robot
        start_contacts = set((c[2], c[4]) for c in obj_start_info["contacts"] if c[2] != robot_uid)
        end_contacts = set((c[2], c[4]) for c in obj_end_info["contacts"] if c[2] != robot_uid)

        # computing set difference to check if object had surface contact (excluding robot) at both times
        surface_contact = len(start_contacts) > 0 and len(end_contacts) > 0 and start_contacts <= end_contacts
        if not surface_contact:
            return False

        if x_direction > 0:
            return pos_diff[0] > x_direction
        elif x_direction < 0:
            return pos_diff[0] < x_direction

        if y_direction > 0:
            return pos_diff[1] > y_direction
        elif y_direction < 0:
            return pos_diff[1] < y_direction

    @staticmethod
    def lift_grasped_object(surface_bodies=None, start_info=None, end_info=None):
        """
        Preconditions:
            - The robot is grasping some object
        Postconditions:
            - The robot is still touching the same object
            - The object was lifted by z_direction
            - The object was not moved in x-y direction by more than EPSILON_BLOCK_POS
            - The object was not rotated around z axis by more than EPSILON_ROTATION_DEGREES

        Returns True if the object with 'obj_name' was grasped by the robot and lifted more than 'z_direction' meters.
        """


        found_grasped = False
        for name in ['block_red', 'block_blue', 'block_pink']:
            if is_block_grasped(obj_name=name, state_info=start_info):
                obj_name = name
                found_grasped = True
                break
        
        if not found_grasped:
            #print("Lifting: not found grasped!")
            return False
        #print("LIFTING: FOUND GRASPED OBJECT ", obj_name)
        obj_start_info = start_info["scene_info"]["movable_objects"][obj_name]
        obj_end_info = end_info["scene_info"]["movable_objects"][obj_name]

        start_pos = np.array(obj_start_info["current_pos"])
        end_pos = np.array(obj_end_info["current_pos"])
        pos_diff = end_pos - start_pos
        z_diff = pos_diff[2]

        robot_uid = start_info["robot_info"]["uid"]
        end_contacts = set(c[2] for c in obj_end_info["contacts"])

        for surface_link in ['base_link', 'plank_link', 'drawer_link']:
            surface_uid = start_info["scene_info"]["fixed_objects"]['table']["uid"]
            surface_link_id = start_info["scene_info"]["fixed_objects"]['table']["links"][surface_link]
            start_contacts_links = set((c[2], c[4]) for c in obj_start_info["contacts"])
            end_contacts_links = set((c[2], c[4]) for c in obj_end_info["contacts"])
            surface_criterion = (surface_uid, surface_link_id) in start_contacts_links and (surface_uid, surface_link_id) not in end_contacts_links

            if not surface_criterion:
                if not block_lifting_epsilon(start_pos):
                    #print("Lifting: surface criterion not satisfied for block ", obj_name)
                    continue

            start_pos = np.array(obj_start_info["current_pos"])
            end_pos = np.array(obj_end_info["current_pos"])
            pos_diff = np.abs(end_pos[:2] - start_pos[:2]).sum()

            start_orn = R.from_quat(obj_start_info["current_orn"])
            end_orn = R.from_quat(obj_end_info["current_orn"])
            rotation = end_orn * start_orn.inv()
            _, _, z = rotation.as_euler("xyz", degrees=True)

            if pos_diff > EPSILON_BLOCK_POS:
                #print("LIFTING: OBJECT MOVED TOO MUCH ", obj_name)
                return False

            z_degrees = EPSILON_ROTATION_DEGREES
            if abs(z) > z_degrees:
                #print("LIFTING: OBJECT ROTATED TOO MUCH ", obj_name)
                return False

            z_direction = 0.03
            #print("LIFTING: z diff is ", z_diff)

            if (
                z_diff > z_direction
                and robot_uid in end_contacts
                and len(end_contacts) == 1
            ):
                return True
            
        return False


    @staticmethod
    def move_door_rel(joint_name, threshold, start_info, end_info):
        """
        Preconditions:
            - The robot is touching the door joint_name
        Postconditions:
            - The robot is still touching the door joint_name
            - The door was moved by threshold
        
        Returns True if the joint specified by 'obj_name' and 'joint_name' (e.g. a door or drawer)
        is moved from at least 'start_threshold' to 'end_threshold'.
        """

        
        robot_contacts_links_start = set(c[4] for c in start_info["robot_info"]["contacts"])
        robot_contacts_links_end = set(c[4] for c in end_info["robot_info"]["contacts"])
        
        if joint_name == 'base__drawer':
            drawer_link_id = start_info['scene_info']['fixed_objects']['table']['links']['drawer_link']
            if drawer_link_id not in robot_contacts_links_start or drawer_link_id not in robot_contacts_links_end:
                return False
            #print("MOVE_DRAWER: Passed the grasping criterion")

        elif joint_name == 'base__slide':
            # Slider is not grasped in beginning AND slider is grasped at the end
            if not (is_slider_grasped(state_info=start_info) and is_slider_grasped(state_info=end_info)):
                return False
            #print("MOVE_SLIDER: Passed grasping criterion")

        return was_drawer_slider_moved(joint_name, threshold, start_info, end_info)

    @staticmethod
    def place_grasped_block_over_block(object_name, start_info=None, end_info=None):
        """
        Preconditions:
            - The robot is grasping some block
        Postconditions:
            - The is grasping the same block
            - The block is above target block object_name
            - The target block is not inside a closed drawer
        """

        # Before grasping a block, check for whether the gripper is near the block.
        # TODO: Make sure the gripper isn't moving over a closed drawer with blocks inside.
        target_block_start = start_info["scene_info"]["movable_objects"][object_name]
        target_block_end = end_info["scene_info"]["movable_objects"][object_name]
        
        grasped_block = None
        for block_name in start_info["scene_info"]["movable_objects"]:
            if is_block_grasped(block_name, state_info=start_info) and is_block_grasped(block_name, state_info=end_info) and block_name != object_name:
                grasped_block = block_name
        
        if grasped_block == None:
            return False
        
        #both_contacts_empty = len(robot_contacts_start) == 0 and len(robot_contacts_end) == 0
        #only_1_contact = len(robot_contacts_start) == 1 and len(robot_contacts_end) == 1 and list(robot_contacts_start)[0] == list(robot_contacts_end)[0]
        # contacts should stay the same
        #if not (both_contacts_empty or only_1_contact):
        #    return False

        start_grasped_block_pos = np.array(start_info["scene_info"]["movable_objects"][grasped_block]['current_pos'])
        end_grasped_block_pos = np.array(end_info["scene_info"]["movable_objects"][grasped_block]['current_pos'])

        target_block_start_pos = np.array(target_block_start['current_pos'])
        # block in drawer and drawer is closed
        #print(f"PLACE_GRASPED_OVER_BLOCK: For block {object_name}, {block_in_drawer(target_block_start_pos)} and {is_drawer_closed(start_info['scene_info']['fixed_objects']['table']['links']['drawer_link'])}")
        if block_in_drawer(target_block_start_pos) and is_drawer_closed(start_info['scene_info']['doors']['base__drawer']['current_state']):
            return False
        
        target_block_end_pos = np.array(target_block_end['current_pos'])

        # Make sure initial and final target block pos is approx equal
        block_movement = np.sum(np.abs(target_block_end_pos - target_block_start_pos))
        disp_eps = 0.001
        
        if block_movement > disp_eps:
            return False

        # Find x-y coord boundaries for where the gripper should reach (relative to the target block?)
        start_obj_gripper_diff = np.sqrt(np.sum(np.square(target_block_start_pos[:2] - start_grasped_block_pos[:2])))
        end_obj_gripper_diff = np.sqrt(np.sum(np.square(target_block_end_pos[:2] - end_grasped_block_pos[:2])))

        # Make sure gripper wasn't too close to the block in the first place.
        if start_obj_gripper_diff < REL_EPS:
            return False
        
        # Make sure gripper isn't too far from the block.
        if end_obj_gripper_diff > REL_EPS:
            return False
        
        return True
    
    @staticmethod
    def place_grasped_block_over_surface(dest_link, start_info=None, end_info=None):
        """
        Preconditions:
            - The is grasping some block
        Postconditions:
            - The is grasping the same block
            - The block is above dest_link surface
        """

        robot_uid = {start_info["robot_info"]["uid"]}

        all_blocks_uids = [start_info["scene_info"]["movable_objects"][obj_name]['uid'] 
                            for obj_name in start_info["scene_info"]["movable_objects"]]
        robot_block_uid_pairs = [set([start_info["robot_info"]["uid"], block_uid]) for block_uid in all_blocks_uids]


        for obj_name in start_info["scene_info"]["movable_objects"]:

            obj_start_info = start_info["scene_info"]["movable_objects"][obj_name]
            obj_end_info = end_info["scene_info"]["movable_objects"][obj_name]

            obj_start_pos = start_info["scene_info"]["movable_objects"][obj_name]['current_pos']
            obj_end_pos = end_info["scene_info"]["movable_objects"][obj_name]['current_pos']

            obj_start_contacts = set(c[2] for c in obj_start_info["contacts"])
            obj_end_contacts = set(c[2] for c in obj_end_info["contacts"])

            if dest_link == 'plank_link':
                if not (not in_slider_area(obj_start_pos) and in_slider_area(obj_end_pos)):
                    continue
            elif dest_link == 'drawer_link':
                # Need it to be 
                #print(f"PLACING_GRASPED_BLOCK_DRAWER: {obj_name} before: {in_drawer_area(obj_start_pos)} and after: {in_drawer_area(obj_end_pos)}")

                if (not in_drawer_area(obj_start_pos)) and in_drawer_area(obj_end_pos):
                    #print(f"PLACING_GRASPED_BLOCK_DRAWER: PASSED THE CHECK for ", obj_name)
                    pass
                else:
                    continue
            elif dest_link == 'table_link':
                # At the end, not over table, or over other block
                #print(f"PLACING_GRASPED_BLOCK_TABLE: {obj_name} before: {is_block_over_other_block(obj_name, start_info)} and after: {is_block_over_other_block(obj_name, end_info)}")
                
                # Block was not in table area and got moved to table area
                if not in_table_area(obj_start_pos) and in_table_area(obj_end_pos):
                    pass
                # Block stayed in table area
                elif in_table_area(obj_start_pos) and in_table_area(obj_end_pos):
                    # Block is over other block initially, then not over other block.
                    if is_block_over_other_block(obj_name, start_info) and not is_block_over_other_block(obj_name, end_info):
                        pass
                    else:
                        continue
                else:
                    continue

                # 2 options:
                # Not over table area in the beginning, and then in table area (but not over a block)
                # Over other block in the beginning, and not over other block in the end over other block in the end.
                #print(f"PLACING_GRASPED_BLOCK: PASSED THE CHECK")
                #print(f"{len(obj_start_contacts) == 1}, {len(obj_end_contacts) == 1}, {len(obj_start_contacts - robot_uid) == 0}, {len(obj_end_contacts - robot_uid) == 0}")

            # Robot still holding the object
            # At the end, only the robot is holding the object
            #print(f"PLACING_GRASPED_BLOCK: Contacts for {obj_name} are {obj_start_contacts} and {obj_end_contacts}")
            if len(obj_end_contacts) == 1 and len(obj_end_contacts - robot_uid) == 0:
                #print(f"PLACING_GRASPED_BLOCK: First contact criterion for {obj_name} satisfied!")

                # Initially, either only the robot is contacting the object, or the robot+another object (if cubes are stacked and robot is grasping)
                if len(obj_start_contacts) == 1 and len(obj_start_contacts - robot_uid) == 0:
                    return True
                elif np.any([obj_start_contacts == r_b_pair for r_b_pair in robot_block_uid_pairs]):
                    return True
                #elif set([0, 5]) == obj_start_contacts:
                    # TODO: Maybe remove? This means the block was initially on the ground and in contact with the robot.
                #    return True

        return False

    @staticmethod
    def grasp_block(obj_name, start_info=None, end_info=None):
        """
        Preconditions:
            - The robot is not touching anything.
        Postconditions:
            - The robot has grasped the block obj_name
            - The block obj_name was not moved by more than EPSILON_BLOCK_POS
        """
        robot_contacts_start = set(c[2] for c in start_info["robot_info"]["contacts"])
        obj_start_info = start_info["scene_info"]["movable_objects"][obj_name]
        obj_end_info = end_info["scene_info"]["movable_objects"][obj_name]

        #print(f"GRASP_BLOCK: robot contacts start: {robot_contacts_start}")
        #print(f"GRASP_BLOCK: object contacts end: {[c[2] for c in obj_end_info['contacts']]}")
        if len(robot_contacts_start) > 0:
            #print("Making sure if its a block, it is not grasped.")
            for obj_id in robot_contacts_start:
                if obj_id in [2, 3, 4]:
                    other_object_name = [obj for obj in start_info["scene_info"]["movable_objects"] if start_info["scene_info"]["movable_objects"][obj]['uid'] == obj_id][0]
                    if is_block_grasped(other_object_name, state_info=start_info):
                        return False
                    #print(f"GRASP: object {other_object_name} is not grasped in the beginning!")
                else:
                    return False

        #print(f"GRASP_BLOCK: No initial contact passed for block {obj_name}!")

        start_contacts_links = set((c[2], c[4]) for c in obj_start_info["contacts"])
        end_contacts_links = set((c[2], c[4]) for c in obj_end_info["contacts"])
        #print(f"GRASP_BLOCK: start contacts for {obj_name} are {start_contacts_links} and end are {end_contacts_links}")

        # check the block wasn't lifted from the surface it was on in the beginning
        start_pos = np.array(obj_start_info["current_pos"])
        end_pos = np.array(obj_end_info["current_pos"])
        pos_diff = end_pos - start_pos
        z_diff = pos_diff[2]

        found_surface = False
        for surface_link in ['drawer_link', 'plank_link', 'base_link']:
            surface_uid = start_info["scene_info"]["fixed_objects"]['table']["uid"]
            surface_link_id = start_info["scene_info"]["fixed_objects"]['table']["links"][surface_link]

            if (surface_uid, surface_link_id) in start_contacts_links and (surface_uid, surface_link_id) in end_contacts_links:
                found_surface = True
                break
            elif (surface_uid, surface_link_id) in start_contacts_links and (surface_uid, surface_link_id) not in end_contacts_links:
                found_surface = True
                #print("GRASP_BLOCK: Object z lift ", z_diff)
                if z_diff > EPSILON_GRASP_LIFT:
                    return False
                else:
                    break
            

        if not found_surface:
            start_contacts_objects = set([link[0] for link in start_contacts_links])
            end_contacts_objects = set([link[0] for link in end_contacts_links])
            # The block could also be stacked on some other block
            for block_name in ['block_red', 'block_blue', 'block_pink']:
                block_uid = start_info["scene_info"]["movable_objects"][block_name]["uid"]
                if block_uid in start_contacts_objects and block_uid in end_contacts_objects:
                    found_surface = True
                    break
                elif block_uid in start_contacts_objects and block_uid not in end_contacts_objects:
                    found_surface = True
                    if z_diff > EPSILON_GRASP_LIFT:
                        return False
                    else:
                        break


        if not found_surface:
            return False
        # the block wasn't moved/lifted too much
        start_pos = np.array(obj_start_info["current_pos"])
        end_pos = np.array(obj_end_info["current_pos"])
        pos_diff = np.abs(end_pos - start_pos).sum()
        if pos_diff > EPSILON_BLOCK_POS:
            #print("Grasp: block got moved too much!")
            return False
        
        end_gripper_width = end_info["robot_info"]["gripper_opening_width"]
        #print("GRASP_BLOCK: Gripper width at the end is: ", end_gripper_width)
        if not (0.03 < end_gripper_width < 0.06):
            #print("Grasp: gripper width is not right!")
            return False
        #print(f"GRASP_BLOCK: CHECKING END CONTACTS FOR BLOCK {obj_name}")
        return is_block_grasped(obj_name, end_info) 


    @staticmethod   
    def grasp_slider(start_info=None, end_info=None):
        """
        Preconditions:
            - The robot is not touching anything
        PostconditionsL
            - The robot is grasping the slider 
            - The slider was not moved by more than EPSILON_SLIDER_DRAWER        
        """
        robot_contacts_start = set(c[2] for c in start_info["robot_info"]["contacts"])
        robot_contacts_end = set(c[2] for c in end_info["robot_info"]["contacts"])

        if len(robot_contacts_start) > 0:
            for obj_id in robot_contacts_start:
                if obj_id in [2, 3, 4]:
                    other_object_name = [obj for obj in start_info["scene_info"]["movable_objects"] if start_info["scene_info"]["movable_objects"][obj]['uid'] == obj_id][0]
                    if is_block_grasped(other_object_name, state_info=start_info):
                        return False
                    #print(f"GRASP: object {other_object_name} is not grasped in the beginning!")
                else:
                    return False

        if len(robot_contacts_end) != 1:
            return False


        if was_drawer_slider_moved('base__slide', EPSILON_SLIDER_DRAWER, start_info, end_info) or was_drawer_slider_moved('base__slide', -EPSILON_SLIDER_DRAWER, start_info, end_info):
            return False
        
        return is_slider_grasped(end_info)
        
    @staticmethod   
    def grasp_drawer(start_info=None, end_info=None):
        """
        Preconditions:
            - The robot is not touching anything
        Postconditions
            - The robot is touching the drawer 
            - The drawer was not moved by more than EPSILON_SLIDER_DRAWER        
        """
        robot_contacts_start = set(c[2] for c in start_info["robot_info"]["contacts"])
        robot_contacts_end = set(c[4] for c in end_info["robot_info"]["contacts"])

        if len(robot_contacts_start) > 0:
            for obj_id in robot_contacts_start:
                if obj_id in [2, 3, 4]:
                    other_object_name = [obj for obj in start_info["scene_info"]["movable_objects"] if start_info["scene_info"]["movable_objects"][obj]['uid'] == obj_id][0]
                    if is_block_grasped(other_object_name, state_info=start_info):
                        return False
                    #print(f"GRASP: object {other_object_name} is not grasped in the beginning!")
                else:
                    return False

        if len(robot_contacts_end) != 1:
            return False

        if was_drawer_slider_moved('base__drawer', EPSILON_SLIDER_DRAWER, start_info, end_info) or was_drawer_slider_moved('base__drawer', -EPSILON_SLIDER_DRAWER, start_info, end_info):
            return False

        drawer_link_id = start_info['scene_info']['fixed_objects']['table']['links']['drawer_link']

        if list(robot_contacts_end)[0] == drawer_link_id:
            return True

        return False

    @staticmethod   
    def ungrasp_drawer(start_info=None, end_info=None):
        """
        Preconditions:
            - The robot is touching the drawer
        Postconditions
            - The robot is not touching the drawer 
            - The drawer was not moved by more than EPSILON_SLIDER_DRAWER        
        """

        robot_contacts_start = set(c[4] for c in start_info["robot_info"]["contacts"])
        robot_contacts_end = set(c[4] for c in end_info["robot_info"]["contacts"])
        drawer_link_id = start_info['scene_info']['fixed_objects']['table']['links']['drawer_link']

        if len(robot_contacts_start) != 1 or list(robot_contacts_start)[0] != drawer_link_id:
            return False

        if was_drawer_slider_moved('base__drawer', EPSILON_SLIDER_DRAWER, start_info, end_info) or was_drawer_slider_moved('base__drawer', -EPSILON_SLIDER_DRAWER, start_info, end_info):
            return False

        if len(robot_contacts_end) == 0:
            return True

        return False

    @staticmethod   
    def ungrasp_slider(start_info=None, end_info=None):
        """
        Preconditions:
            - The robot is touching the slider door
        Postconditions
            - The robot is not touching the slider door 
            - The slider door was not moved by more than EPSILON_SLIDER_DRAWER        
        """

        robot_contacts_start = set(c[4] for c in start_info["robot_info"]["contacts"])
        robot_contacts_end = set(c[4] for c in end_info["robot_info"]["contacts"])
        slider_link_id = start_info['scene_info']['fixed_objects']['table']['links']['slide_link']
        #print(f"UNGRASP_SLIDER: robot contacts start are {robot_contacts_start}")

        if len(robot_contacts_start) != 1 or slider_link_id not in robot_contacts_start:
            #print(f"UNGRASP_SLIDER: initial contact is not slider")
            return False
        
        if not is_slider_grasped(state_info=start_info):
            #print(f"UNGRASP_SLIDER: the slider was not grasped initially!")
            return False

        if was_drawer_slider_moved('base__slide', EPSILON_SLIDER_DRAWER, start_info, end_info) or was_drawer_slider_moved('base__slide', -EPSILON_SLIDER_DRAWER, start_info, end_info):
            #print("UNGRASP_SLIDER: slider moved too much")
            return False
        
        #print("UNGRASP_SLIDER: end contacts: ", len(robot_contacts_end))
        if len(robot_contacts_end) == 0:
            return True
        
        return False


    @staticmethod   
    def ungrasp_block(object_names, start_info=None, end_info=None):
        """
        Preconditions:
            - Some block is grasped
        Postconditions
            - The grasped block is now not grasped 
            - The grasped block is in contact with something (e.g. the ground)
            - The grasped block was not moved by more than EPSILON_BLOCK_POS
            - The grasped block was not rotated around z axis by more than EPSILON_ROTATION_DEGREES
        
        Grasp a block nearby the gripper, specified by object_name. Typically called after 
        place_gripper_over_block.
        """

        for object_name in object_names:
            if is_block_grasped(object_name, start_info) and not is_block_grasped(object_name, end_info):
                #print("UNGRASPING: The grasped block is ", object_name)
                obj_start_info = start_info["scene_info"]["movable_objects"][object_name]
                obj_end_info = end_info["scene_info"]["movable_objects"][object_name]
                start_pos = np.array(obj_start_info["current_pos"])
                end_pos = np.array(obj_end_info["current_pos"])
                pos_xy_diff = np.abs(end_pos[:2] - start_pos[:2]).sum()
                
                if pos_xy_diff > EPSILON_BLOCK_POS:
                    #print(f"Ungrasp: {object_name} moved xy too much! by {pos_xy_diff}")
                    continue
                
                start_orn = R.from_quat(obj_start_info["current_orn"])
                end_orn = R.from_quat(obj_end_info["current_orn"])
                rotation = end_orn * start_orn.inv()
                x, y, z = rotation.as_euler("xyz", degrees=True)
                z_degrees = EPSILON_ROTATION_DEGREES
                if abs(z) > z_degrees:
                    #print(f"Ungrasp: {object_name} rotated too much! by {z}")
                    continue
                end_contacts = set((c[2]) for c in obj_end_info["contacts"])

                #print(f"Ungrasp: block end contacts for block {object_name} :", end_contacts)

                if len(end_contacts) == 0:
                    #print("Ungrasp: block end contacts are empty for block", object_name)
                    return False
                
                robot_uid = start_info["robot_info"]["uid"]
                if robot_uid in end_contacts:
                    return False

                return True
            else:
                #print(f"UNGRASPING: need {object_name} to be grasped in the beginning and not in the end!")
                pass
        return False
    
    @staticmethod
    def contact_block(block_name, side, start_info, end_info):
        """
        Preconditions:
            - The robot is not contact anything in the beginning.
        Postconditions:
            - The robot should contact the correct block on the correct side (left or right)
                (relative to the static camera)
            - The block should not be moved by more than EPSILON_BLOCK_TOUCH
        """
        ground_uid = start_info["scene_info"]["fixed_objects"]["table"]["uid"]
        robot_contacts_start = [c[2] for c in start_info["robot_info"]["contacts"]]
        robot_contacts_end_no_ground = [c[2] for c in end_info["robot_info"]["contacts"] if c[2] != ground_uid]
        robot_contacts_links = [c[3] for c in end_info["robot_info"]["contacts"]]
        #robot_contacts_end_ycoord = [c[6][1] for c in end_info["robot_info"]["contacts"]]

        if len(robot_contacts_start) > 0:
            #print(f"CONTACT_BLOCK for {block_name}: something in contact in the beginning: {robot_contacts_start}")
            return False
        
        if len(robot_contacts_end_no_ground) != 1:
            #print(f"CONTACT_BLOCK for {block_name}: multiple robot contacts at the end: {robot_contacts_end_no_ground}!")
            return False
        
        block_end_info = end_info["scene_info"]["movable_objects"][block_name]
        block_end_contacts = set([c[2] for c in block_end_info["contacts"]])

        robot_uid = start_info["robot_info"]["uid"]
        table_drawer_slider_body_uid = 5
        # Robot is touching the block and there are other contacts other than the robot (e.g. floor)
        if table_drawer_slider_body_uid in block_end_contacts and robot_uid in block_end_contacts:
            #print(f"CONTACT_BLOCK for {block_name}: contact end criterion passed!")
            pass
        else:
            #print(f"CONTACT_BLOCK for {block_name}: contact end criterion not passed, end contacts: {block_end_contacts}!")
            return False
        
        #max_vel = 0.5
        #if np.any(np.abs(block_end_info["current_lin_vel"]) > max_vel) or np.any(np.abs(block_end_info["current_ang_vel"]) > max_vel):
        #    return False

        block_start_info = start_info["scene_info"]["movable_objects"][block_name]

        start_pos = np.array(block_start_info["current_pos"])
        end_pos = np.array(block_end_info["current_pos"])
        pos_diff = np.abs(end_pos - start_pos).sum()

        if pos_diff > EPSILON_BLOCK_TOUCH:
            return False
    
        left_gripper_outer = 11
        right_gripper_outer = 9
        # Right/left side of cube
        if side == 'right' and robot_contacts_links[0] == left_gripper_outer:
            return True
        elif side == 'left' and robot_contacts_links[0] == right_gripper_outer:
            return True
        else:
            return False
        
    @staticmethod
    def toggle_light(light_name, start_state, end_state, start_info, end_info):
        # TODO: Add that the robot needs to be over the switch already?
        return (
            start_info["scene_info"]["lights"][light_name]["logical_state"] == start_state
            and end_info["scene_info"]["lights"][light_name]["logical_state"] == end_state
        )

def block_lifting_epsilon(block_pos):
    if abs(block_pos[2] - 0.459988873889266) < EPSILON_GRASP_LIFT or abs(block_pos[2] - 0.459988873889266) < EPSILON_GRASP_LIFT:
        return True
    else:
        return False


def was_drawer_slider_moved(joint_name, threshold, start_info, end_info):
    start_joint_state = start_info["scene_info"]["doors"][joint_name]["current_state"]
    end_joint_state = end_info["scene_info"]["doors"][joint_name]["current_state"]
    #print("Joint state: ", end_joint_state - start_joint_state)
    return (
        0 < threshold < end_joint_state - start_joint_state or 0 > threshold > end_joint_state - start_joint_state
    )

def is_block_grasped(obj_name, state_info):
        robot_contacts = np.array([c[2] for c in state_info["robot_info"]["contacts"]])
        #print("GRASP CHECK: robot contacts ", robot_contacts)
        if len(robot_contacts) == 0:
            return False
        
        target_obj_uid = state_info["scene_info"]["movable_objects"][obj_name]['uid']
        #print("Target uid: ", target_obj_uid)
        # At least 2 points of contact with the target block
        #print("GRASP CHECK: num contacts with obj", obj_name,  np.sum(robot_contacts == target_obj_uid))
        if np.sum(robot_contacts == target_obj_uid) < 2:
            return False
        return True

def is_slider_grasped(state_info):
    slider_link_id = state_info['scene_info']['fixed_objects']['table']['links']['slide_link']
    robot_contacts_list = [c[4] for c in state_info["robot_info"]["contacts"]]
    if np.sum(np.array(robot_contacts_list) == slider_link_id) >= 2:
        return True



def is_gripper_near_block(obj_name, state_info=None):
        target_block_data = [v for k, v in state_info["scene_info"]["movable_objects"].items() if k == obj_name]

        if len(target_block_data) != 1:
            return False
        
        gripper_pos = np.array(state_info['robot_info']['tcp_pos'])
        target_block_pos = np.array(target_block_data[0]['current_pos'])
        start_obj_gripper_diff = np.sqrt(np.sum(np.square(target_block_pos[:2] - gripper_pos[:2])))
        return start_obj_gripper_diff < REL_EPS

def is_block_over_other_block(block_name, state_info=None):

    target_block_pos = np.array(state_info["scene_info"]["movable_objects"][block_name]['current_pos'])
    
    all_other_blocks = np.array([v['current_pos'] for k, v in state_info["scene_info"]["movable_objects"].items() if k != block_name])
    #print(f"CHECK_BLOCK_OVER_BLOCK: Target block pos {target_block_pos}, all other blocks: {all_other_blocks}")

    other_blocks_pos = np.array([v['current_pos'] for k, v in state_info["scene_info"]["movable_objects"].items() 
                            if k != block_name 
                            and in_table_area(v['current_pos']) 
                            and not block_in_drawer(v['current_pos'])])

    if len(other_blocks_pos) == 0:
        return False
    target_other_xy_diffs = np.array([np.sqrt(np.sum(np.square(target_block_pos[:2] - block_pos[:2]))) 
                                      for block_pos in other_blocks_pos])

    #print("block distances: ", target_other_xy_diffs)
    # Rectangle (pink) largest side is 0.1 in length 
    return np.any(target_other_xy_diffs < 0.05) 


def block_in_drawer(block_coord):
    return 0.355 < block_coord[2] < 0.375

def is_drawer_closed(drawer_link_val):
    return drawer_link_val <= 0.03

def in_table_area(xyz):
    x, y, z = xyz[0], xyz[1], xyz[2]
    x_match = (x >= -0.38 and x <= 0.35)
    y_match = (y >=- 0.14 and y <= -0.03)
    return x_match and y_match


def in_slider_area(xyz):
    x, y, z = xyz[0], xyz[1], xyz[2]
    x_match = (x >= -0.38 and x <= 0.18)
    y_match = (y >= 0.05 and y <= 0.12)
    return x_match and y_match

def in_drawer_area(xyz):
    x, y, z = xyz[0], xyz[1], xyz[2]
    #x_match = (x >= -0.38 and x <= 0.18)
    y_match = y < -0.175
    #print("CHECKING IN DRAWER AREA: y is ", y, y_match)
    return y_match

"""
@staticmethod
def place_object(dest_body, dest_link=None, start_info=None, end_info=None):
    
    #Returns True if the object that the robot has currently lifted is placed on the body 'dest_body'
    #(on 'dest_link' if provided).
    #The robot may not touch the object after placing.
    

    # TODO: ADD SO THE OBJECT NEEDS TO BE GRASPED ALREADY?
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
"""
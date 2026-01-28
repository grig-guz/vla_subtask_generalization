import robosuite.utils.transform_utils as transform_utils
import numpy as np


class BaseObjectState:
    def __init__(self):
        pass

    def get_geom_state(self):
        raise NotImplementedError

    def check_contact(self, other):
        raise NotImplementedError

    def check_contain(self, other):
        raise NotImplementedError

    def get_joint_state(self):
        raise NotImplementedError

    def is_open(self):
        raise NotImplementedError

    def is_close(self):
        raise NotImplementedError

    def get_size(self):
        raise NotImplementedError

    def check_ontop(self, other):
        raise NotImplementedError


class ObjectState(BaseObjectState):
    def __init__(self, env, object_name, is_fixture=False):
        self.env = env
        self.object_name = object_name
        self.is_fixture = is_fixture
        self.query_dict = (
            self.env.fixtures_dict if self.is_fixture else self.env.objects_dict
        )
        self.object_state_type = "object"
        self.has_turnon_affordance = hasattr(
            self.env.get_object(self.object_name), "turn_on"
        )
        self.has_open_close_affordance = hasattr(
            self.env.get_object(self.object_name), "is_open"
        )

        self.subtask_init_pos = None
        self.is_grasped_init = None
        self.is_turned_on_init = None

    def set_init_pos(self):
        object_pos = self.env.sim.data.body_xpos[self.env.obj_body_id[self.object_name]]
        self.subtask_init_pos = np.copy(object_pos)
        self.is_grasped_init = self.check_grasped_state()
        if self.has_turnon_affordance:
            self.is_turned_on_init = self.turn_on_state()

        

    def lifted(self):
        if not (isinstance(self.is_grasped_init, bool) or isinstance(self.is_grasped_init, np.bool_)):
            return False
        object_pos = self.env.sim.data.body_xpos[self.env.obj_body_id[self.object_name]]
        return object_pos[2] - self.subtask_init_pos[2] > 0.05 and \
                np.linalg.norm(self.subtask_init_pos[:2] - object_pos[:2]) < 0.125
        

    def get_geom_state(self):
        object_pos = self.env.sim.data.body_xpos[self.env.obj_body_id[self.object_name]]
        object_quat = self.env.sim.data.body_xquat[
            self.env.obj_body_id[self.object_name]
        ]
        return {"pos": object_pos, "quat": object_quat}

    def check_contact(self, other):
        object_1 = self.env.get_object(self.object_name)
        object_2 = self.env.get_object(other.object_name)
        return self.env.check_contact(object_1, object_2)

    def check_ungrasped(self):
        if not (isinstance(self.is_grasped_init, bool) or isinstance(self.is_grasped_init, np.bool_)):
            return False

        object_pos = self.env.sim.data.body_xpos[self.env.obj_body_id[self.object_name]]
        #print(self.is_grasped_init, not self.check_grasped_state(), np.linalg.norm(self.subtask_init_pos[:2] - object_pos[:2]) < 0.125)
        #return not self.check_grasped_state() and \
        return not self.env.check_contact(self.env.get_object(self.object_name), self.env.robots[0].gripper) and \
                np.linalg.norm(self.subtask_init_pos[:2] - object_pos[:2]) < 0.15

    def check_grasped(self):
        #print(f"Checking grasped for object {self.object_name}, grasped before: {self.is_grasped_init}, type?: {type(self.is_grasped_init)}")
        if not (isinstance(self.is_grasped_init, bool) or isinstance(self.is_grasped_init, np.bool_)):
            return False

        #print(f"Object {self.object_name}, grasped before: {self.is_grasped_init}, after: {self.check_grasped_state()}")
        return not self.is_grasped_init and self.check_grasped_state()

    def check_contact_gripper(self):
        gripper = self.env.robots[0].gripper
        object_1 = self.env.get_object(self.object_name)
        return self.env.check_contact(gripper, object_1)

    def check_grasped_state(self):

        gripper = self.env.robots[0].gripper
        object_1 = self.env.get_object(self.object_name)
        object_pos = self.env.sim.data.body_xpos[self.env.obj_body_id[self.object_name]]
        return self.env._check_grasp(gripper, object_1) or (self.check_contact_gripper() and object_pos[2] - self.subtask_init_pos[2] > 0.025)

    def check_contain(self, other):
        object_1 = self.env.get_object(self.object_name)
        object_1_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[self.object_name]
        ]
        object_2 = self.env.get_object(other.object_name)
        object_2_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[other.object_name]
        ]
        return object_1.in_box(object_1_position, object_2_position)

    def get_joint_state(self):
        # Return None if joint state does not exist
        joint_states = []
        for joint in self.env.get_object(self.object_name).joints:
            qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint)
            joint_states.append(self.env.sim.data.qpos[qpos_addr])
        return joint_states

    def check_ontop(self, other):
        this_object = self.env.get_object(self.object_name)
        this_object_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[self.object_name]
        ]
        other_object = self.env.get_object(other.object_name)
        other_object_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[other.object_name]
        ]

        return (
            (this_object_position[2] <= other_object_position[2])
            and self.check_contact(other)
            and (
                np.linalg.norm(this_object_position[:2] - other_object_position[:2])
                < 0.03
            )
        )

                        

    def check_over(self, other):
        this_object = self.env.get_object(self.object_name)
        this_object_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[self.object_name]
        ]
        other_object = self.env.get_object(other.object_name)
        other_object_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[other.object_name]
        ]
        
        return (
            (this_object_position[2] <= other_object_position[2])
            and not self.check_contact(other)
            and (
                np.linalg.norm(this_object_position[:2] - other_object_position[:2])
                < 0.04
            )
        )


    def set_joint(self, qpos=1.5):
        for joint in self.env.get_object(self.object_name).joints:
            self.env.sim.data.set_joint_qpos(joint, qpos)

    def is_open(self):
        for joint in self.env.get_object(self.object_name).joints:
            qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self.env.sim.data.qpos[qpos_addr]
            if self.env.get_object(self.object_name).is_open(qpos):
                return True
        return False

    def is_close(self):
        for joint in self.env.get_object(self.object_name).joints:
            qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self.env.sim.data.qpos[qpos_addr]
            if not (self.env.get_object(self.object_name).is_close(qpos)):
                return False
        return True

    def is_close_state(self):
        for joint in self.env.get_object(self.object_name).joints:
            qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self.env.sim.data.qpos[qpos_addr]
            if not (self.env.get_object(self.object_name).is_close(qpos)):
                return False
        return True


        
    def turn_on(self):
        
        if self.is_turned_on_init == None:
            return self.turn_on_state()
        
        return self.is_turned_on_init == False and self.turn_on_state()

    def turn_on_state(self):
        for joint in self.env.get_object(self.object_name).joints:
            qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self.env.sim.data.qpos[qpos_addr]
            if self.env.get_object(self.object_name).turn_on(qpos):
                return True
        return False

    def turn_off(self):
        for joint in self.env.get_object(self.object_name).joints:
            qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self.env.sim.data.qpos[qpos_addr]
            if not (self.env.get_object(self.object_name).turn_off(qpos)):
                return False
        return True

    def update_state(self):
        if self.has_turnon_affordance:
            self.turn_on()



class SiteObjectState(BaseObjectState):
    """
    This is to make site based objects to have the same API as normal Object State.
    """

    def __init__(self, env, object_name, parent_name, is_fixture=False):
        self.env = env
        self.object_name = object_name
        self.parent_name = parent_name
        self.is_fixture = self.parent_name in self.env.fixtures_dict
        self.query_dict = (
            self.env.fixtures_dict if self.is_fixture else self.env.objects_dict
        )
        self.object_state_type = "site"
        self.has_open_close_affordance = hasattr(
            self.env.get_object(self.parent_name), "is_open"
        )
        self.is_close_init = None
        self.is_open_init = None


    def get_geom_state(self):
        object_pos = self.env.sim.data.get_site_xpos(self.object_name)
        object_quat = transform_utils.mat2quat(
            self.env.sim.data.get_site_xmat(self.object_name)
        )
        return {"pos": object_pos, "quat": object_quat}
    
    def check_grasped(self):
        gripper = self.env.robots[0].gripper

        if 'top' in self.object_name:
            body_id = self.env.sim.model.body_name2id('white_cabinet_1_cabinet_top')
        else:
            body_id = self.env.sim.model.body_name2id('white_cabinet_1_cabinet_middle')
        geom_ids = np.where(self.env.sim.model.geom_bodyid == body_id)[0]
        geom_names = np.array([self.env.sim.model.geom_id2name(g_id) for g_id in geom_ids])
        
        return self.env.check_contact(gripper, geom_names)

    def check_ungrasped(self):
        return not self.check_grasped()


    def check_contain(self, other):
        this_object = self.env.object_sites_dict[self.object_name]
        this_object_position = self.env.sim.data.get_site_xpos(self.object_name)
        this_object_mat = self.env.sim.data.get_site_xmat(self.object_name)

        other_object = self.env.get_object(other.object_name)
        other_object_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[other.object_name]
        ]

        return this_object.in_box(
            this_object_position, this_object_mat, other_object_position
        )

    def check_over(self, other):
        this_object = self.env.object_sites_dict[self.object_name]
        this_object_position = self.env.sim.data.get_site_xpos(self.object_name)
        this_object_mat = self.env.sim.data.get_site_xmat(self.object_name)

        other_object = self.env.get_object(other.object_name)
        other_object_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[other.object_name]
        ]
        parent_object = self.env.get_object(self.parent_name)

        return this_object.over(
            this_object_position, this_object_mat, other_object_position
        ) and not self.env.check_contact(parent_object, other_object)


    def check_contact(self, other):
        """
        There is no dynamics for site objects, so we return true all the time.
        """
        return True

    def check_ontop(self, other):
        this_object = self.env.object_sites_dict[self.object_name]
        if hasattr(this_object, "under"):
            this_object_position = self.env.sim.data.get_site_xpos(self.object_name)
            this_object_mat = self.env.sim.data.get_site_xmat(self.object_name)
            other_object = self.env.get_object(other.object_name)
            other_object_position = self.env.sim.data.body_xpos[
                self.env.obj_body_id[other.object_name]
            ]
            # print(self.object_name, this_object_position)
            # print(other_object_position)
            parent_object = self.env.get_object(self.parent_name)
            if parent_object is None:
                return this_object.under(
                    this_object_position, this_object_mat, other_object_position
                )
            else:
                return this_object.under(
                    this_object_position, this_object_mat, other_object_position
                ) and self.env.check_contact(parent_object, other_object)
        else:
            return True

    def set_joint(self, qpos=1.5):
        for joint in self.env.object_sites_dict[self.object_name].joints:
            self.env.sim.data.set_joint_qpos(joint, qpos)


    def is_open_high(self):
        if self.is_open_init == None:
            return False
        return self.is_open_state()# and self.check_ungrasped()
    
    def is_close_high(self):
        if self.is_close_init == None:
            return False
        return self.is_close_state()# and self.check_ungrasped()

    def is_open_low(self):
        if self.is_open_init == None:
            return False
        return not self.is_open_init and self.is_open_state() and self.check_grasped()
    
    def is_close_low(self):
        if self.is_close_init == None:
            return False
        return not self.is_close_init and self.is_close_state() and self.check_grasped()

    def is_close_state(self):
        for joint in self.env.object_sites_dict[self.object_name].joints:
            qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self.env.sim.data.qpos[qpos_addr]

            if not (self.env.get_object(self.parent_name).is_close(qpos)):
                return False
        return True

    def is_open_state(self):
        for joint in self.env.object_sites_dict[self.object_name].joints:
            qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self.env.sim.data.qpos[qpos_addr]

            if self.env.get_object(self.parent_name).is_open(qpos):
                return True
        return False

    def set_init_pos(self):
        if self.has_open_close_affordance:

            self.is_close_init = self.is_close_state()
            self.is_open_init = self.is_open_state()
            #if self.object_name == "white_cabinet_1_top_region":
           #     breakpoint()
            #print("setting init pos for ", self.object_name, self.is_close_init, self.is_open_init)

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import torch

from . import mdp

##
# Pre-defined configs
##

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


##
# Scene definition
##
_FRANKA_STACK_IK_REL_INIT_JOINT_POS: dict[str, float] = {
    "panda_joint1": 0.0444,
    "panda_joint2": -0.1894,
    "panda_joint3": -0.1107,
    "panda_joint4": -2.5148,
    "panda_joint5": 0.0044,
    "panda_joint6": 2.3775,
    "panda_joint7": 0.6952,
    "panda_finger_joint.*": 0.0400,
}



@configclass
class PracaInzynierskaSkSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.55], 
            joint_pos=_FRANKA_STACK_IK_REL_INIT_JOINT_POS
        ),
        )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5,0,0.3],rot=[1.0,0,0,0]),
        spawn=sim_utils.CuboidCfg(
            size=(1.6, 1.2, 0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.2, 0.2, 0.2), 
            )
        )
    )

    # cube
    cube_1 = RigidObjectCfg( 
        prim_path="{ENV_REGEX_NS}/Cube_1", 
        init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.45, 0.0, 0.62],         
        ),
        spawn=sim_utils.CuboidCfg(
                size=(0.05, 0.05, 0.05),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.5,
                    dynamic_friction=0.4,
                ),
        ),
    )
        
##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
    gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        actions = ObsTerm(func=mdp.last_action)

        # obserwacje pod 3 kostki 
        #object = ObsTerm(func=mdp.object_pos)
        #cube_positions = ObsTerm(func=mdp.cube_positions_in_world_frame)
        #cube_orientations = ObsTerm(func=mdp.cube_orientations_inf_world_frame)

        #obserwacje dla 1 kostki 
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)

        def __post_init__(self) -> None:
            self.enable_corruption = False #szum dla obserwacji
            self.concatenate_terms = True # jeden duzy tensor 

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

  # rand joints, rand cube position, 

    reset_all = EventTerm(func = mdp.reset_scene_to_default,
                          mode = "reset")

    randomize_franka_joint_state = EventTerm(
        func = mdp.reset_joints_by_offset,
        mode = "reset",
      params={
            "position_range": (-0.3, 0.3),  
            "velocity_range": (0.0, 0.0),    
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_cube1_pos = EventTerm(
        func = mdp.reset_root_state_uniform,
        mode = "reset",
        params = {
            "pose_range": {
                "x": (-0.2,0.2),
                "y": (-0.2,0.2),
                "z": (0,0),
                "roll": (0,0),
                "pitch": (0,0),
                "yaw": (-math.pi,math.pi),
                
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_1"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

   # (1) reaching_reward 

   #(2) grasping_reward

   #(3) lifting_reward


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) Klocek zostaje upuszczony lub zrzucony ze stołu
    cube_1_height_below_minimum = DoneTerm(func = mdp.root_height_below_minimum,
                                           params = {
                                               "minimum_height": 0.52,
                                               "asset_cfg": SceneEntityCfg("cube_1"),
                                           },
                                           )

    # (3) Uderzenie robota w stól
    robot_collision = DoneTerm(func = mdp.joint_effort_out_of_limit,
                               params = {
                                   "asset_cfg": SceneEntityCfg("cube_1"),
                               },
                            ) 

    


##
# Environment configuration
##


@configclass
class PracaInzynierskaSkEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: PracaInzynierskaSkSceneCfg = PracaInzynierskaSkSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
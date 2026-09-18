# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv



from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.stack.mdp.observations import object_grasped
from isaaclab.assets import RigidObject
from isaaclab.sensors import FrameTransformer


def grasp_reward (env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, ee_frame_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg,object_grasped_reward: float
                    )-> torch.Tensor:

    check_if_grasped = object_grasped(env,robot_cfg,ee_frame_cfg,object_cfg)

    grasped_reward_val = torch.where(check_if_grasped,object_grasped_reward, 0.0)

    return grasped_reward_val



def reach_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, ee_frame_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg, constant: float)-> torch.Tensor:

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    distance_ee_to_obj = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)
    
    reach_reward_val = torch.exp(-distance_ee_to_obj*constant)

    return reach_reward_val


def lift_reward (env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, min_height: float) -> torch.Tensor:

    object: RigidObject = env.scene[object_cfg.name]

    obj_height = object.data.root_pos_w[:,2]

    check_if_lifted = obj_height > min_height

    return  torch.where(check_if_lifted,1.0,0.0)
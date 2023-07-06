# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    class env: # 环境参数
        num_envs = 4096 # 同时训练的环境数量，headless模式下4096大概占用8G显存
        num_observations = 235 # 观测空间的维度
        num_privileged_obs = None # 特权观测的数量 if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12 # 动作空间的维度，表示每个时间步的动作向量的维度
        env_spacing = 3.  # 环境之间的间隔距离 not used with heightfields/trimeshes 
        send_timeouts = True # 当训练环境的步骤超过指定的时间限制时，算法可以接收到超时信号 send time out information to the algorithm
        episode_length_s = 20 # 每个回合的最大持续时间 episode length in seconds 

    class terrain:
        mesh_type = 'trimesh' # 地形类型 "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # 地形的水平缩放比例 [m]
        vertical_scale = 0.005 # 地形的垂直缩放比例 [m]
        border_size = 25 # 地形边界的大小 [m]
        curriculum = True # 是否使用课程学习来逐步增加地形的难度。通过逐渐引入更复杂的地形类型或配置，课程学习可以帮助训练过程更好地适应不同的地形环境。该参数控制地形的复杂度，允许逐步增加地形的难度级别。
        static_friction = 1.0 # 地形的静摩擦系数
        dynamic_friction = 1.0 # 地形的动摩擦系数
        restitution = 0. # 地形的恢复系数
        # rough terrain only:
        measure_heights = True # 是否测量地形高度
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 测量点的x位置 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] # 测量点的y位置
        selected = False # 选择特定的地形类型 select a unique terrain type and pass all arguments
        terrain_kwargs = None # 地形字典(暂时不知道怎么写) Dict of arguments for selected terrain
        max_init_terrain_level = 5 # 起始课程状态的地形级别 starting curriculum state
        terrain_length = 8. # 地形的长度
        terrain_width = 8. # 地形的宽度
        num_rows= 10 # 地形的行数(等级) number of terrain rows (levels)
        num_cols = 20 # 地形的列数(种类) number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2] # 不同地形类型的比例
        # trimesh only:
        slope_treshold = 0.75 # 用于纠正斜坡的阈值。超过此阈值的斜坡将被修正为垂直表面。slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = False # 是否使用课程学习来逐步增加任务难度，通过逐渐引入更复杂的命令序列，帮助训练过程更好地适应不同的任务。该参数控制命令序列的复杂度，允许逐步增加命令的难度级别。
        max_curriculum = 1. # 定义课程学习的最大阶段
        num_commands = 4 # 命令的数量 default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # 命令更改之前的时间间隔 time before command are changed[s]
        heading_command = True # 是否使用航向命令.如果设置为True，将根据航向误差重新计算角速度命令，以控制机器人的朝向。 if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state: # 初始状态
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P' # 控制类型 P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # PD控制中的刚度参数 [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # PD控制中的阻尼参数 [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5 # 动作缩放系数，用于计算目标角度
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4 # 控制动作更新的频率

    class asset:
        file = "" # 机器人的urdf文件路径
        name = "legged_robot"  # 机器人的名称 actor name
        foot_name = "None" # 足部物体的名称，用于索引身体状态和接触力张量 name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = [] # 接触时惩罚
        terminate_after_contacts_on = [] # 接触时终止仿真
        disable_gravity = False # 禁用重力
        collapse_fixed_joints = True # 合并通过固定连接连接的物体 merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # 固定机器人的基座 fixe the base of the robot
        default_dof_drive_mode = 3 # 默认的关节驱动模式 see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 启用自身碰撞检测 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # 将碰撞柱体替换为胶囊体 replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # 翻转一些.obj模型的可视化模型 Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001 # 物体的密度
        angular_damping = 0. # 物体的角阻尼
        linear_damping = 0. # 物体的线性阻尼
        max_angular_velocity = 1000. # 物体的最大角速度
        max_linear_velocity = 1000. # 物体的最大线速度
        armature = 0. # 物体的关节劲度
        thickness = 0.01 # 物体的厚度

    class domain_rand:
        randomize_friction = True # 随机摩擦力
        friction_range = [0.5, 1.25] # 摩擦力的范围
        randomize_base_mass = False # 随机化基座质量
        added_mass_range = [-1., 1.] # 添加质量的范围
        push_robots = True # 是否对机器人施加推力
        push_interval_s = 15 # 推力之间的时间间隔
        max_push_vel_xy = 1. # 推力的最大xy平面速度

    class rewards:
        class scales: # 奖励函数中各项的权重
            termination = -0.0 # 终止奖励
            tracking_lin_vel = 1.0 # 线速度跟踪奖励
            tracking_ang_vel = 0.5 # 角速度跟踪奖励
            lin_vel_z = -2.0 # 垂直方向线速度奖励
            ang_vel_xy = -0.05 # 水平方向角速度奖励
            orientation = -0. # 姿态（方向）奖励
            torques = -0.00001 # 关节扭矩奖励
            dof_vel = -0. # 关节速度奖励
            dof_acc = -2.5e-7 # 关节加速度奖励
            base_height = -0. # 基座高度奖励
            feet_air_time =  1.0 # 脚部空中时间奖励
            collision = -1. # 碰撞惩罚
            feet_stumble = -0.0 # 脚部失控惩罚
            action_rate = -0.01 # 动作速率惩罚
            stand_still = -0. # 静止惩罚

        only_positive_rewards = True # 将负总奖励截断为零。当设置为True时，如果总奖励为负值，将将其截断为零。这可以防止过早终止训练问题，并确保模型能够学习到积极的奖励信号。如果任务的目标是最大化正奖励，可以将此参数设置为True。 if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # 跟踪奖励中的高斯标准差。跟踪奖励用于衡量机器人与目标值之间的距离或误差。通过指定高斯标准差来调整跟踪奖励的衰减速度。较小的标准差值将使跟踪奖励对误差更为敏感，较大的标准差值将使跟踪奖励对误差更为宽容。 tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # 关节位置限制的软约束，超过该限制将被惩罚 percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1. # 关节速度限制的软约束，超过该限制将被惩罚 
        soft_torque_limit = 1. # 关节扭矩限制的软约束，超过该限制将被惩罚
        base_height_target = 1. # 基座高度的目标值
        max_contact_force = 100. # 接触力的最大值，超过该限制将被惩罚 forces above this value are penalized

    class normalization:
        class obs_scales: # 观测值归一化的比例尺
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100. # 观测值的截断范围
        clip_actions = 100. # 动作值的截断范围

    class noise:
        add_noise = True # 给观测值和动作添加噪声
        noise_level = 1.0 # 噪声的整体缩放系数 scales other values
        class noise_scales: # 各项噪声的缩放系数
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0 # 相机参考的环境编号
        pos = [10, 0, 6]  # 相机位置 [m]
        lookat = [11., 5, 3.]  # 相机观察点 [m]

    class sim:
        dt =  0.005 # 物理模拟的时间步长
        substeps = 1 # 每个时间步长的模拟子步数
        gravity = [0., 0. ,-9.81]  # 重力加速度 [m/s^2]
        up_axis = 1  # 上方向轴 0 is y, 1 is z

        class physx:
            num_threads = 10 # 物理引擎使用的线程数
            solver_type = 1  # 物理引擎的求解器类型 0: pgs, 1: tgs
            num_position_iterations = 4 # 位置迭代的次数
            num_velocity_iterations = 0 # 速度迭代的次数
            contact_offset = 0.01  # 接触偏移量 [m]
            rest_offset = 0.0   # 恢复偏移量 [m]
            bounce_threshold_velocity = 0.5 # 反弹速度的阈值 [m/s]
            max_depenetration_velocity = 1.0 # 最大去穿透速度
            max_gpu_contact_pairs = 2**23 # GPU上可处理的最大接触对数 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5 # 默认缓冲区大小的倍数
            contact_collection = 2 # 接触收集类型 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1 # 设置一个固定的随机种子，以确保实验的可重现性。可以尝试不同的种子值来评估算法的稳定性和一致性。
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0 # 初始噪声标准差控制了策略的探索程度
        actor_hidden_dims = [512, 256, 128] # 整策略和评论家网络的隐藏层维度。
        critic_hidden_dims = [512, 256, 128] # 整策略和评论家网络的隐藏层维度。
        activation = 'elu' # 激活函数 can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0 # 值函数损失的系数
        use_clipped_value_loss = True # 是否使用剪切的值函数损失
        clip_param = 0.2 # 剪切参数
        entropy_coef = 0.01 # 熵系数
        num_learning_epochs = 5 # 训练的迭代次数
        num_mini_batches = 4 # 每个训练迭代中的小批量样本的大小 mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 # 学习率。较大的学习率可能导致不稳定的训练过程，而较小的学习率可能导致收敛速度较慢。 5.e-4
        schedule = 'adaptive' # 学习率调度策略 could be adaptive, fixed
        gamma = 0.99 # 折扣因子
        lam = 0.95 # GAE参数
        desired_kl = 0.01 # 期望的KL散度
        max_grad_norm = 1. # 梯度裁剪的最大范数

    class runner:
        policy_class_name = 'ActorCritic' # 策略类名
        algorithm_class_name = 'PPO' # 算法类名
        num_steps_per_env = 24 # 每个迭代中的环境步数 per iteration
        max_iterations = 1500 # 迭代数 number of policy updates

        # logging
        save_interval = 50 # 保存模型的间隔 check for potential saves every this many iterations
        experiment_name = 'test' # 实验名称
        run_name = '' # 运行名称
        # load and resume
        resume = False # 是否恢复训练
        load_run = -1 # 加载的运行索引 -1 = last run
        checkpoint = -1 # 加载的检查点索引 -1 = last saved model
        resume_path = None # 恢复路径 updated from load_run and chkpt
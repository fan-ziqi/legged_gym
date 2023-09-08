import glob
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
MOTION_FILES = glob.glob('../../datasets/mocap_motions_cyberdog2/*')

class Cyberdog2AMPCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        include_history_steps = None  # Number of steps of history to include.
        num_observations = 42
        num_privileged_obs = 48
        # num_observations = 235
        # num_privileged_obs = 235
        reference_state_initialization = True
        reference_state_initialization_prob = 0.85
        amp_motion_files = MOTION_FILES
        ee_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        get_commands_from_joystick = False

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

    # class terrain( LeggedRobotCfg.terrain ):
    #     mesh_type = 'trimesh'
    #     # measure_heights = False
    #     border_size = 15 # [m] 边界大小
    #     # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, flat]
    #     # terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0]

    #     curriculum = False
    #     terrain_proportions = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0] # 1m
    #     terrain_length = 32 # 地形的长度
    #     terrain_width = 32 # 地形的宽度

    #     num_rows = 1 # number of terrain rows (levels)
    #     num_cols = 1 # number of terrain cols (types)
    #     # measured_points_x = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # 测量点的x位置 1.4mx2.0m rectangle (without center line)
    #     # measured_points_y = [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] # 测量点的y位置


    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.30] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': -0.0 ,  # [rad]
            'RR_hip_joint': -0.0,   # [rad]

            'FL_thigh_joint': 0.66,     # [rad]
            'RL_thigh_joint': 0.66,   # [rad]
            'FR_thigh_joint': 0.66,     # [rad]
            'RR_thigh_joint': 0.66,   # [rad]

            'FL_calf_joint': -1.17,   # [rad]
            'RL_calf_joint': -1.17,    # [rad]
            'FR_calf_joint': -1.17,  # [rad]
            'RR_calf_joint': -1.17,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 25.}  # [N*m/rad]
        damping = {'joint': 0.8}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 6

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/cyberdog2/urdf/cyberdog2.urdf'
        name = "cyberdog2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = [
            "base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        # flip_visual_attachments = False
  
    class domain_rand:
        randomize_friction = True
        friction_range = [0.25, 1.75]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0
        randomize_gains = True
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.03
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.3
            gravity = 0.05
            height_measurements = 0.1

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = 0.0
            tracking_lin_vel = 1.5 * 1. / (.005 * 6)
            tracking_ang_vel = 0.5 * 1. / (.005 * 6)
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            orientation = 0.0
            torques = 0.0
            dof_vel = 0.0
            dof_acc = 0.0
            base_height = 0.0
            feet_air_time =  0.0
            collision = 0.0
            feet_stumble = 0.0
            action_rate = 0.0
            stand_still = 0.0
            dof_pos_limits = 0.0

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 10.
        heading_command = False
        class ranges:
            lin_vel_x = [-0.8, 0.8] # min max [m/s]
            lin_vel_y = [-0.2, 0.2]   # min max [m/s]
            ang_vel_yaw = [-1.57, 1.57]    # min max [rad/s]
            heading = [-3.14, 3.14]

    # class sim( LeggedRobotCfg.sim ):
    #     class physx( LeggedRobotCfg.sim.physx ):
    #         max_gpu_contact_pairs = 2**23 # GPU上可处理的最大接触对数 2**24 -> needed for 8000 envs and more

class Cyberdog2AMPCfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'AMPOnPolicyRunner'
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'amp'
        experiment_name = 'amp_cyberdog2'
        algorithm_class_name = 'AMPPPO'
        policy_class_name = 'ActorCritic'
        max_iterations = 2000

        amp_reward_coef = 2.0
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.3
        amp_discr_hidden_dims = [1024, 512]

        min_normalized_std = [0.01, 0.01, 0.01] * 4

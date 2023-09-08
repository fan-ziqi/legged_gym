from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Cyberdog2RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        # num_observations = 363 # 观测空间的维度
    
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh'
        border_size = 15 # [m] 边界大小
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, flat]
        terrain_proportions = [0.1, 0.1, 0.3, 0.3, 0.2]
        # terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0] # 上楼梯
        num_rows = 10 # number of terrain rows (levels)
        num_cols = 10 # number of terrain cols (types)
        # measured_points_x = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # 测量点的x位置 1.4mx2.0m rectangle (without center line)
        # measured_points_y = [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] # 测量点的y位置

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
        stiffness = {'joint': 20.0}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/cyberdog2/urdf/cyberdog2.urdf'
        name = "cyberdog2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        # flip_visual_attachments = False

    class rewards( LeggedRobotCfg.rewards ):
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.000005
            feet_air_time =  1.5
            base_height = -0.5
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.5
            stand_still = -2.0
            lin_vel_z = -2.0

    class commands( LeggedRobotCfg.commands ):
        curriculum = False
        resampling_time = 4.
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
            heading = [-3.14, 3.14]

class Cyberdog2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01

    class policy( LeggedRobotCfgPPO.policy ):
        # actor_hidden_dims = [128, 64, 32]
        # critic_hidden_dims = [128, 64, 32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'rough'
        experiment_name = 'rough_cyberdog2'
        max_iterations = 1500

########################################平地########################################

class Cyberdog2FlatCfg( Cyberdog2RoughCfg ):
    class env( Cyberdog2RoughCfg.env ):
        num_envs = 4096
        num_observations = 48
  
    class terrain( Cyberdog2RoughCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False
    
    class asset( Cyberdog2RoughCfg.asset ):
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

class Cyberdog2FlatCfgPPO( Cyberdog2RoughCfgPPO ):
    class runner( Cyberdog2RoughCfgPPO.runner ):
        run_name = 'flat'
        experiment_name = 'flat_cyberdog2'
        max_iterations = 500

########################################梅花桩########################################

class Cyberdog2MeihuaCfg( Cyberdog2RoughCfg ):
    class env( Cyberdog2RoughCfg.env ):
        num_envs = 4096
        num_observations = 363 # 观测空间的维度
    
    class terrain( Cyberdog2RoughCfg.terrain ):
        curriculum = False
        terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0] # 梅花桩
        measured_points_x = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # 测量点的x位置 1.4mx2.0m rectangle (without center line)
        measured_points_y = [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] # 测量点的y位置
  
    class rewards( Cyberdog2RoughCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( Cyberdog2RoughCfg.rewards.scales ):
            torques = -0.0002
            feet_air_time =  2.0
            base_height = 2.0
            tracking_lin_vel = 2.0
            stand_still = -2.0

    class commands( Cyberdog2RoughCfg.commands ):
        curriculum = False
        class ranges( Cyberdog2RoughCfg.commands.ranges ):
            lin_vel_x = [-0.0, 0.5] # min max [m/s]
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0.0, 0.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

class Cyberdog2MeihuaCfgPPO( Cyberdog2RoughCfgPPO ):
    #class policy( Cyberdog2RoughCfgPPO.policy ):
        #actor_hidden_dims = [128, 64, 32]
        #critic_hidden_dims = [128, 64, 32]

    class runner( Cyberdog2RoughCfgPPO.runner ):
        run_name = 'meihua'
        experiment_name = 'rough_cyberdog2'
        max_iterations = 1500


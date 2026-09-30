"""
LIMO Pro - Minimal training launch (sim only, no perception, no TTS).

Lance UNIQUEMENT ce qui est necessaire pour le training CM-GAP_SAC:
  - Gazebo Harmonic
  - robot_state_publisher
  - Spawn robot
  - ros_gz_bridge minimal (6 topics: scan, imu, odom, cmd_vel, clock + dynamic_pose)
  - fix_scan_frame.py
  - odom_tf_publisher.py

Pas de camera, pas de perception YOLO+DeepSORT, pas de Nav2.

CRITICAL FIX (current iteration):
    Changed bridge direction syntax from @ (bidirectional) to [ and ]
    (unidirectional). The bidirectional bridge silently fails for high-
    rate sensor topics in this Jazzy + gz-sim 8.11 stack: the topics
    are declared on the ROS side but messages never propagate.

    Direction conventions:
        [  : GZ -> ROS  (sensor data flowing from Gazebo to ROS)
        ]  : ROS -> GZ  (commands flowing from ROS to Gazebo)
        @  : bidirectional (AVOIDED for sensors)

    No GZ_PARTITION is set: training runs in the default Gazebo
    partition (matches the pre-multi-env behavior the user remembers
    as working).

Usage:
    ros2 launch limo_description sim_training.launch.py world:=hospital
    ros2 launch limo_description sim_training.launch.py world:=hospital enable_camera:=true
"""
import os
import subprocess
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def launch_setup(context, *args, **kwargs):
    pkg = get_package_share_directory('limo_description')
    world_arg = LaunchConfiguration('world').perform(context)
    enable_camera_arg = LaunchConfiguration('enable_camera').perform(context)

    # Resolve world file path
    if os.path.isabs(world_arg):
        world = world_arg
    else:
        candidate_names = [f'{world_arg}.sdf', f'{world_arg}_world.sdf']
        world = None
        for name in candidate_names:
            candidate = os.path.join(pkg, 'worlds', name)
            if os.path.exists(candidate):
                world = candidate
                break
        if world is None:
            raise FileNotFoundError(f'Monde introuvable: {world_arg}')

    print(f'[sim_training] Monde      : {world}')
    print(f'[sim_training] Camera     : {enable_camera_arg}')

    xacro_file = os.path.join(pkg, 'urdf', 'limo_ackerman.xacro')
    urdf_out = '/tmp/limo_training.urdf'
    ws_install = os.path.expanduser('~/limo_jazzy_ws/install/setup.bash')
    script_fix = os.path.join(pkg, 'scripts', 'fix_scan_frame.py')
    script_odom = os.path.join(pkg, 'scripts', 'odom_tf_publisher.py')

    _source_cmd = f'. /opt/ros/jazzy/setup.bash && . {ws_install}'

    subprocess.run(
        ['bash', '-c',
         f'{_source_cmd} && ros2 run xacro xacro {xacro_file} '
         f'enable_camera:={enable_camera_arg} -o {urdf_out}'],
        check=True,
    )
    with open(urdf_out, 'r') as f:
        robot_desc = f.read()

    # Single env, default partition. ROS_DOMAIN_ID=20 to isolate from
    # any host-level ROS noise.
    env = {
        'GZ_IP': '127.0.0.1',
        'ROS_DOMAIN_ID': '20',
        'DISPLAY': os.environ.get('DISPLAY', ':0'),
        'XAUTHORITY': os.environ.get('XAUTHORITY', os.path.expanduser('~/.Xauthority')),
        'HOME': os.path.expanduser('~'),
        '__GLX_VENDOR_LIBRARY_NAME': 'nvidia',
    }

    # 1. Gazebo
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', '-v', '3', world],
        additional_env=env, output='screen'
    )

    # 2. robot_state_publisher
    rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        parameters=[{'robot_description': robot_desc, 'use_sim_time': False}],
        output='screen'
    )

    # 3. Bridge - UNIDIRECTIONAL syntax for each topic.
    #    [ means GZ -> ROS  (sensor data from sim to user code)
    #    ] means ROS -> GZ  (commands from user code to sim)
    #    @ would be bidirectional but is unreliable for high-rate sensors.
    bridge = TimerAction(period=2.0, actions=[
        ExecuteProcess(cmd=[
            'bash', '-c',
            f'{_source_cmd} && '
            f'ros2 run ros_gz_bridge parameter_bridge '
            # Commands: ROS -> GZ
            f'/model/limo/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist '
            # Sensors: GZ -> ROS
            f'/model/limo/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry '
            f'/model/limo/laser/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan '
            f'/model/limo/imu@sensor_msgs/msg/Imu[gz.msgs.IMU '
            f'/world/hospital/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V '
            f'/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
        ], additional_env=env, output='screen')
    ])

    # 4. Spawn robot
    spawn = TimerAction(period=4.0, actions=[
        ExecuteProcess(cmd=[
            'bash', '-c',
            f'{_source_cmd} && '
            f'gz sdf -p {urdf_out} > /tmp/limo_training.sdf && '
            f'sed -i "s#model://limo_description/#file://{pkg}/#g" /tmp/limo_training.sdf && '
            f'ros2 run ros_gz_sim create -name limo -file /tmp/limo_training.sdf -x 0 -y 0 -z 0.2'
        ], additional_env=env, output='screen')
    ])

    # 5-6. Fix scan + odom TF (after spawn)
    fix_scan = TimerAction(period=6.0, actions=[
        ExecuteProcess(cmd=['bash', '-c', f'{_source_cmd} && python3 {script_fix}'],
                       additional_env=env, output='screen')
    ])
    odom_tf = TimerAction(period=6.0, actions=[
        ExecuteProcess(cmd=['bash', '-c', f'{_source_cmd} && python3 {script_odom}'],
                       additional_env=env, output='screen')
    ])

    return [gazebo, rsp, bridge, spawn, fix_scan, odom_tf]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'world', default_value='hospital',
            description='Nom du monde (sans extension) ou chemin absolu vers un .sdf',
        ),
        DeclareLaunchArgument(
            'enable_camera', default_value='true',
            description='true pour activer la camera RGBD (eval avec perception). '
                        'false pour le training (gain de SPS).',
        ),
        OpaqueFunction(function=launch_setup),
    ])
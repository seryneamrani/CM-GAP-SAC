"""
LIMO Pro — Simulation base launch (URDF unifie + argument world).

Usage:
    ros2 launch limo_description sim.launch.py            # monde par defaut: limo_world
    ros2 launch limo_description sim.launch.py world:=hospital
    ros2 launch limo_description sim.launch.py world:=warehouse
    ros2 launch limo_description sim.launch.py world:=dynamic

Lance:
  - Gazebo Harmonic avec le monde selectionne
  - robot_state_publisher (TF statiques depuis URDF regenere)
  - Spawn du robot dans Gazebo (meme URDF que RSP)
  - ros_gz_bridge avec tous les capteurs (scan, imu, odometry, cmd_vel, camera)
  - fix_scan_frame.py (reecrit frame_id=laser_link, publie MultiEchoLaserScan)
  - odom_tf_publisher.py (TF odom->base_link)
  - Relays topic_tools :
      /model/limo/odometry -> /odom
      /cmd_vel             -> /model/limo/cmd_vel
  - RViz2

Ce launch ne contient NI Cartographer NI Nav2 — voir mapping.launch.py
et navigation.launch.py pour ca.
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

    # Resolution du chemin du monde : on accepte soit un nom court (hospital)
    # soit un nom complet (hospital_world), soit un chemin absolu.
    if os.path.isabs(world_arg):
        world = world_arg
    else:
        # Si l'utilisateur a donne "hospital" -> hospital_world.sdf
        # Si deja "hospital_world" ou "limo_world" -> .sdf
        candidate_names = [
            f'{world_arg}.sdf',
            f'{world_arg}_world.sdf',
        ]
        world = None
        for name in candidate_names:
            candidate = os.path.join(pkg, 'worlds', name)
            if os.path.exists(candidate):
                world = candidate
                break
        if world is None:
            raise FileNotFoundError(
                f'Monde introuvable: {world_arg}. '
                f'Essaye: limo, hospital, warehouse, dynamic.'
            )

    print(f'[sim.launch] Monde selectionne: {world}')

    xacro_file = os.path.join(pkg, 'urdf', 'limo_ackerman.xacro')
    urdf_out = '/tmp/limo.urdf'
    ws_install = os.path.expanduser('~/limo_jazzy_ws/install/setup.bash')
    script_fix = os.path.join(pkg, 'scripts', 'fix_scan_frame.py')
    script_odom = os.path.join(pkg, 'scripts', 'odom_tf_publisher.py')

    # --- Regeneration synchrone du URDF depuis xacro ---
    _source_cmd = f'. /opt/ros/jazzy/setup.bash && . {ws_install}'
    try:
        subprocess.run(
            ['bash', '-c', f'{_source_cmd} && ros2 run xacro xacro {xacro_file} -o {urdf_out}'],
            check=True
        )
        print(f'[sim.launch] URDF regenere: {urdf_out}')
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'Echec regeneration xacro : {e}')

    with open(urdf_out, 'r') as f:
        robot_desc = f.read()

    env = {'GZ_IP': '127.0.0.1', 'ROS_DOMAIN_ID': '20'}

    # 1. Gazebo
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', '-v', '2', world],
        additional_env=env,
        output='screen'
    )

    # 2. Robot State Publisher
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_desc, 'use_sim_time': False}],
        output='screen'
    )

    # 3. Spawn robot
    spawn = TimerAction(period=4.0, actions=[
        ExecuteProcess(
            cmd=[
                'bash', '-c',
                f'{_source_cmd} && '
                f'gz sdf -p {urdf_out} > /tmp/limo.sdf && '
                f'sed -i "s#model://limo_description/#file://{pkg}/#g" /tmp/limo.sdf && '
                f'ros2 run ros_gz_sim create -name limo -file /tmp/limo.sdf -x 0 -y 0 -z 0.2'
            ],
            additional_env=env,
            output='screen'
        )
    ])

    # 4. Bridge Gazebo <-> ROS (complet, camera incluse)
    bridge = TimerAction(period=7.0, actions=[
        ExecuteProcess(
            cmd=[
                'bash', '-c',
                f'{_source_cmd} && '
                f'ros2 run ros_gz_bridge parameter_bridge '
                f'/model/limo/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist '
                f'/model/limo/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry '
                f'/model/limo/laser/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan '
                f'/model/limo/imu@sensor_msgs/msg/Imu@gz.msgs.IMU '
                f'/model/limo/camera/image@sensor_msgs/msg/Image@gz.msgs.Image '
                f'/model/limo/camera/depth_image@sensor_msgs/msg/Image@gz.msgs.Image '
                f'/model/limo/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo '
                f'/model/limo/camera/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked '
                f'/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock'
            ],
            additional_env=env,
            output='screen'
        )
    ])

    # 5. Fix scan frame
    fix_scan = TimerAction(period=8.0, actions=[
        ExecuteProcess(
            cmd=['bash', '-c', f'{_source_cmd} && python3 {script_fix}'],
            output='screen'
        )
    ])

    # 6. Odom TF publisher
    odom_tf = TimerAction(period=8.0, actions=[
        ExecuteProcess(
            cmd=['bash', '-c', f'{_source_cmd} && python3 {script_odom}'],
            output='screen'
        )
    ])

    # 7. Relays Nav2-friendly
    odom_relay = TimerAction(period=8.5, actions=[
        ExecuteProcess(
            cmd=['bash', '-c', f'{_source_cmd} && python3 ' + os.path.join(pkg, 'scripts', 'odom_relay_fixed.py')],
            output='screen'
        )
    ])

    cmd_vel_relay = TimerAction(period=8.5, actions=[
        ExecuteProcess(
            cmd=['bash', '-c', f'{_source_cmd} && ros2 run topic_tools relay /cmd_vel /model/limo/cmd_vel'],
            output='screen'
        )
    ])

    # 8. RViz
    rviz = TimerAction(period=9.0, actions=[
        ExecuteProcess(
            cmd=['bash', '-c', f'{_source_cmd} && ros2 run rviz2 rviz2'],
            output='screen'
        )
    ])

    return [
        gazebo, rsp, spawn, bridge,
        fix_scan, odom_tf,
        odom_relay, cmd_vel_relay,
        rviz,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value='limo',
            description='Nom du monde (limo, hospital, warehouse, dynamic) ou chemin absolu .sdf'
        ),
        OpaqueFunction(function=launch_setup),
    ])

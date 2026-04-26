import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    pkg = get_package_share_directory('limo_description')
    world = os.path.join(pkg, 'worlds', 'hospital_world.sdf')
    urdf = os.path.join(pkg, 'urdf', 'limo.urdf')
    xacro_file = os.path.join(pkg, 'urdf', 'limo_ackerman.xacro')
    ws_install = os.path.expanduser('~/limo_jazzy_ws/install/setup.bash')
    script_fix = os.path.join(pkg, 'scripts', 'fix_scan_frame.py')
    script_odom = os.path.join(pkg, 'scripts', 'odom_tf_publisher.py')
    nav2_params = os.path.join(pkg, 'config', 'nav2', 'nav2_params.yaml')
    carto_config = os.path.join(pkg, 'config')

    with open(urdf, 'r') as f:
        robot_desc = f.read()

    env = {'GZ_IP': '127.0.0.1', 'ROS_DOMAIN_ID': '20'}
    source_cmd = f'. /opt/ros/jazzy/setup.bash && . {ws_install}'

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
                f'{source_cmd} && '
                f'ros2 run xacro xacro {xacro_file} -o /tmp/limo.urdf && '
                f'gz sdf -p /tmp/limo.urdf > /tmp/limo.sdf && '
                f'sed -i "s#model://limo_description/#file://{pkg}/#g" /tmp/limo.sdf && '
                f'ros2 run ros_gz_sim create -name limo -file /tmp/limo.sdf -x 0 -y 0 -z 0.2'
            ],
            additional_env=env,
            output='screen'
        )
    ])

    # 4. Bridge
    bridge = TimerAction(period=7.0, actions=[
        ExecuteProcess(
            cmd=[
                'bash', '-c',
                f'{source_cmd} && '
                f'ros2 run ros_gz_bridge parameter_bridge '
                f'/model/limo/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist '
                f'/model/limo/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry '
                f'/model/limo/laser/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan '
                f'/model/limo/imu@sensor_msgs/msg/Imu@gz.msgs.IMU '
                f'/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock'
            ],
            additional_env=env,
            output='screen'
        )
    ])

    # 5. Fix scan frame
    fix_scan = TimerAction(period=8.0, actions=[
        ExecuteProcess(
            cmd=['bash', '-c', f'{source_cmd} && python3 {script_fix}'],
            output='screen'
        )
    ])

    # 6. Odom TF publisher
    odom_tf = TimerAction(period=8.0, actions=[
        ExecuteProcess(
            cmd=['bash', '-c', f'{source_cmd} && python3 {script_odom}'],
            output='screen'
        )
    ])

    # 7. Cartographer
    cartographer = TimerAction(period=10.0, actions=[
        ExecuteProcess(
            cmd=[
                'bash', '-c',
                f'{source_cmd} && '
                f'ros2 launch limo_description cartographer.launch.py'
            ],
            output='screen'
        )
    ])

    # 8. Nav2
    nav2 = TimerAction(period=15.0, actions=[
        ExecuteProcess(
            cmd=[
                'bash', '-c',
                f'{source_cmd} && '
                f'ros2 launch limo_description nav2.launch.py'
            ],
            output='screen'
        )
    ])

    # 9. Relay cmd_vel
    relay = TimerAction(period=20.0, actions=[
        ExecuteProcess(
            cmd=[
                'bash', '-c',
                f'{source_cmd} && '
                f'ros2 run topic_tools relay /cmd_vel /model/limo/cmd_vel'
            ],
            output='screen'
        )
    ])

    # 10. RViz
    rviz = TimerAction(period=12.0, actions=[
        ExecuteProcess(
            cmd=['bash', '-c', f'{source_cmd} && ros2 run rviz2 rviz2'],
            output='screen'
        )
    ])

    return LaunchDescription([
        gazebo,
        rsp,
        spawn,
        bridge,
        fix_scan,
        odom_tf,
        cartographer,
        nav2,
        relay,
        rviz,
    ])

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_file   = LaunchConfiguration('world', default='empty.sdf')

    # ── Robot description (avec plugins Gazebo activés) ──────────────────────
    xacro_file = PathJoinSubstitution([
        FindPackageShare('limo_description'), 'urdf', 'limo_ackerman.xacro'
    ])

    robot_description = ParameterValue(
        Command(['xacro ', xacro_file, ' use_gazebo:=true']),
        value_type=str
    )

    # ── Bridge config ─────────────────────────────────────────────────────────
    bridge_config = PathJoinSubstitution([
        FindPackageShare('limo_description'), 'config', 'ros_gz_bridge.yaml'
    ])

    return LaunchDescription([

        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('world', default_value='empty.sdf',
                              description='Gazebo world file'),

        # 1. Gazebo Harmonic
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', world_file],
            output='screen'
        ),

        # 2. Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robot_description': robot_description
            }]
        ),

        # 3. Spawn le robot dans Gazebo
        Node(
            package='ros_gz_sim',
            executable='create',
            name='spawn_limo',
            output='screen',
            arguments=[
                '-name', 'limo',
                '-topic', 'robot_description',
                '-x', '0', '-y', '0', '-z', '0.15'
            ]
        ),

        # 4. ROS <-> GZ Bridge (capteurs + cmd_vel + odom + tf)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='ros_gz_bridge',
            output='screen',
            parameters=[{'config_file': bridge_config}]
        ),
    ])
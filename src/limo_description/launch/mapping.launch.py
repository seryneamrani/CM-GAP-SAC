"""
LIMO Pro — Mapping launch (Cartographer + teleop).

Usage:
    ros2 launch limo_description mapping.launch.py world:=hospital
    ros2 launch limo_description mapping.launch.py world:=warehouse
    ros2 launch limo_description mapping.launch.py world:=dynamic

Lance tout ce qu'il faut pour CONSTRUIRE une carte :
  - sim.launch.py (Gazebo + robot + bridge + TF)
  - cartographer_node (SLAM 2D)
  - cartographer_occupancy_grid_node (publication de /map)
  - RViz2 (deja dans sim.launch.py)

Apres avoir explore le monde en teleop, sauver la carte avec :
    bash ~/limo_jazzy_ws/src/limo_description/scripts/save_map.sh <nom>

Le teleop clavier doit etre lance dans un AUTRE terminal :
    ros2 run teleop_twist_keyboard teleop_twist_keyboard \\
        --ros-args -r /cmd_vel:=/cmd_vel
"""
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg = get_package_share_directory('limo_description')
    config_dir = os.path.join(pkg, 'config')
    sim_launch = os.path.join(pkg, 'launch', 'sim.launch.py')

    world_arg = DeclareLaunchArgument(
        'world',
        default_value='hospital',
        description='Monde Gazebo pour le mapping (hospital, warehouse, dynamic)'
    )

    # 1. Inclure la simulation de base
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_launch),
        launch_arguments={'world': LaunchConfiguration('world')}.items()
    )

    # 2. Cartographer (demarre APRES que la sim et le bridge soient prets)
    cartographer = TimerAction(period=12.0, actions=[
        Node(
            package='cartographer_ros',
            executable='cartographer_node',
            name='cartographer_node',
            output='screen',
            parameters=[{'use_sim_time': False}],
            arguments=[
                '-configuration_directory', config_dir,
                '-configuration_basename', 'limo_2d.lua'
            ],
            remappings=[
                ('scan', '/scan'),
                ('odom', '/odom'),
            ]
        )
    ])

    # 3. Occupancy grid publisher (publie /map pour RViz et pour save_map)
    occupancy_grid = TimerAction(period=12.5, actions=[
        Node(
            package='cartographer_ros',
            executable='cartographer_occupancy_grid_node',
            name='cartographer_occupancy_grid_node',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'resolution': 0.05,
                'publish_period_sec': 1.0,
            }]
        )
    ])

    # Note: le teleop_twist_keyboard doit etre lance MANUELLEMENT dans un
    # autre terminal, car il lit stdin et bloque le terminal du launch.

    return LaunchDescription([
        world_arg,
        sim,
        cartographer,
        occupancy_grid,
    ])

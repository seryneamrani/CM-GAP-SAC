"""
LIMO Pro - Mapping launch (SLAM Toolbox + sim).

Usage:
    ros2 launch limo_description mapping_slamtoolbox.launch.py world:=limo
    ros2 launch limo_description mapping_slamtoolbox.launch.py world:=hospital

Lance:
  - sim.launch.py (Gazebo + bridge + TF + RViz)
  - slam_toolbox (mapping en async mode)

Apres mapping, sauver via service ROS :
    ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap \\
        "{name: {data: '/home/seryne/limo_jazzy_ws/src/limo_description/config/maps/limo'}}"
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
    sim_launch = os.path.join(pkg, 'launch', 'sim.launch.py')
    slam_params = os.path.join(pkg, 'config', 'slam_toolbox.yaml')

    world_arg = DeclareLaunchArgument(
        'world',
        default_value='limo',
        description='Monde Gazebo (limo, hospital, warehouse, dynamic)'
    )

    # 1. Inclure la simulation
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_launch),
        launch_arguments={'world': LaunchConfiguration('world')}.items()
    )

    # 2. SLAM Toolbox (demarre apres que sim soit prete)
    slam = TimerAction(period=12.0, actions=[
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[slam_params]
        )
    ])

    return LaunchDescription([
        world_arg,
        sim,
        slam,
    ])

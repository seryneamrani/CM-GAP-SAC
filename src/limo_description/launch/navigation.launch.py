"""
LIMO Pro — Navigation launch (sim + map_server + AMCL + Nav2).

Usage:
    # A1 baseline (sans perception)
    ros2 launch limo_description navigation.launch.py \\
        world:=hospital map:=hospital perception:=false

    # A2 (avec YOLO + DeepSORT + obstacle_projector)
    ros2 launch limo_description navigation.launch.py \\
        world:=hospital map:=hospital perception:=true

Si l'argument 'map' n'est pas fourni, il prend la valeur de 'world'.
Si 'perception' n'est pas fourni, default true (A2).

Lance:
  - sim.launch.py (Gazebo + robot + bridge + TF + [perception A2])
  - nav2_map_server
  - nav2_amcl
  - nav2 stack (controller, planner, BT, behaviors, smoother, ...)
  - lifecycle_manager
"""
import os
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, DeclareLaunchArgument,
    TimerAction, OpaqueFunction
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _setup(context, *args, **kwargs):
    pkg = get_package_share_directory('limo_description')
    nav2_params = os.path.join(pkg, 'config', 'nav2', 'nav2_params.yaml')
    sim_launch = os.path.join(pkg, 'launch', 'sim.launch.py')

    world = LaunchConfiguration('world').perform(context)
    map_name = LaunchConfiguration('map').perform(context)
    perception = LaunchConfiguration('perception').perform(context)
    if not map_name:
        map_name = world

    maps_dir = os.path.join(pkg, 'config', 'maps')
    candidate_names = [
        f'{map_name}_map.yaml',
        f'{map_name}.yaml',
    ]
    map_yaml = None
    for name in candidate_names:
        candidate = os.path.join(maps_dir, name)
        if os.path.exists(candidate):
            map_yaml = candidate
            break
    if map_yaml is None:
        raise FileNotFoundError(
            f'Carte introuvable: {map_name}. '
            f'Cherchee dans {maps_dir}/ parmi {candidate_names}. '
            f'As-tu lance mapping.launch.py + save_map.sh {map_name} avant?'
        )

    print(f'[navigation.launch] Carte: {map_yaml}')
    print(f'[navigation.launch] Perception A2: {perception}')

    # 1. Sim — propage perception
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_launch),
        launch_arguments={
            'world': world,
            'perception': perception,
        }.items()
    )

    # 2. Map Server
    map_server = TimerAction(period=10.0, actions=[
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'yaml_filename': map_yaml,
                'topic_name': 'map',
                'frame_id': 'map',
            }]
        )
    ])

    # 3. AMCL
    amcl = TimerAction(period=10.5, actions=[
        Node(
            package='nav2_amcl', executable='amcl', name='amcl',
            output='screen', parameters=[nav2_params]
        )
    ])

    # 4. Nav2 stack
    controller = TimerAction(period=11.0, actions=[
        Node(package='nav2_controller', executable='controller_server',
             name='controller_server', output='screen', parameters=[nav2_params])
    ])
    smoother = TimerAction(period=11.0, actions=[
        Node(package='nav2_smoother', executable='smoother_server',
             name='smoother_server', output='screen', parameters=[nav2_params])
    ])
    planner = TimerAction(period=11.0, actions=[
        Node(package='nav2_planner', executable='planner_server',
             name='planner_server', output='screen', parameters=[nav2_params])
    ])
    behaviors = TimerAction(period=11.0, actions=[
        Node(package='nav2_behaviors', executable='behavior_server',
             name='behavior_server', output='screen', parameters=[nav2_params])
    ])
    bt_navigator = TimerAction(period=11.0, actions=[
        Node(package='nav2_bt_navigator', executable='bt_navigator',
             name='bt_navigator', output='screen', parameters=[nav2_params])
    ])
    waypoint_follower = TimerAction(period=11.0, actions=[
        Node(package='nav2_waypoint_follower', executable='waypoint_follower',
             name='waypoint_follower', output='screen', parameters=[nav2_params])
    ])
    velocity_smoother = TimerAction(period=11.0, actions=[
        Node(package='nav2_velocity_smoother', executable='velocity_smoother',
             name='velocity_smoother', output='screen', parameters=[nav2_params])
    ])

    # 5. Lifecycle manager
    lifecycle = TimerAction(period=12.0, actions=[
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'autostart': True,
                'node_names': [
                    'map_server', 'amcl',
                    'controller_server', 'smoother_server', 'planner_server',
                    'behavior_server', 'bt_navigator', 'waypoint_follower',
                    'velocity_smoother',
                ]
            }]
        )
    ])

    return [
        sim,
        map_server, amcl,
        controller, smoother, planner, behaviors,
        bt_navigator, waypoint_follower, velocity_smoother,
        lifecycle,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'world', default_value='hospital',
            description='Monde Gazebo (hospital, warehouse, dynamic)'
        ),
        DeclareLaunchArgument(
            'map', default_value='',
            description='Nom de la carte dans config/maps/. Vide = world.'
        ),
        DeclareLaunchArgument(
            'perception', default_value='true',
            description='Activer YOLO+DeepSORT+Projector (A2). false = A1 pur.'
        ),
        OpaqueFunction(function=_setup),
    ])
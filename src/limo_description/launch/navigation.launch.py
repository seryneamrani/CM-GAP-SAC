"""
LIMO Pro — Navigation launch (sim + map_server + AMCL + Nav2).

Usage:
    ros2 launch limo_description navigation.launch.py \\
        world:=hospital map:=hospital

Si l'argument 'map' n'est pas fourni, il prend la valeur de 'world'.

Lance:
  - sim.launch.py (Gazebo + robot + bridge + TF)
  - nav2_map_server (charge config/maps/<map>_map.yaml et publie /map)
  - nav2_amcl (localisation sur la carte -> publie TF map->odom)
  - nav2 stack complete (controller, planner, BT, behaviors, smoother,
    velocity_smoother, waypoint_follower)
  - lifecycle_manager pour tous les nodes ci-dessus

Workflow :
  1. S'assurer que la carte existe : config/maps/<map>_map.yaml + .pgm
  2. Dans RViz, utiliser "2D Pose Estimate" pour initialiser AMCL
  3. Utiliser "Nav2 Goal" pour envoyer des goals
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
    if not map_name:
        map_name = world  # fallback : meme nom que le monde

    # Resolution du chemin de la carte
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

    # 1. Sim
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_launch),
        launch_arguments={'world': world}.items()
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
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[nav2_params]
        )
    ])

    # 4. Nav2 stack
    controller = TimerAction(period=11.0, actions=[
        Node(
            package='nav2_controller', executable='controller_server',
            name='controller_server', output='screen',
            parameters=[nav2_params]
        )
    ])

    smoother = TimerAction(period=11.0, actions=[
        Node(
            package='nav2_smoother', executable='smoother_server',
            name='smoother_server', output='screen',
            parameters=[nav2_params]
        )
    ])

    planner = TimerAction(period=11.0, actions=[
        Node(
            package='nav2_planner', executable='planner_server',
            name='planner_server', output='screen',
            parameters=[nav2_params]
        )
    ])

    behaviors = TimerAction(period=11.0, actions=[
        Node(
            package='nav2_behaviors', executable='behavior_server',
            name='behavior_server', output='screen',
            parameters=[nav2_params]
        )
    ])

    bt_navigator = TimerAction(period=11.0, actions=[
        Node(
            package='nav2_bt_navigator', executable='bt_navigator',
            name='bt_navigator', output='screen',
            parameters=[nav2_params]
        )
    ])

    waypoint_follower = TimerAction(period=11.0, actions=[
        Node(
            package='nav2_waypoint_follower', executable='waypoint_follower',
            name='waypoint_follower', output='screen',
            parameters=[nav2_params]
        )
    ])

    velocity_smoother = TimerAction(period=11.0, actions=[
        Node(
            package='nav2_velocity_smoother', executable='velocity_smoother',
            name='velocity_smoother', output='screen',
            parameters=[nav2_params]
        )
    ])

    # 5. Lifecycle manager (gere le cycle de vie des 9 nodes)
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
                    'map_server',
                    'amcl',
                    'controller_server',
                    'smoother_server',
                    'planner_server',
                    'behavior_server',
                    'bt_navigator',
                    'waypoint_follower',
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
            'world',
            default_value='hospital',
            description='Monde Gazebo (hospital, warehouse, dynamic)'
        ),
        DeclareLaunchArgument(
            'map',
            default_value='',
            description='Nom de la carte dans config/maps/ (sans extension). '
                        'Si vide, prend la valeur de world.'
        ),
        OpaqueFunction(function=_setup),
    ])

"""
evaluation.launch.py — Orchestre une session d'evaluation complete.

Usage :
  ros2 launch limo_evaluation evaluation.launch.py \\
      world:=hospital architecture:=A1 repetitions:=3

Composants lances :
  - sim + nav2 + AMCL (via navigation.launch.py de limo_description)
  - evaluation_node (collecte metriques)
  - rosbag2 record (archivage)
  - goal_runner (envoie les goals)

Attention : avant lancement, la carte du monde doit exister dans
config/maps/<world>_map.yaml du package limo_description.
"""
import os
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, DeclareLaunchArgument,
    TimerAction, ExecuteProcess, OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _setup(context, *args, **kwargs):
    pkg_eval = get_package_share_directory('limo_evaluation')
    pkg_desc = get_package_share_directory('limo_description')
    
    world = LaunchConfiguration('world').perform(context)
    arch = LaunchConfiguration('architecture').perform(context)
    reps = int(LaunchConfiguration('repetitions').perform(context))
    
    eval_params = os.path.join(pkg_eval, 'config', 'eval_params.yaml')
    waypoints = os.path.join(pkg_eval, 'config', 'waypoints.yaml')
    nav_launch = os.path.join(pkg_desc, 'launch', 'navigation.launch.py')
    
    output_dir = f'/tmp/limo_eval_results/{arch}_{world}'
    bag_path = f'{output_dir}/bag_{arch}_{world}'
    
    # 1. Navigation (sim + amcl + nav2)
    nav = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav_launch),
        launch_arguments={'world': world, 'map': world}.items(),
    )
    
    # 2. Evaluation node
    eval_node = TimerAction(period=15.0, actions=[
        Node(
            package='limo_evaluation',
            executable='evaluation_node',
            name='evaluation_node',
            output='screen',
            parameters=[eval_params, {'output_dir': output_dir}],
        )
    ])
    
    # 3. Bag recording (topics critiques)
    bag_record = TimerAction(period=15.0, actions=[
        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'record',
                '-o', bag_path,
                '/scan', '/odom', '/cmd_vel',
                '/tf', '/tf_static',
                '/plan', '/local_plan',
                '/amcl_pose', '/initialpose',
                '/eval/run_config', '/eval/metrics_live',
                '/model/limo/camera/image',  # utile pour A2/A4 reanalyse
            ],
            output='screen',
        )
    ])
    
    # 4. Goal runner (lance avec un delai pour laisser tout le reste demarrer)
    goal_runner = TimerAction(period=25.0, actions=[
        Node(
            package='limo_evaluation',
            executable='goal_runner',
            name='goal_runner',
            output='screen',
            parameters=[{
                'world': world,
                'architecture': arch,
                'repetitions': reps,
                'waypoints_file': waypoints,
            }],
        )
    ])
    
    return [nav, eval_node, bag_record, goal_runner]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value='hospital',
            description='Monde a evaluer (hospital, warehouse, dynamic, limo)',
        ),
        DeclareLaunchArgument(
            'architecture',
            default_value='A1',
            description='Architecture testee (A1, A2, A3, A4)',
        ),
        DeclareLaunchArgument(
            'repetitions',
            default_value='3',
            description='Nb de repetitions par paire (default 3 -> 15 runs total)',
        ),
        OpaqueFunction(function=_setup),
    ])

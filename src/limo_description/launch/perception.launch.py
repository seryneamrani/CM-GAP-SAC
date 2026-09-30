"""
LIMO Pro — Perception launch (standalone, sim OR real robot).

Objectif : produire /tracked_obstacles (YOLO+DeepSORT) pour alimenter
    - la branche CM-GAP-SAC (level:=minimal suffit), et/ou
    - la branche Nav2 classique (level:=full : + projector + classifier).

Ce launch est SEPARE de sim.launch.py pour pouvoir tester la perception
isolement, et pour tourner tel quel sur le robot reel (use_sim:=false).

------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------
# Cas A — sim deja lancee SANS perception ni bridge camera :
#   Terminal 1 : ros2 launch limo_description sim.launch.py world:=hospital perception:=false
#   Terminal 2 : ros2 launch limo_perception perception.launch.py use_sim:=true level:=minimal
#
#   ATTENTION: sim.launch.py bridge ET relaie deja la camera dans sa version
#       actuelle. Si tu le lances tel quel, NE PAS remettre use_sim:=true
#       ici (double bridge). Deux options propres :
#         (1) retirer le bloc camera + relays de sim.launch.py, puis
#             use_sim:=true ici  (recommande : un seul endroit bridge).
#         (2) garder sim.launch.py tel quel et lancer ce launch en
#             use_sim:=false   (il consomme /camera/image_raw deja relaye).
#
# Cas B — robot reel (pas de Gazebo) :
#   ros2 launch limo_perception perception.launch.py use_sim:=false level:=full
#
# Cas C — brancher CM-GAP-SAC en real_perception :
#   (sim sans perception) + ce launch en minimal, puis lancer le training
#   avec tracker_mode: real_perception dans le YAML.
------------------------------------------------------------------------

Arguments:
    use_sim     true  -> bridge GZ->ROS camera + relays vers /camera/image_raw
                false -> robot reel, topics natifs, aucun bridge
    level       minimal -> perception_node seul (-> /tracked_obstacles)
                full    -> + obstacle_projector + track_classifier
    prediction  true/false  (full uniquement) -> + trajectory_predictor (LSTM)
    tts         true/false  (full uniquement) -> scene_describer + espeak
    lstm_model  chemin du checkpoint Social-LSTM Lite
    domain_id   ROS_DOMAIN_ID (doit matcher sim.launch.py = 20)
    debug_image true -> perception_node publie /perception/debug_image
"""
import os
from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess, TimerAction, DeclareLaunchArgument, OpaqueFunction,
)
from launch.substitutions import LaunchConfiguration


def launch_setup(context, *args, **kwargs):
    use_sim = LaunchConfiguration('use_sim').perform(context).lower() in ('true', '1', 'yes')
    level = LaunchConfiguration('level').perform(context).lower()
    prediction = LaunchConfiguration('prediction').perform(context).lower() in ('true', '1', 'yes')
    tts = LaunchConfiguration('tts').perform(context).lower() in ('true', '1', 'yes')
    lstm_model = LaunchConfiguration('lstm_model').perform(context)
    domain_id = LaunchConfiguration('domain_id').perform(context)
    debug_image = LaunchConfiguration('debug_image').perform(context).lower() in ('true', '1', 'yes')

    full = (level == 'full')

    # --- Topics camera : sim (gz, a bridger/relayer) vs reel (natifs) ---
    # En sim, le RGBD gz publie /model/limo/camera/*. perception_node
    # souscrit a /camera/image_raw : on relaie. Sur le reel, l'Orbbec Dabai
    # publie deja un nom ROS standard -> on le remappe directement.
    if use_sim:
        cam_image_in = '/camera/image_raw'        # apres relay (voir plus bas)
        cam_info_in = '/camera/camera_info'
        cam_depth_in = '/camera/depth/image_raw'
    else:
        # Orbbec Dabai (typique). Ajuste si ton `ros2 topic list` differe.
        cam_image_in = '/camera/color/image_raw'
        cam_info_in = '/camera/color/camera_info'
        cam_depth_in = '/camera/depth/image_raw'

    paths = _resolve_paths()
    src = f'. /opt/ros/jazzy/setup.bash && . {paths["ws_install"]}'
    env = {'ROS_DOMAIN_ID': str(domain_id), 'GZ_IP': '127.0.0.1'}

    print(f'[perception.launch] use_sim={use_sim}  level={level}  '
          f'prediction={prediction}  tts={tts}  domain={domain_id}')

    actions = []

    # ---------------------------------------------------------------
    # (sim only) Bridge GZ->ROS camera + relays vers /camera/image_raw
    # ---------------------------------------------------------------
    if use_sim:
        # Direction `[` = GZ->ROS seulement (camera = lecture seule).
        cam_bridge = ExecuteProcess(
            cmd=['bash', '-c',
                 f'{src} && ros2 run ros_gz_bridge parameter_bridge '
                 f'/model/limo/camera/image@sensor_msgs/msg/Image[gz.msgs.Image '
                 f'/model/limo/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image '
                 f'/model/limo/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo'],
            additional_env=env, output='screen')

        relay_img = ExecuteProcess(
            cmd=['bash', '-c',
                 f'{src} && ros2 run topic_tools relay '
                 f'/model/limo/camera/image /camera/image_raw'],
            additional_env=env, output='screen')
        relay_info = ExecuteProcess(
            cmd=['bash', '-c',
                 f'{src} && ros2 run topic_tools relay '
                 f'/model/limo/camera/camera_info /camera/camera_info'],
            additional_env=env, output='screen')
        relay_depth = ExecuteProcess(
            cmd=['bash', '-c',
                 f'{src} && ros2 run topic_tools relay '
                 f'/model/limo/camera/depth_image /camera/depth/image_raw'],
            additional_env=env, output='screen')

        actions.append(cam_bridge)
        # Relays apres le bridge (laisse le bridge creer les topics).
        actions.append(TimerAction(period=2.0, actions=[relay_img, relay_info, relay_depth]))
        perception_delay = 4.0   # laisse le temps au bridge + relays
    else:
        perception_delay = 0.5

    # ---------------------------------------------------------------
    # perception_node (YOLO + DeepSORT) — TOUJOURS (minimal & full)
    # ---------------------------------------------------------------
    perception_node = TimerAction(period=perception_delay, actions=[
        ExecuteProcess(
            cmd=['bash', '-c',
                 f'{src} && python3 {paths["perception"]} '
                 f'--ros-args '
                 f'-r /camera/image_raw:={cam_image_in} '
                 f'-p publish_debug_image:={"true" if debug_image else "false"}'],
            additional_env=env, output='screen')
    ])
    actions.append(perception_node)

    # ---------------------------------------------------------------
    # (full only) Projector + Classifier (+ Predictor + Describer)
    # ---------------------------------------------------------------
    if full:
        projector_node = TimerAction(period=perception_delay + 2.0, actions=[
            ExecuteProcess(
                cmd=['bash', '-c',
                     f'{src} && python3 {paths["projector"]} '
                     f'--ros-args '
                     f'-r /camera/depth/image_raw:={cam_depth_in} '
                     f'-r /camera/camera_info:={cam_info_in}'],
                additional_env=env, output='screen')
        ])
        classifier_node = TimerAction(period=perception_delay + 3.0, actions=[
            ExecuteProcess(
                cmd=['bash', '-c', f'{src} && python3 {paths["classifier"]}'],
                additional_env=env, output='screen')
        ])
        actions.extend([projector_node, classifier_node])

        if prediction:
            predictor_node = TimerAction(period=perception_delay + 4.0, actions=[
                ExecuteProcess(
                    cmd=['bash', '-c',
                         f'{src} && python3 {paths["predictor"]} '
                         f'--ros-args -p model_path:={lstm_model}'],
                    additional_env=env, output='screen')
            ])
            actions.append(predictor_node)

        if tts:
            describer_node = TimerAction(period=perception_delay + 5.0, actions=[
                ExecuteProcess(
                    cmd=['bash', '-c',
                         f'{src} && python3 {paths["describer"]} '
                         f'--ros-args -p tts_enabled:=true'],
                    additional_env=env, output='screen')
            ])
            actions.append(describer_node)

    return actions


def _resolve_paths():
    """Chemins des scripts perception. Ils vivent dans scripts/ (pas
    dans le module installe), donc on pointe vers les sources, comme
    sim.launch.py le fait deja."""
    perception_dir = os.path.expanduser(
        '~/limo_jazzy_ws/src/limo_perception/scripts')
    return {
        'ws_install': os.path.expanduser('~/limo_jazzy_ws/install/setup.bash'),
        'perception': os.path.join(perception_dir, 'perception_node.py'),
        'projector': os.path.join(perception_dir, 'obstacle_projector.py'),
        'classifier': os.path.join(perception_dir, 'track_classifier_node.py'),
        'predictor': os.path.join(perception_dir, 'trajectory_predictor_node.py'),
        'describer': os.path.join(perception_dir, 'scene_describer.py'),
    }


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim', default_value='true',
            description='true: bridge+relay camera GZ->ROS. false: robot reel.'),
        DeclareLaunchArgument('level', default_value='minimal',
            description='minimal: perception_node seul. full: + projector + classifier.'),
        DeclareLaunchArgument('prediction', default_value='false',
            description='(full) activer trajectory_predictor LSTM.'),
        DeclareLaunchArgument('tts', default_value='false',
            description='(full) activer scene_describer + espeak.'),
        DeclareLaunchArgument('lstm_model',
            default_value=os.path.expanduser(
                '~/limo_jazzy_ws/src/limo_perception/models/social_lstm_lite.pt'),
            description='Checkpoint Social-LSTM Lite.'),
        DeclareLaunchArgument('domain_id', default_value='20',
            description='ROS_DOMAIN_ID (doit matcher sim.launch.py).'),
        DeclareLaunchArgument('debug_image', default_value='true',
            description='perception_node publie /perception/debug_image.'),
        OpaqueFunction(function=launch_setup),
    ])
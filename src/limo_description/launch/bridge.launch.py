#!/usr/bin/env python3
"""
bridge.launch.py V2 -- ros_gz_bridge + TF tree complet pour LIMO Pro + Gazebo Harmonic

Bridges :
  GZ -> ROS :
    /clock, /camera/image_raw, /camera/depth/image_raw, /camera/camera_info,
    /camera/points, /scan, /scan/points, /imu, /odom, /tf
  ROS -> GZ :
    /cmd_vel

Static transforms (pour reconnecter l'arbre TF Gazebo namespace 'limo/' avec
l'URDF non-namespace) :
  map           -> limo/odom        (identite, en attendant Cartographer)
  limo/base_link -> base_link       (identite, pont entre les 2 namespaces)

Apres lancement, l'arbre TF connecte :
  map -> limo/odom -> limo/base_link -> base_link -> {camera, laser, imu}
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Utiliser le clock simule'
    )

    # ros_gz_bridge args
    bridge_args = [
        # Horloge
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',

        # Camera RGBD
        '/model/limo/camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
        '/model/limo/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        '/model/limo/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
        '/model/limo/camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',

        # LiDAR
        '/model/limo/laser/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/model/limo/laser/scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',

        # IMU
        '/model/limo/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',

        # Odometry
        '/model/limo/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry',

        # TF (Pose_V -> TFMessage) -- contient limo/odom -> limo/base_link
        '/model/limo/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',

        # Commandes (ROS -> GZ)
        '/model/limo/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
    ]

    bridge_remappings = [
        ('/model/limo/camera/image',         '/camera/image_raw'),
        ('/model/limo/camera/camera_info',   '/camera/camera_info'),
        ('/model/limo/camera/depth_image',   '/camera/depth/image_raw'),
        ('/model/limo/camera/points',        '/camera/points'),
        ('/model/limo/laser/scan',           '/scan'),
        ('/model/limo/laser/scan/points',    '/scan/points'),
        ('/model/limo/imu',                  '/imu'),
        ('/model/limo/odometry',             '/odom'),
        ('/model/limo/tf',                   '/tf'),
        ('/model/limo/cmd_vel',              '/cmd_vel'),
    ]

    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        arguments=bridge_args,
        remappings=bridge_remappings,
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    # Static transform : pont entre namespace Gazebo 'limo/' et URDF
    static_tf_robot = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_limo_to_robot',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', 'limo/base_link',
            '--child-frame-id', 'base_link',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='log',
    )

    # Static transform : map -> limo/odom (en attendant Cartographer)
    # Quand Cartographer sera lance, supprimer ce node ou il y aura conflit
    static_tf_map = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_map_to_odom',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', 'map',
            '--child-frame-id', 'limo/odom',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='log',
    )

    return LaunchDescription([
        declare_use_sim_time,
        bridge_node,
        static_tf_robot,
        static_tf_map,
    ])

"""Launch file: bring up Gazebo Harmonic + ros_gz bridge + perception bridge.

This launch is for SMOKE-TESTING the env. Training will get its own launch
in a later prompt that also starts the train node.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    world_arg = DeclareLaunchArgument(
        "world",
        default_value="warehouse",
        description="Gazebo world: warehouse | hospital | hybrid",
    )

    use_gt_arg = DeclareLaunchArgument(
        "use_gt_pedestrians",
        default_value="false",
        description="Use Gazebo ground-truth pedestrian poses (ablation A5)",
    )

    # ros_gz_bridge for /scan, /odom, /cmd_vel, /clock
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
        ],
        output="screen",
    )

    # The world bringup (Gazebo + LIMO Pro spawn) is assumed to be already
    # running, or to be launched separately from the user's existing
    # full_simulation.launch.py.

    return LaunchDescription([
        world_arg,
        use_gt_arg,
        bridge,
    ])

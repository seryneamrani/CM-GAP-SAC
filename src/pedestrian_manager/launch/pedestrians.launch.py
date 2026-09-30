from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg = get_package_share_directory("pedestrian_manager")
    cfg = os.path.join(pkg, "config", "pedestrians.yaml")

    bridge_args = []
    for i in range(1, 9):
        # ROS -> Gazebo (le manager publie, Gazebo reçoit)
        bridge_args.append(
            f"/model/ped_{i}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist")
        # Gazebo -> ROS (Gazebo publie, le manager reçoit)
        bridge_args.append(
            f"/model/ped_{i}/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose")

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=bridge_args,
        output="screen",
    )

    manager = Node(
        package="pedestrian_manager",
        executable="pedestrian_manager_node",
        name="pedestrian_manager",
        output="screen",
        parameters=[{"config_file": cfg}],
    )

    return LaunchDescription([bridge, manager])
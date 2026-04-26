from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    urdf_file = PathJoinSubstitution([
        FindPackageShare('limo_description'),
        'urdf',
        'limo.urdf.xacro'
    ])

    robot_description = {'robot_description': Command([
        FindExecutable(name='xacro'),
        ' ',
        urdf_file
    ])}

    return LaunchDescription([
        # robot_state_publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[robot_description]
        ),

        # RViz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', PathJoinSubstitution([
                FindPackageShare('limo_description'),
                'rviz',
                'limo.rviz'
            ])]
        )
    ])

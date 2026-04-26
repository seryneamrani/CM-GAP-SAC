from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Chemin vers le package et le xacro
    limo_description_share = FindPackageShare('limo_description')
    xacro_file = PathJoinSubstitution(
        [limo_description_share, 'urdf', 'limo.urdf.xacro']
    )

    # Argument pour pouvoir changer le fichier xacro si besoin
    declare_model_path = DeclareLaunchArgument(
        'model',
        default_value=xacro_file,
        description='Full path to the URDF/Xacro file'
    )

    # Node pour publier l'état du robot
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': LaunchConfiguration('model')}]
    )

    # Node pour visualiser dans RViz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', PathJoinSubstitution([limo_description_share, 'rviz', 'limo.rviz'])]
    )

    return LaunchDescription([
        declare_model_path,
        robot_state_publisher_node,
        rviz_node
    ])

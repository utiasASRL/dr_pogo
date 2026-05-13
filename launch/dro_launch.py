from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description() -> LaunchDescription:
    output_path = PathJoinSubstitution(
        [FindPackageShare("dr_pogo"), "output"]
    )
    return LaunchDescription(
        [
            Node(
                package="dr_pogo",
                executable="dro_node",
                name="dro_node",
                output="screen",
                parameters=[
                    {"output_path": output_path}
                ],
            ),
        ]
    )

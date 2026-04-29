from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description() -> LaunchDescription:
	rviz_file = PathJoinSubstitution(
           [FindPackageShare("dr_pogo"), "config", "rviz.rviz"])
	return LaunchDescription(
		[
			Node(
				package="dr_pogo",
				executable="dro_node",
				name="dro_node",
				output="screen",
			),
			Node(
				package="dr_pogo",
				executable="raplace_node",
				name="raplace_node",
				output="screen",
			),
            Node(
				package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_file]
            ),
		]
	)

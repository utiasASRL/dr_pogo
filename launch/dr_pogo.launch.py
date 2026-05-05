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
				package="dr_pogo",
				executable="registration_node",
				name="registration_node",
				output="screen",
			),
			Node(
				package="dr_pogo",
				executable="pogo_node",
				name="pogo_node",
				output="screen",
				parameters=[
					{"config_file": PathJoinSubstitution(
						[FindPackageShare("dr_pogo"), "config", "config_pogo.yaml"]
					)}
				],
			),
            Node(
				package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_file]
            ),
			Node(
				package="tf2_ros",
				executable="static_transform_publisher",
				name="static_transform_publisher",
				output="screen",
				arguments=["0", "0", "0", "0", "0", "0", "odom", "map"]
			),
		]
	)

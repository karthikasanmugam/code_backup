from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    project_root = LaunchConfiguration("project_root")
    host = LaunchConfiguration("host")
    port = LaunchConfiguration("port")
    debug = LaunchConfiguration("debug")

    return LaunchDescription([
        DeclareLaunchArgument(
            "project_root",
            default_value="/home/s1/.ihub/vla/lerobot_2",
            description="Absolute path to the lerobot_2 checkout containing app.py, ui/, and lerobot/.",
        ),
        DeclareLaunchArgument("host", default_value="0.0.0.0"),
        DeclareLaunchArgument("port", default_value="8104"),
        DeclareLaunchArgument("debug", default_value="false"),
        SetEnvironmentVariable("DAKSHA_PROJECT_ROOT", project_root),
        SetEnvironmentVariable("DAKSHA_HOST", host),
        SetEnvironmentVariable("DAKSHA_PORT", port),
        SetEnvironmentVariable("DAKSHA_DEBUG", debug),
        Node(
            package="daksha_vla_ui",
            executable="daksha_vla_ui",
            name="daksha_vla_ui",
            output="screen",
        ),
    ])

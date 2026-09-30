#!/bin/bash
source ~/limo_jazzy_ws/install/setup.bash

run_bridge() {
    local name="$1"; shift
    while true; do
        echo "[bridge-$name] starting..."
        "$@"
        echo "[bridge-$name] died, restarting in 1s..."
        sleep 1
    done
}

run_bridge scan ros2 run ros_gz_bridge parameter_bridge \
    /model/limo/laser/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan &

run_bridge imu ros2 run ros_gz_bridge parameter_bridge \
    /model/limo/imu@sensor_msgs/msg/Imu[gz.msgs.IMU &

run_bridge odom ros2 run ros_gz_bridge parameter_bridge \
    /model/limo/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry &

run_bridge cmd_vel ros2 run ros_gz_bridge parameter_bridge \
    /model/limo/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist &

run_bridge clock ros2 run ros_gz_bridge parameter_bridge \
    /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock &

wait

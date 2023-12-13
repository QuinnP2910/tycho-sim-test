#!/bin/bash

export DISPLAY=:0.0  # Set the display, adjust the value if needed

# Run roscore in a new terminal
gnome-terminal --tab --title="roscore" -- bash -c "roscore; exec bash"

# Run usb_cam_node in a new terminal
gnome-terminal --tab --title="usb_cam" -- bash -c "rosrun usb_cam usb_cam_node _image_width:=1280 _image_height:=720; exec bash"

# Run camera_calibration in a new terminal
# Commented out as it's not clear whether you want to run it or not
# gnome-terminal --tab --title="camera_calibration" -- bash -c "rosrun camera_calibration cameracalibrator.py --size 8x6 --square 0.025 image:=/usb_cam/image_raw camera:=/usb_cam; exec bash"

# Run continuous_detection.launch in a new terminal
gnome-terminal --tab --title="continuous_detection" -- bash -c "roslaunch apriltag_ros continuous_detection.launch; exec bash"

# Run TagPublisher.py in a new terminal
gnome-terminal --tab --title="TagPublisher" -- bash -c "python TagPublisher.py; exec bash"

# Run TychoSimAprilTag.py in a new terminal
gnome-terminal --tab --title="TychoSimAprilTag" -- bash -c "python TychoSimAprilTag.py; exec bash"

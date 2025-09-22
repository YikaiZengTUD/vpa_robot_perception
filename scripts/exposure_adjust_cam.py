#!/usr/bin/env python3
import subprocess, shlex, time
import rospy
from sensor_msgs.msg import Image
DEV = "/dev/video0"
EXPO_HI = 5000   # warm-up exposure
EXPO_LO = 300    # operating exposure

def sh(cmd):
    return subprocess.call(shlex.split(cmd))

# 1) warm-up exposure before starting node
sh(f"v4l2-ctl -d {DEV} --set-ctrl=auto_exposure=1,exposure_time_absolute={EXPO_HI}")
sh(f"v4l2-ctl -d {DEV} --set-ctrl=iso_sensitivity=2")
time.sleep(1.0)
# show current exposure
print("Current exposure before launch:")
sh(f"v4l2-ctl -d {DEV} --get-ctrl=exposure_time_absolute")
# 2) start camera node
cam = subprocess.Popen(
    shlex.split("roslaunch vpa_robot_interface vpa_camera.launch")
)

rospy.init_node('exposure_adjust_cam', anonymous=False)
img_msg = rospy.wait_for_message('robot_cam/image_raw', Image)
print("[CAM] Image received, adjusting exposure...")

# 4) drop to operating exposure
sh(f"v4l2-ctl -d {DEV} --set-ctrl=auto_exposure=1,exposure_time_absolute={EXPO_LO}")

# 5) show final exposure
sh(f"v4l2-ctl -d {DEV} --get-ctrl=exposure_time_absolute")

# 6) keep script alive until roslaunch exits
cam.wait()

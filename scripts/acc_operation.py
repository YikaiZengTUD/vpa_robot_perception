#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
import cv2
import numpy as np
from cv_bridge import CvBridge
import socket
# import your existing detector
from toolbox.front_car_detector import FrontCarDetector

EPS = 1e-5

class ACCLeadNode:

    def __init__(self):
        rospy.init_node("acc_lead_node")

        self.robot_name = socket.gethostname()
        
        self.detector = FrontCarDetector()

        self.bridge = CvBridge()

        self.front_region = [110, 210]

        self.last_valid_time = None

        self.K = rospy.get_param("~K", 2.916)      # power-law K (fit from 2–3 samples)
        self.p = rospy.get_param("~p", 1.644)        # power-law exponent
        self.alpha = rospy.get_param("~alpha", 0.6) # Increase EMA smoothing factor for faster response
        self.z_min = rospy.get_param("~z_min", 0.1)
        self.z_max = rospy.get_param("~z_max", 1)

        self.hold_time = 1.2
        self.Z_ema = None

        self.lead_distance_pub = rospy.Publisher("perception/lead_car_distance", Float32, queue_size=1)

        rospy.Subscriber("robot_cam/image_raw", Image, self.image_callback, queue_size=1)

        rospy.loginfo("%s: [ACC] ACCLeadNode initialized.", self.robot_name)

    def _estimate_distance(self, avg_radius_px: float) -> float:
        # power-law distance: Z = (K / r)^ (1/p), clipped and EMA-smoothed
        r = max(float(avg_radius_px), 1e-6)
        Z = (self.K / r) ** (1.0 / max(self.p, 1e-3))
        Z = float(np.clip(Z, self.z_min, self.z_max))
        self.Z_ema = Z if self.Z_ema is None else (1.0 - self.alpha) * self.Z_ema + self.alpha * Z
        return self.Z_ema


    def image_callback(self, msg: Image):

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("Failed to convert image: %s", str(e))
            return
        
        ok, det = self.detector.detect(cv_image)

        if ok:
            r1,r2  = det['radii']
            center1_xy, center2_xy = det['centers']
            center_x = (center1_xy[0] + center2_xy[0]) / 2.0
            # print(f"Detected radii: r1={r1:.2f}, r2={r2:.2f}, center_x={center_x:.2f}")
            # print(f"Front region: {self.front_region}")
            if center_x > self.front_region[0] and center_x < self.front_region[1]:
                # Estimate distance based on radii
                avg_radius = (r1 + r2) / 2.0
                distance = self._estimate_distance(avg_radius)
                # print(f"Estimated distance: {distance:.2f} m, avg_radius: {avg_radius:.2f} px")
                self.last_valid_time = msg.header.stamp.to_sec()
            else:
                # out of front region
                if self.last_valid_time is None:
                    distance = self.z_max + 0.1  # No detection, set to max + margin
                elif msg.header.stamp.to_sec() - self.last_valid_time > self.hold_time:
                    distance = self.z_max + 0.1
                    self.last_valid_time = None
                else:
                    distance = self.Z_ema if self.Z_ema is not None else self.z_max + 0.1
        else:
            if self.last_valid_time is None:
                # no detection yet
                distance = self.z_max + 0.1  # No detection, set to max + margin
            elif msg.header.stamp.to_sec() - self.last_valid_time > self.hold_time:
                # lost detection for too long
                distance = self.z_max + 0.1
                self.last_valid_time = None
            else:
                # hold last valid distance
                distance = self.Z_ema if self.Z_ema is not None else self.z_max + 0.1

        self.lead_distance_pub.publish(Float32(data=distance))


if __name__ == "__main__":
    try:
        node = ACCLeadNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
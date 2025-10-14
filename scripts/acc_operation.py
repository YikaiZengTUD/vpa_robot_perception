#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, Range
from std_msgs.msg import Float32, Int32
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
        self.z_min = rospy.get_param("~z_min", 0.04)
        self.z_max = rospy.get_param("~z_max", 1.5)

        self.hold_time = 1.2
        self.Z_ema = None

        self.lead_distance_pub = rospy.Publisher("perception/lead_car_distance", Float32, queue_size=1)

        rospy.Subscriber("robot_cam/image_raw", Image, self.image_callback, queue_size=1)
        self.tof_range = 1.5
        self.tof_found_car = False
        self.tof_status = 0 # For start
        self.tof_status_sub = rospy.Subscriber("front_range_status", Int32, self.status_tof_callback, queue_size=1)

        self.tof_sub = rospy.Subscriber("front_range", Range, self.tof_callback, queue_size=1)
        rospy.loginfo("%s: [ACC] ACCLeadNode initialized.", self.robot_name)

    def status_tof_callback(self, msg: Int32):
        self.tof_status = msg.data

    def tof_callback(self, msg: Range):
        if self.tof_status == 9:
            self.tof_range = np.clip(msg.range, self.z_min, self.z_max)
        else:
            self.tof_range = self.z_max + 0.1  # invalid reading

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
        
        if self.tof_status == 9 and self.tof_range < 1:
            # the tof detect something
            ok, _ = self.detector.detect(cv_image)
            if ok: # we check if this is a car
                self.tof_found_car = True
                distance = float(np.clip(self.tof_range, self.z_min, self.z_max))
                self.lead_distance_pub.publish(Float32(data=distance))
                self.last_valid_time = msg.header.stamp.to_sec()
                return
            else: # not a car, ignore tof
                self.tof_found_car = False
                distance = self.z_max + 0.1
                self.lead_distance_pub.publish(Float32(data=distance))
                return
            
        # not using camera for distace estimation

        # if we have found a car using tof before, we will keep trusting tof until it goes away

        # if self.tof_found_car:
        #     # we must need tof to make sure the car is gone, as the image check may sometimes fail
        #     if self.tof_range > 1.3:
        #         self.tof_found_car = False

        #     distance = self.tof_range # trust tof when we have a valid reading
        #     self.lead_distance_pub.publish(Float32(data=distance))
        #     return

        # ok, det = self.detector.detect(cv_image)

        # if ok:
        #     r1,r2  = det['radii']
        #     center1_xy, center2_xy = det['centers']
        #     center_x = (center1_xy[0] + center2_xy[0]) / 2.0
        #     # print(f"Detected radii: r1={r1:.2f}, r2={r2:.2f}, center_x={center_x:.2f}")
        #     # print(f"Front region: {self.front_region}")
        #     if center_x > self.front_region[0] and center_x < self.front_region[1]:
        #         # Estimate distance based on radii
        #         avg_radius = (r1 + r2) / 2.0
        #         # first try to use ToF if available
        #         if self.tof_range < 1.5:
        #             # we may need to check
        #             distance = float(np.clip(self.tof_range, self.z_min, self.z_max))
        #             self.tof_found_car = True
        #         else:
        #             # somehow the front car is not in FOV of the tof sensor
        #             distance = self._estimate_distance(avg_radius)
        #         # print(f"Estimated distance: {distance:.2f} m, avg_radius: {avg_radius:.2f} px")
        #         self.last_valid_time = msg.header.stamp.to_sec()
        #     else:
        #         # out of front region
        #         if self.last_valid_time is None:
        #             distance = self.z_max + 0.1  # No detection, set to max + margin
        #         elif msg.header.stamp.to_sec() - self.last_valid_time > self.hold_time:
        #             distance = self.z_max + 0.1
        #             self.last_valid_time = None
        #         else:
        #             distance = self.Z_ema if self.Z_ema is not None else self.z_max + 0.1
        # else:
        #     if self.last_valid_time is None:
        #         # no detection yet
        #         distance = self.z_max + 0.1  # No detection, set to max + margin
        #     elif msg.header.stamp.to_sec() - self.last_valid_time > self.hold_time:
        #         # lost detection for too long
        #         distance = self.z_max + 0.1
        #         self.last_valid_time = None
        #     else:
        #         # hold last valid distance
        #         distance = self.Z_ema if self.Z_ema is not None else self.z_max + 0.1

        # self.lead_distance_pub.publish(Float32(data=distance))


if __name__ == "__main__":
    try:
        node = ACCLeadNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
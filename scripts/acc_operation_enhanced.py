#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, Range
from std_msgs.msg import Float32, Int32
import cv2
import numpy as np
from cv_bridge import CvBridge
import socket
# import your existing detector
from toolbox.front_car_detector_enhanced import FrontCarDetectorEnhanced

EPS = 1e-5

class ACCLeadNode:

    def __init__(self):
        rospy.init_node("acc_lead_node")

        self.robot_name = socket.gethostname()
        
        self.detector = FrontCarDetectorEnhanced()

        self.bridge = CvBridge()

        self.last_valid_time = None

        self.lead_distance_pub = rospy.Publisher("perception/lead_car_distance", Float32, queue_size=1)

        rospy.Subscriber("robot_cam/image_raw", Image, self.image_callback, queue_size=1)
        
        self.z_min = 0
        self.z_max = 1.5

        self.tof_range = 1.5
        self.tof_found_car = False
        self.tof_read_last = None
        self.tof_status = 0 # For start
        self.tof_status_sub = rospy.Subscriber("front_range_status", Int32, self.status_tof_callback, queue_size=1)

        self.tof_sub = rospy.Subscriber("front_range", Range, self.tof_callback, queue_size=1)

        rospy.loginfo("%s: [ACC] Front Car Detections initialized.", self.robot_name)

    def status_tof_callback(self, msg: Int32):
        self.tof_status = msg.data
        

    def tof_callback(self, msg: Range):
        if self.tof_status == 9:
            self.tof_range = np.clip(msg.range-0.04, self.z_min, self.z_max)  # small seeting because of mounting offset
        else:
            self.tof_range = self.z_max + 0.1  # invalid reading


    def image_callback(self, msg: Image):

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("Failed to convert image: %s", str(e))
            return
        
        if self.tof_status == 9 and self.tof_range < 1:
            # the tof detect something
            car_tag_list = self.detector.detect_front_car(cv_image)
            if len(car_tag_list) > 0: # we check if this is a car
                self.tof_found_car = True
                distance = float(np.clip(self.tof_range, self.z_min, self.z_max))
                self.lead_distance_pub.publish(Float32(data=distance))
                self.last_valid_time = msg.header.stamp.to_sec()    
                
                self.tof_read_last = self.tof_range
                return
            else: # visual detector did not find car
                if self.tof_read_last is None:
                    # we get no car at the beginning
                    self.tof_found_car = False
                    distance = self.z_max + 0.1
                    self.lead_distance_pub.publish(Float32(data=distance))
                    return
                if self.tof_range < 0.7:
                    # we still think there is a car
                    if self.tof_found_car:
                        distance = float(np.clip(self.tof_range, self.z_min, self.z_max))
                        self.lead_distance_pub.publish(Float32(data=distance))
                        self.last_valid_time = msg.header.stamp.to_sec()    
                        return
                    
                self.tof_found_car = False
                distance = self.z_max + 0.1
                self.lead_distance_pub.publish(Float32(data=distance))
                return
        else:
            # no valid tof reading
            # we assume no car detected by tof
            self.tof_found_car = False
            distance = self.z_max + 0.1
            self.lead_distance_pub.publish(Float32(data=distance))


if __name__ == "__main__":
    try:
        node = ACCLeadNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
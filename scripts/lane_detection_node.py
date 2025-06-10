#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray,Bool
from cv_bridge import CvBridge
import cv2
from toolbox.lane_detector import LaneDetector

import socket

class LaneDetectorNode:

    def __init__(self):
        rospy.init_node("lane_detector_node")

        self.robot_name = socket.gethostname()
        
        self.debug = rospy.get_param("~debug", False)

        self.publish_result = rospy.get_param("~publish_result", False)

        self.detector = LaneDetector(debug=self.debug, visual_debug=self.publish_result)
        if self.publish_result:
            self.res_pub    = rospy.Publisher("result_image", Image, queue_size=1)
        self.lane_center    = []
        self.bridge = CvBridge()
        self.center_pub     = rospy.Publisher("perception/lane_centers", Int32MultiArray, queue_size=1)
        self.near_stop_pub  = rospy.Publisher("perception/near_stop_line", Bool, queue_size=1)
        self.near_car_pub   = rospy.Publisher("perception/near_car", Bool, queue_size=1)
        
        rospy.Subscriber("robot_cam/image_raw", Image, self.image_callback, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(0.2), self.timer_callback)
        rospy.loginfo("%s: LaneDetectorNode initialized. Debug: %s",self.robot_name, self.debug)

    def image_callback(self, msg: Image):
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("Failed to convert image: %s", str(e))
            return

        if self.publish_result:
            result_frame,_,_,self.lane_center = self.detector.center_detect(cv_image)
        else:
            self.lane_center = self.detector.center_detect(cv_image)
        
        # Publish the centers if debug is enabled.
        # Other intermedia results are exposed but not published.
        if self.publish_result:
            try:
                result_image = self.bridge.cv2_to_imgmsg(result_frame, "bgr8")
                self.res_pub.publish(result_image)
            except Exception as e:
                rospy.logerr("Failed to publish result image: %s", str(e))

    def timer_callback(self, event):
        self.near_stop_pub.publish(Bool(data=self.detector.near_stop_line))
        self.near_car_pub.publish(Bool(data=self.detector.near_car))

        lane_centers_msg = Int32MultiArray()
        lane_centers_msg.data = self.lane_center
        self.center_pub.publish(lane_centers_msg)

if __name__ == "__main__":
    try:
        lane_detector_node = LaneDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("LaneDetectorNode terminated.")
    except Exception as e:
        rospy.logerr("An error occurred: %s", str(e))
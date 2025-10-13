#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Bool, Int32
from cv_bridge import CvBridge
import socket
from toolbox.tag_detector import AprilTagWrapper
import numpy as np

import math 

def yaw_zyx_from_R(R):

    return math.atan2(R[1,0], R[0,0])  # radians

class AprilTagDetectorNode:
    def __init__(self):
        rospy.init_node("tag_detector_node")

        self.robot_name = socket.gethostname()
        self.debug      = rospy.get_param("~debug", False)
        self.use_pose   = rospy.get_param("~use_pose", True)
        self.tag_size   = rospy.get_param("~tag_size", 0.06)
        self.bridge     = CvBridge()

        self.tag_x = 0.12
        self.tag_y = 0.0
        self.tag_id_pub = rospy.Publisher("perception/detected_tag_id", Int32, queue_size=1)
        # this default for DB19 robot, should allowed to be overridden by launch file
        self.base_to_camera_default = np.array([
            [-0.25881905,  0.0,         0.96592583,  0.0585],
            [-0.96592583,  0.0,        -0.25881905,  0.0   ],
            [ 0.0,        -1.0,         0.0,         0.0742],
            [ 0.0,         0.0,         0.0,         1.0   ]
        ])
        self.detector = AprilTagWrapper(debug=self.debug, tag_size=self.tag_size)
        rospy.Subscriber("perception/near_stop_line", Bool, self.stop_line_cb, queue_size=1)

        self.near_stop_line = False
        self.near_stop_line_last = False
        self.pose_pub = rospy.Publisher("start_pose", Pose2D, queue_size=1)
        self.image_msg = None
        rospy.loginfo("%s: AprilTagDetectorNode started. Debug=%s Pose=%s", self.robot_name, self.debug, self.use_pose)


    def stop_line_cb(self, msg):
        self.near_stop_line_last = self.near_stop_line
        self.near_stop_line = msg.data
        
        if self.near_stop_line and not self.near_stop_line_last:
            # When we are near the stop line, we want to process the image
            # but not repeat the processing if we are still near the stop line
            rospy.loginfo("%s: [TAG DETECTOR INFO] Near stop line: %s", self.robot_name, self.near_stop_line)
            is_success = False
            time_out_count = 5
            try:
                while not is_success and time_out_count > 0:
                    image_msg = rospy.wait_for_message("robot_cam/image_raw", Image, timeout=2.0)
                    is_success = self.image_callback(image_msg)
                    rospy.sleep(0.1)
                    time_out_count -= 1
                if not is_success:
                    rospy.logwarn("%s: [TAG DETECTOR INFO] Failed to detect tag after multiple attempts.", self.robot_name)
                    self.near_stop_line_last = False  # allow retry next time, assume we just not near enough
            except rospy.ROSException as e:
                rospy.logerr("%s: Failed to receive image: %s", self.robot_name, str(e))
        else:
            return

    def image_callback(self, msg):

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("[%s] Image conversion failed: %s", self.robot_name, str(e))
            return False

        detections, _ = self.detector.detect(frame)

        tag_ids = [d['id'] for d in detections]

        if self.debug:
            print("Detected tags:", tag_ids)
        
        det = detections[0] if detections else None
        if det is not None and 'pose_R' in det and 'pose_t' in det and det['pose_R'] is not None and det['pose_t'] is not None:
            self.tag_id_pub.publish(det['id'])

            T_base_to_camera = self.base_to_camera_default

            pose_R = det['pose_R']
            pose_t = det['pose_t']

            R_tc = pose_R
            R_ct = R_tc.T

            t_tc = pose_t.reshape(3)   

            t_ct = -R_ct @ t_tc                       # camera origin in TAG frame
            T_camera_to_tag = np.eye(4, dtype=float)
            T_camera_to_tag[:3, :3] = R_ct
            T_camera_to_tag[:3, 3]  = t_ct


            R_bc = T_base_to_camera[:3, :3]
            t_bc = T_base_to_camera[:3, 3]

            R_bt = R_bc @ R_ct                        # final rotation base->tag
            p_b  = t_bc + R_bc @ t_tc                 # final translation base->tag (uses tag-in-CAMERA)

            T_base_to_tag = np.eye(4, dtype=float)
            T_base_to_tag[:3, :3] = R_bt
            T_base_to_tag[:3, 3]  = p_b
            T_tag_to_base = np.linalg.inv(T_base_to_tag) 

            R_bt = T_base_to_tag[:3, :3]
            R_bc = T_base_to_camera[:3, :3]

            R_ct_from_base = R_bc.T @ R_bt

            yaw_rad = yaw_zyx_from_R(R_ct_from_base)

            x = -float(T_tag_to_base[0,3])
            y = -float(T_tag_to_base[1,3])

            pose_msg = Pose2D()
            pose_msg.x = self.tag_x - y
            pose_msg.y = self.tag_y - x
            pose_msg.theta = -yaw_rad
            self.pose_pub.publish(pose_msg)
            rospy.loginfo("%s: [TAG DETECT INFO] Detected tag %d, start pose sent as (x=%.2f, y=%.2f, theta=%.2f rad)", self.robot_name, det['id'], pose_msg.x, pose_msg.y, pose_msg.theta)
            return True

        else:
            rospy.logwarn("%s: [TAG DETECT INFO] No valid tag detected near stop line.", self.robot_name)
            return False

            # Save the current frame as an image for debugging
            # save_path = f"/tmp/tag_detection_{rospy.Time.now().to_nsec()}.jpg"
            # cv2.imwrite(save_path, frame)
            # rospy.loginfo("[%s] Saved tag detection image to %s", self.robot_name, save_path)
            # rospy.loginfo("[%s] Detected tag %d at position: (%.2f, %.2f, %.2f)", 
            #               self.robot_name, det['id'], 
            #               T_base_tag[0, 3], T_base_tag[1, 3], T_base_tag[2, 3])


if __name__ == "__main__":
    try:
        AprilTagDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

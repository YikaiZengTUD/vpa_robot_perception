#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import socket
from toolbox.tag_detector import AprilTagWrapper
import tf2_ros
import numpy as np
import geometry_msgs.msg
class AprilTagDetectorNode:
    def __init__(self):
        rospy.init_node("tag_detector_node")

        self.robot_name = socket.gethostname()
        self.debug      = rospy.get_param("~debug", False)
        self.use_pose   = rospy.get_param("~use_pose", True)
        self.tag_size   = rospy.get_param("~tag_size", 0.06)
        self.bridge     = CvBridge()

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        # this default for DB19 robot, should allowed to be overridden by launch file
        self.base_to_camera_default = np.array([
            [0, -0.258819045,  0.965925826,  0.0585],
            [1,  0,            0.0,          0.0   ],
            [0, -0.965925826, -0.258819045,  0.0742],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.detector = AprilTagWrapper(debug=self.debug, tag_size=self.tag_size)
        rospy.Subscriber("perception/near_stop_line", Bool, self.stop_line_cb, queue_size=1)

        self.near_stop_line = False
        self.near_stop_line_last = False
        self.image_msg = None
        rospy.loginfo("%s: AprilTagDetectorNode started. Debug=%s Pose=%s", self.robot_name, self.debug, self.use_pose)


    def stop_line_cb(self, msg):
        self.near_stop_line_last = self.near_stop_line
        self.near_stop_line = msg.data
        
        if self.near_stop_line and not self.near_stop_line_last:
            # When we are near the stop line, we want to process the image
            # but not repeat the processing if we are still near the stop line
            rospy.loginfo("%s: Near stop line: %s", self.robot_name, self.near_stop_line)
            try:
                image_msg = rospy.wait_for_message("robot_cam/image_raw", Image, timeout=2.0)
                self.image_callback(image_msg)
            except rospy.ROSException as e:
                rospy.logerr("%s: Failed to receive image: %s", self.robot_name, str(e))
        else:
            return

    def image_callback(self, msg):

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("[%s] Image conversion failed: %s", self.robot_name, str(e))
            return

        detections, _ = self.detector.detect(frame)

        tag_ids = [d['id'] for d in detections]

        if self.debug:
            print("Detected tags:", tag_ids)
        
        det = detections[0] if detections else None
        if det is not None and 'pose_R' in det and 'pose_t' in det and det['pose_R'] is not None and det['pose_t'] is not None:
            T_cam_tag = np.eye(4)
            T_cam_tag[:3, :3] = det['pose_R']
            T_cam_tag[:3, 3] = det['pose_t'].flatten()
            T_base_tag = self.base_to_camera_default @ T_cam_tag

            trans = geometry_msgs.msg.TransformStamped()
            trans.header.stamp = rospy.Time.now()
            trans.header.frame_id = "base_link"
            trans.child_frame_id = f"tag_{det['id']}"
            trans.transform.translation.x = T_base_tag[0, 3]
            trans.transform.translation.y = T_base_tag[1, 3]
            trans.transform.translation.z = T_base_tag[2, 3]

            quat = tf2_ros.transformations.quaternion_from_matrix(T_base_tag)
            trans.transform.rotation.x = quat[0]
            trans.transform.rotation.y = quat[1]
            trans.transform.rotation.z = quat[2]
            trans.transform.rotation.w = quat[3]

            self.tf_broadcaster.sendTransform(trans)
        else:
            rospy.logwarn("[%s] No valid tag detected near stop line.", self.robot_name)


            if self.debug:
                rospy.loginfo("[%s] Detected tag %d at position: (%.2f, %.2f, %.2f)", 
                              self.robot_name, det['id'], 
                              T_base_tag[0, 3], T_base_tag[1, 3], T_base_tag[2, 3])
    

if __name__ == "__main__":
    try:
        AprilTagDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

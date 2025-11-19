import math
import platform
import cv2
import numpy as np
import os

if platform.system() == 'Windows':
    import pupil_apriltags
    from pupil_apriltags import Detector as ATDetector
    # Set DLL path for Windows
    package_dir = os.path.dirname(pupil_apriltags.__file__)
    parent_dir = os.path.dirname(package_dir)
    dll_dir = os.path.join(parent_dir, "pupil_apriltags.libs")
    os.add_dll_directory(dll_dir)

else:
    from dt_apriltags import Detector as ATDetector
    # ideally it should stay same libary, but pupil_apriltags is not available for Ubuntu 20.04 for its native numpy version. to prevent break dependency,
    # we use dt_apriltags instead, which is native to Ubuntu 20.04 and has similar API
    # we will be able to swicth to pupil_apriltags on later version of hardware


class AprilTagWrapper:
    def __init__(self, tag_family='tag36h11', debug=False, tag_size=0.06):
        self.debug    = debug
        if self.debug:
            self.debug_image = None
            self.debug_gray = None
        self.detector = ATDetector(families=tag_family)
        self.use_pose = True
        self.tag_size = tag_size
        self.camera_params = [305.5718893575089 / 2, 308.8338858195428 / 2, 303.0797142544728 / 2, 231.8845403702499 / 2]
        # self.camera_params = [305.5718893575089, 308.8338858195428, 303.0797142544728, 231.8845403702499]

    def set_camera_params(self, camera_params):
        """
        Set camera parameters for pose estimation.
        :param camera_params: List of camera parameters [fx, fy, cx, cy]
        """
        if len(camera_params) != 4:
            raise ValueError("Camera parameters must be a list of four values: [fx, fy, cx, cy]")
        self.camera_params = camera_params
    
    def detect(self, frame_bgr, valid_tag_lowbound=300, valid_tag_upbound=400):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # gray = cv2.equalizeHist(gray)

        if self.use_pose:
            results = self.detector.detect(
                gray,
                estimate_tag_pose=self.use_pose,
                camera_params=self.camera_params,
                tag_size=self.tag_size
            )
        else:
            results = self.detector.detect(gray)

        detections = []
        debug_image = frame_bgr.copy()

        
        for tag in results:
            if tag.tag_id > valid_tag_upbound or tag.tag_id < valid_tag_lowbound:
                continue  # Skip tags with id < 300, MiniCCAM lab settings
            det = {
            'id': tag.tag_id,
            'center': tag.center,
            'corners': tag.corners
            }
            if self.use_pose:
                det['pose_R'] = getattr(tag, 'pose_R', None)
                det['pose_t'] = getattr(tag, 'pose_t', None)

            detections.append(det)
            
            if len(results) == 0:
                if self.debug:
                    self.debug_gray = gray
                    cv2.putText(debug_image, "No tags detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    self.debug_image = debug_image
                return detections, debug_image if self.debug else None
                    
            if self.debug:
                pts = np.int32(tag.corners)
                cx, cy = map(int, tag.center)
                
                cv2.polylines(debug_image, [pts], True, (0, 255, 255), 2)  # Yellow outline for tag
  
                # Draw tag ID
                cv2.putText(debug_image, str(tag.tag_id), (cx + 5, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                self.debug_image = debug_image
                if self.use_pose and tag.pose_t is not None and tag.pose_R is not None:
                    # Draw 3D coordinate axes from pose
                    axis_length = self.tag_size / 2
                    fx, fy, cx_, cy_ = self.camera_params

                    axis_3d = np.float32([
                    [0, 0, 0],
                    [axis_length, 0, 0],
                    [0, axis_length, 0],
                    [0, 0, -axis_length]
                    ])

                    rvec, _ = cv2.Rodrigues(tag.pose_R)
                    tvec = tag.pose_t

                    axis_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, 
                                np.array([[fx, 0, cx_], [0, fy, cy_], [0, 0, 1]]), 
                                np.zeros(5))
                    axis_2d = axis_2d.reshape(-1, 2).astype(int)

                    cv2.line(debug_image, tuple(axis_2d[0]), tuple(axis_2d[1]), (0, 0, 255), 2)  # X - red
                    cv2.line(debug_image, tuple(axis_2d[0]), tuple(axis_2d[2]), (0, 255, 0), 2)  # Y - green
                    cv2.line(debug_image, tuple(axis_2d[0]), tuple(axis_2d[3]), (255, 0, 0), 2)  # Z - blue
                    # axis
                    # Red = X Green = Y Blue = Z

                    # if  len(detections) > 0:


        
        return detections, debug_image if self.debug else None
    def visualize(self):
        from matplotlib import pyplot as plt

        if self.debug_image is None:
            raise ValueError("Debug image is not initialized. Ensure detect() is called before visualize().")
        if self.debug_gray is None:
            self.debug_gray = cv2.cvtColor(self.debug_image, cv2.COLOR_BGR2GRAY)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(cv2.cvtColor(self.debug_image, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Debug Image")
        axs[0].axis("off")

        axs[1].imshow(self.debug_gray, cmap='gray')
        axs[1].set_title("Grayscale Image")
        axs[1].axis("off")

        plt.tight_layout()
        plt.show()

# === Debug function for static image testing ===
def test_on_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    detector = AprilTagWrapper(debug=True)
    det = detector.detect(image)
    if len(det[0]) == 0:
        print("No tags detected")
        return None
    detector.visualize()
    tag_det = det[0]
    return tag_det


if __name__ == "__main__":

    test_tag = test_on_image("test/test_img/tagacc/image02.png")
    if test_tag is None:
        exit(0)

    def assert_proper(T):
        R = T[:3, :3]
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6), "Rotation det != +1"
        assert np.allclose(np.cross(R[:,0], R[:,1]), R[:,2], atol=1e-6), "Not right-handed"


    def R_base_to_camera_optical(lean_deg: float) -> np.ndarray:
        a = math.radians(lean_deg)
        # base -> camera_link (pitch about BASE Y)
        Ry = np.array([
            [ math.cos(a), 0, math.sin(a)],
            [ 0,           1, 0          ],
            [-math.sin(a), 0, math.cos(a)]
        ], dtype=float)

        # camera_link -> camera_optical (REP-103 canonical)
        R_cl2opt = np.array([
            [ 0,  0,  1],
            [-1,  0,  0],
            [ 0, -1,  0],
        ], dtype=float)

        # base -> camera_optical
        return R_cl2opt @ Ry
    
    

    T_base_to_camera = np.eye(4)
    T_base_to_camera[:3,:3] = R_base_to_camera_optical(15.0)  # +15° forward lean
    T_base_to_camera[:3, 3] = [0.0585, 0.0, 0.0742]   # camera position in BASE coords
    print("T_base_to_camera:\n", T_base_to_camera)
    pose_R = test_tag[0]['pose_R']
    pose_t = test_tag[0]['pose_t']

# --- begin drop-in fix ---


    # 1) Rotations
    R_tc = pose_R
    R_ct = R_tc.T                             # camera -> tag rotation

    # 2) Translations
    t_tc = pose_t.reshape(3)                  # tag in CAMERA frame (z_cam > 0 if in front)

    # Optional sanity: camera-frame depth
    # print("camera-frame t_tc (m):", t_tc)

    # Reference camera->tag homogeneous (useful to inspect but not used for base translation)
    t_ct = -R_ct @ t_tc                       # camera origin in TAG frame
    T_camera_to_tag = np.eye(4, dtype=float)
    T_camera_to_tag[:3, :3] = R_ct
    T_camera_to_tag[:3, 3]  = t_ct
    # print("T_camera_to_tag:\n", T_camera_to_tag)

    # 3) Compose BASE -> TAG
    R_bc = T_base_to_camera[:3, :3]
    t_bc = T_base_to_camera[:3, 3]

    R_bt = R_bc @ R_ct                        # final rotation base->tag
    p_b  = t_bc + R_bc @ t_tc                 # final translation base->tag (uses tag-in-CAMERA)

    T_base_to_tag = np.eye(4, dtype=float)
    T_base_to_tag[:3, :3] = R_bt
    T_base_to_tag[:3, 3]  = p_b
    T_tag_to_base = np.linalg.inv(T_base_to_tag)    
    print("T_tag_to_base:\n", T_tag_to_base)

    def yaw_zyx_from_R(R):

        return math.atan2(R[1,0], R[0,0])  # radians

    R_bt = T_base_to_tag[:3, :3]
    R_bc = T_base_to_camera[:3, :3]

    R_ct_from_base = R_bc.T @ R_bt

    yaw_rad = yaw_zyx_from_R(R_ct_from_base)
    yaw_deg = math.degrees(yaw_rad)
    x = -float(T_tag_to_base[0,3])
    y = -float(T_tag_to_base[1,3])
    print(f"Tag position (x,y) in BASE frame (m): {x,y}")
    print(f"Camera/Base yaw in TAG coords (from T_base_to_tag): {yaw_deg:.2f}°")
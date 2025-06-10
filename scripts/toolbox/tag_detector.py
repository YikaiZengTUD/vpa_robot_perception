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
        self.detector = ATDetector(families=tag_family)
        self.use_pose = True
        self.tag_size = tag_size
        self.camera_params = [305.5718893575089, 308.8338858195428, 303.0797142544728, 231.8845403702499]  # fx, fy, cx, cy
    
    def set_camera_params(self, camera_params):
        """
        Set camera parameters for pose estimation.
        :param camera_params: List of camera parameters [fx, fy, cx, cy]
        """
        if len(camera_params) != 4:
            raise ValueError("Camera parameters must be a list of four values: [fx, fy, cx, cy]")
        self.camera_params = camera_params
    
    def detect(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

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
            if tag.tag_id < 300:
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
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(self.debug_image, cv2.COLOR_BGR2RGB))
        plt.title("AprilTag Detections")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
# === Debug function for static image testing ===
def test_on_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    detector = AprilTagWrapper(debug=True)
    detector.detect(image)
    detector.visualize()

if __name__ == "__main__":
    test_on_image("test/test_img/image06.png")
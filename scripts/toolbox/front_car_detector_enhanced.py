import cv2
import numpy as np
from toolbox.tag_detector import AprilTagWrapper


class FrontCarDetectorEnhanced:

    def __init__(self,debug=False):
        
        self.apriltag_detector = AprilTagWrapper(tag_family='tag36h11', debug=False, tag_size=0.08)
        # Camera parameters: fx, fy, cx, cy
        
    def increase_brightness(self, frame, value=30):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        v = cv2.add(v, value)
        v = np.clip(v, 0, 255)

        final_hsv = cv2.merge((h, s, v))
        # we should alos convert close to black to totally black
        black_threshold = 30
        mask = cv2.inRange(frame, (0, 0, 0), (black_threshold, black_threshold, black_threshold))
        final_hsv[mask > 0] = 0  # Set these pixels to black in HSV
        bright_frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return bright_frame
    
    def _show_debug_frame(self, frame, window_name="Debug Frame"):
        import matplotlib.pyplot as plt
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title(window_name)
        plt.axis('off')  # Hide axes
        plt.show()

    def detect_front_car(self, frame_bgr):
        bright_frame = self.increase_brightness(frame_bgr, value=50)
        # self._show_debug_frame(bright_frame, window_name="Brightened Frame")
        detections = self.apriltag_detector.detect(bright_frame, valid_tag_lowbound=0, valid_tag_upbound=100)
        front_car_tags = []
        if detections is None or len(detections) == 0:
            # print("No tags detected")
            return []
        else:
            for det in detections:
                if det is None or len(det) == 0:
                    continue
                
                tag_id = det[0]['id']
                if 0 <= tag_id < 100:
                    front_car_tags.append(tag_id)
            return front_car_tags

# === Debug function for static image testing ===
def test_on_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    detector = FrontCarDetectorEnhanced()
    front_tags = detector.detect_front_car(image)
    for tag_id in front_tags:
        print(f"Detected front car tag ID: {tag_id}")
    print("Debugging completed.")


    # debug_image = detector.increase_brightness(image, value=50)
    # detector._show_debug_frame(debug_image, window_name="Brightened Image")
    # det = detector.detect_front_car(debug_image)
  
    # det = detector.detect_front_car(image)
    # if len(det) == 0:
    #     print("No tags detected")
    #     return None
    # detector.apriltag_detector.visualize()
    # tag_det = det[0]
    return None

if __name__ == "__main__":
    image_path = "test/test_img/tagacc/image01.png"
    tag_detection = test_on_image(image_path)

    if tag_detection:
        print(f"Detected tag ID: {tag_detection['id']}")


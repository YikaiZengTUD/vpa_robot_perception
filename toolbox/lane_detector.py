import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

class LaneDetector:
    def __init__(self, debug=False, visual_debug=False):
        self.debug = debug

        self.publish_visualization = visual_debug
        self.scanline_y     = [0.6, 0.7, 0.8]

        self.lower_yellow   = (20, 80, 200)
        self.upper_yellow   = (40, 230, 255)
        
        self.lower_white    = (0, 0, 200)
        self.upper_white    = (150, 30, 255)

        self.lower_red1 = (0, 100, 100)
        self.upper_red1 = (10, 255, 255)

        self.lower_red2 = (160, 100, 100)
        self.upper_red2 = (179, 255, 255)

        self.near_stop_line = False
        self.near_car       = False

    def stop_detect(self, hsv_frame):
        # step 1: remove the upper part of the image
        mask_red1 = cv2.inRange(hsv_frame, self.lower_red1, self.upper_red1)
        mask_red2 = cv2.inRange(hsv_frame, self.lower_red2, self.upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        num_red_pixels = np.count_nonzero(mask_red)
        if self.debug:
            print(f"Number of red pixels detected: {num_red_pixels}")
        if num_red_pixels > 2000:
            ys, xs = np.where(mask_red > 0)
            if len(ys) > 0:
                avg_y = int(np.mean(ys))
                if self.debug:
                    print(f"Average y of red points: {avg_y}")
                if avg_y > 160:
                    if self.debug:
                        return mask_red, True
                    else:
                        return True

        if self.debug:
            return None, False
        else:
            return False

    def center_detect(self, frame):
        # step 1: remove the upper part of the image
        height, width = frame.shape[:2]
        roi = np.zeros_like(frame)
        roi[int(height * 0.4):, :] = frame[int(height * 0.4):, :]
        frame = roi.copy()
        # step 2: convert to HSV and create masks for yellow and white lanes
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # if self.debug:
        #     interested_point = (129,213)
        #     print(f"HSV value at {interested_point}: {hsv[interested_point[1], interested_point[0]]}")
        mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask_white = cv2.inRange(hsv, self.lower_white, self.upper_white)
        lane_mask = cv2.bitwise_or(mask_yellow, mask_white)
        if self.debug:
            mask_red, self.near_stop_line = self.stop_detect(hsv)
            if mask_red is None:
                mask_red = np.zeros_like(lane_mask)
        else:
            self.near_stop_line = self.stop_detect(hsv)

        contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h != 0 else 0
            mask_roi = lane_mask[y:y+h, x:x+w]
            # hollow_score = 1.0 - (np.count_nonzero(mask_roi) / (w * h)) if w * h > 0 else 0

            if area < 200:
                continue

            if self.debug:
                # print(f"Contour: Area={area}, Aspect Ratio={aspect_ratio:.2f}")
                label = f"A={int(area)} AR={aspect_ratio:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            
            lane_mask[y:y+h, x:x+w] = cv2.bitwise_or(lane_mask[y:y+h, x:x+w], mask_roi)

        centers = []
        for rel_y in self.scanline_y:
            y = int(height * rel_y)
            scan = lane_mask[y, :]
            indices = np.where(scan > 0)[0]
            if self.debug:
                print(f"Scanline at {y}: Found {len(indices)} points")
                print(f"Indices: {indices}")
            if len(indices) > 0:
                clusters = self.cluster_indices(indices)
                clusters = [c for c in clusters if len(c) >= 5]  # filter out small noise blobs

                if len(clusters) == 1:
                    x = int(np.mean(clusters[0]))
                    lane_half_width = self.lane_width_at(rel_y) // 2
                    image_center = width // 2

                    if x < image_center:
                        center = x + lane_half_width
                    else:
                        center = x - lane_half_width
                elif len(clusters) >= 2:
                    left = clusters[0][0]
                    right = clusters[-1][-1]
                    center = (left + right) // 2

                cv2.circle(frame, (center, y), 4, (0, 255, 0), -1)
            cv2.line(frame, (0, y), (width, y), (255, 0, 0), 1)
        if self.debug:
            image_center = width // 2
            cv2.line(frame, (image_center, 0), (image_center, height), (0, 0, 255), 1)
        if self.debug:
            return frame,lane_mask, mask_red, centers
        else:
            return centers

    def visualize(self, frame, mask, edges):
        if not self.debug:
            return
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Lane Detection Result")
        axs[1].imshow(mask, cmap='gray')
        axs[1].set_title("Lane Mask")
        axs[2].imshow(edges, cmap='gray')
        axs[2].set_title("Red Line Mask")
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    def lane_width_at(self, scan_y):
        scan_pos = np.array([0.6, 0.7, 0.8])
        width_px = np.array([170, 220, 270])
        return int(np.interp(scan_y, scan_pos, width_px))
    
    def cluster_indices(self, indices, gap=20):
        clusters = []
        cluster = [indices[0]]
        for i in range(1, len(indices)):
            if indices[i] - indices[i-1] <= gap:
                cluster.append(indices[i])
            else:
                clusters.append(cluster)
                cluster = [indices[i]]
        clusters.append(cluster)
        return clusters


if __name__ == "__main__":
    IMAGE_PATH = "test/test_img/image06.png"
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        raise FileNotFoundError(f"Cannot load image: {IMAGE_PATH}")

    detector = LaneDetector(debug=True)
    frame_out, mask_out, edges_out, centers = detector.center_detect(frame)
    detector.visualize(frame_out, mask_out, edges_out)

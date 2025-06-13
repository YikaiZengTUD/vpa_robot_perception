import cv2
import numpy as np
import os

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
                    if self.publish_visualization:
                        return mask_red, True
                    else:
                        return True

        if self.publish_visualization:
            return mask_red, False
        else:
            return False

    def center_detect(self, frame):
        # step 1: remove the upper part of the image
        height, width = frame.shape[:2]
        roi = np.zeros_like(frame)
        cutting_height = int(height * 0.5)
        roi[cutting_height:, :] = frame[cutting_height:, :]
        frame = roi.copy()
        # step 2: convert to HSV and create masks for yellow and white lanes
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # if self.debug:
        #     interested_point = (129,213)
        #     print(f"HSV value at {interested_point}: {hsv[interested_point[1], interested_point[0]]}")
        mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask_white = cv2.inRange(hsv, self.lower_white, self.upper_white)
        lane_mask = cv2.bitwise_or(mask_yellow, mask_white)
        if self.publish_visualization:
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

            if area < 280:
                continue

            if self.debug:
                # print(f"Contour: Area={area}, Aspect Ratio={aspect_ratio:.2f}")
                label = f"A={int(area)} AR={aspect_ratio:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            
            lane_mask[y:y+h, x:x+w] = cv2.bitwise_or(lane_mask[y:y+h, x:x+w], mask_roi)

        centers = [None, None, None]  # Initialize centers for each scanline
        # check yellow

        for i,y in enumerate(self.scanline_y):
            center = None
            scan_y = int(height * y)
            yellow_indices  = np.where(mask_yellow[scan_y, :] > 0)[0]
            white_indices   = np.where(mask_white[scan_y, :] > 0)[0]
            indices         = np.where(lane_mask[scan_y, :] > 0)[0]
            
            if len(yellow_indices) > 0:
                # if we have yellow points, we can solely rely on them
                yellow_clusters = self.cluster_indices(yellow_indices)
                if self.debug:
                    for cluster in yellow_clusters:
                        print(f"Yellow cluster at y={y}: {cluster}")
                yellow_clusters = [c for c in yellow_clusters if len(c) >= 4]
                if len(yellow_clusters) == 0:
                    continue
                left_cluster    = yellow_clusters[0]
                left_point      = np.mean(left_cluster)

                # Determine right_point using white_indices if available, else estimate
                right_point = left_point + self.lane_width_at(y)
                if len(white_indices) > 0:
                    right_clusters = self.cluster_indices(white_indices)
                    if right_clusters:
                        right_cluster = right_clusters[-1]
                        if len(right_cluster) > 10:
                            right_point = np.mean(right_cluster)

                if self.debug:
                    print(f"Yellow clusters at y={y}: {left_point}, Right point: {right_point}")
                
                center = int((left_point + right_point) / 2)
                centers[i] = center
            else:
                # no yellow points, we need to find the center based on the lane mask
                if len(indices) > 0:
                    clusters = self.cluster_indices(indices)
                    clusters = [c for c in clusters if len(c) >= 10]
                    if len(clusters) >= 2:
                        # if we have two clusters, we can find the center
                        left_cluster    = clusters[0]
                        right_cluster   = clusters[-1]
                        left_point      = np.mean(left_cluster)
                        right_point     = np.mean(right_cluster)
                        if self.debug:
                            print(f"Left cluster at y={scan_y}: {left_point}, Right cluster: {right_point}")
                        center = int((left_point + right_point) / 2)
                        centers[i] = center
                    else:
                        if self.debug:
                            print(f"Not enough clusters found at y={y}, clusters found: {len(clusters)}")
                        continue
                else:
                    if self.debug:
                        print(f"No sufficient points detected at y={y}")

            
            if self.publish_visualization:
                if center is not None:
                    cv2.circle(frame, (center, scan_y), 4, (0, 255, 0), -1)
                    cv2.circle(frame, (int(left_point), scan_y), 4, (255, 255, 0), -1)
                    cv2.circle(frame, (int(right_point),scan_y), 4, (0, 0, 255), -1)
                    cv2.line(frame, (0, scan_y), (width, scan_y), (255, 0, 0), 1)
                    image_center = width // 2
                    cv2.line(frame, (image_center, 0), (image_center, cutting_height), (100, 100, 255), 1)
        if self.debug:
            print("Centers at scanlines:", centers)
        if self.publish_visualization:
            return frame, lane_mask, mask_red, centers
        else:
            return centers

    def visualize(self, frame, mask, edges):
        import matplotlib.pyplot as plt
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
    IMAGE_PATH = "test/test_img/image10.png"
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        raise FileNotFoundError(f"Cannot load image: {IMAGE_PATH}")

    detector = LaneDetector(debug=True,visual_debug=True)
    frame_out, mask_out, edges_out, centers = detector.center_detect(frame)
    detector.visualize(frame_out, mask_out, edges_out)

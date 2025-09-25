import cv2
import numpy as np
import os

class LaneDetector:
    # Essentially it detects everything that is on the lane, including other cars, stop lines, etc.
    # The naming is a bit misleading.
    def __init__(self, debug=False, visual_debug=False):

        self.debug = debug

        self.publish_visualization = visual_debug

        self.lower_yellow   = (20, 70, 130)
        self.upper_yellow   = (50, 255, 255)
        
        self.lower_white    = (0, 0, 110)
        self.lower_white1   = (0, 0, 200)
        self.upper_white    = (150, 50, 255)

        self.lower_red1     = (0, 100, 100)
        self.upper_red1     = (10, 255, 255)

        self.lower_red2     = (160, 100, 100)
        self.upper_red2     = (179, 255, 255)

        self.near_stop_line  = False
        self.near_car        = False

        self.last_left_boundary  = 0
        self.last_right_boundary = 320

    def detect_stop_line(self, hsv_frame):
        
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
        
    def image_process(self,frame):
        # step 1: remove the upper part of the image
        height, width = frame.shape[:2]
        roi = np.zeros_like(frame)
        cutting_height = int(height * 0.25)
        roi[cutting_height:, :] = frame[cutting_height:, :]
        frame = roi.copy()

        # step 2: convert to HSV 
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # print(hsv[120,90])

        if self.publish_visualization:
            self.mask_red,self.near_stop_line = self.detect_stop_line(hsv)
            self.lane_mask, left_boundary, right_boundary, scan_y = self.detect_lane_boundaries(hsv)
        else:
            self.near_stop_line = self.detect_stop_line(hsv)
            left_boundary, right_boundary = self.detect_lane_boundaries(hsv)
        
        if left_boundary is not None:
            self.last_left_boundary = left_boundary
        if right_boundary is not None:
            self.last_right_boundary = right_boundary

        if left_boundary is None and right_boundary is not None:
            left_boundary = right_boundary - 80
        if right_boundary is None and left_boundary is not None:
            right_boundary = left_boundary + 80

        if self.publish_visualization:

            cv2.circle(frame, (int(self.last_left_boundary), scan_y), 4, (255, 255, 0), -1)
            cv2.circle(frame, (int(self.last_right_boundary), scan_y), 4, (0, 0, 255), -1)

            # self.visualize(frame, self.lane_mask, self.mask_red)
            self.result_frame = frame

        return [self.last_left_boundary, self.last_right_boundary]

        
    def detect_lane_boundaries(self, hsv_frame):
        # we have made changes on the layout of lanes
        # there should always be a yellow line on the left side of the lane, 
        # but it can be next to a whitle line, which is the right boundary of the other lane

        left_yellow_boundary = None
        right_white_boundary = None

        mask_yellow = cv2.inRange(hsv_frame, self.lower_yellow, self.upper_yellow)
        mask_white  = cv2.inRange(hsv_frame, self.lower_white, self.upper_white)

        num_of_white = np.count_nonzero(mask_white)

        if num_of_white > 15000:
            # this is too bright
            mask_white  = cv2.inRange(hsv_frame, self.lower_white1, self.upper_white)

        scan_height_based = int(hsv_frame.shape[0]/2) 
        for row_offset in range(5,45,10):
            scan_height = scan_height_based + row_offset
            row = mask_yellow[scan_height]
            yellow_indices = np.where(row > 0)[0]

            # we should be careful that there might be multiple yellow segments
            if len(yellow_indices) > 0:
                yellow_clusters = self._detect_clusters(yellow_indices)
                if len(yellow_clusters) > 0:
                    # choose the widest cluster
                    # get the cluster of largest pts set
                    left_yellow_cluster = yellow_clusters[0]
                    for cluster in yellow_clusters:
                        if len(cluster) > len(left_yellow_cluster):
                            if np.mean(cluster) < 180:
                                left_yellow_cluster = cluster
                    left_yellow_boundary = int(np.mean(left_yellow_cluster))
                    break
            else:
                continue
        
        if left_yellow_boundary is None:
            scan_height = scan_height_based
        # chances are that we did not find any yellow line, we return None
        white_nearby_search = [0,-10,10,-20,20]
        for row_offset in white_nearby_search:
            scan_height_white = scan_height + row_offset
            # we search near where we found the yellow line
            row = mask_white[scan_height_white]
            white_indices = np.where(row > 0)[0]
            if len(white_indices) > 0:
                white_clusters = self._detect_clusters(white_indices)
                if len(white_clusters) > 0:
                    # choose the leftest cluster that is at least 30 pixels away from the yellow line
                    if left_yellow_boundary is not None:
                        valid_white_clusters = [c for c in white_clusters if np.mean(c) - left_yellow_boundary > 40]
                    else:
                        valid_white_clusters = white_clusters
                    if len(valid_white_clusters) > 0:
                        right_white_cluster = valid_white_clusters[-1]
                        for cluster in valid_white_clusters:
                            if len(cluster) > len(right_white_cluster):
                                if np.mean(cluster) > 80:
                                    right_white_cluster = cluster
                        right_white_boundary = int(np.mean(right_white_cluster))
                        break

        if self.debug:
            print(f"Left boundary: {left_yellow_boundary}, Right boundary: {right_white_boundary}")
        if self.publish_visualization:
            combined_mask = mask_white # optinal which one
            return combined_mask, left_yellow_boundary, right_white_boundary, scan_height

        return left_yellow_boundary, right_white_boundary

    def _detect_clusters(self,row):
        # this is a 1D clusters of indexes like [ 1,2,3, 20,21,22, 50,51]
        # we need to break them into [[1,2,3], [20,21,22], [50,51]]
        clusters = []
        break_gap = 20
        current_cluster = [row[0]] if len(row) > 0 else []
        for i in range(1,len(row)):
            if row[i] > row[i-1] + break_gap:
                # break here
                clusters.append(current_cluster)
                current_cluster = [row[i]]
            else:
                current_cluster.append(row[i])
        clusters.append(current_cluster)
        # remove clusters too small, set a length threshold
        clusters = [c for c in clusters if len(c) >= 5]

        return clusters
    
    def visualize(self, frame, mask, edges):
        import matplotlib.pyplot as plt
        if not self.publish_visualization:
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

if __name__ == "__main__":
    IMAGE_PATH = "test/test_img/image43.png"
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        raise FileNotFoundError(f"Cannot load image: {IMAGE_PATH}")

    detector = LaneDetector(debug=True,visual_debug=True)
    detector.image_process(frame)
    detector.visualize(detector.result_frame, detector.lane_mask, detector.mask_red)
import cv2
import numpy as np
import os

class LaneDetector:
    # Essentially it detects everything that is on the lane, including other cars, stop lines, etc.
    # The naming is a bit misleading.
    def __init__(self, debug=False, visual_debug=False):

        self.debug = debug

        self.publish_visualization = visual_debug

        self.lower_yellow   = (20, 80, 200)
        self.upper_yellow   = (40, 230, 255)
        
        self.lower_white    = (0, 0, 200)
        self.upper_white    = (150, 30, 255)

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

        if self.publish_visualization:

            cv2.circle(frame, (int(self.last_left_boundary), scan_y), 4, (255, 255, 0), -1)
            cv2.circle(frame, (int(self.last_right_boundary), scan_y), 4, (0, 0, 255), -1)

            self.visualize(frame, self.lane_mask, self.mask_red)


        
    def detect_lane_boundaries(self, hsv_frame):
        # we have made changes on the layout of lanes
        # there should always be a yellow line on the left side of the lane, 
        # but it can be next to a whitle line, which is the right boundary of the other lane
        # therefore we need to try distinguish the two lines

        left_yellow_boundary = None
        right_white_boundary = None

        mask_yellow = cv2.inRange(hsv_frame, self.lower_yellow, self.upper_yellow)
        mask_white  = cv2.inRange(hsv_frame, self.lower_white, self.upper_white)

        scan_height_based = int(hsv_frame.shape[0]/2) + 10
        for row_offset in range(-10,20,5):
            scan_height = scan_height_based + row_offset
            row = mask_yellow[scan_height,:]
            yellow_indices = np.where(row > 0)[0]
            # we should be careful that there might be multiple yellow segments
            if len(yellow_indices) > 0:
                yellow_clusters = self._detect_clusters(yellow_indices)
                if len(yellow_clusters) > 0:
                    # choose the leftest cluster
                    left_yellow_cluster = yellow_clusters[0]
                    left_yellow_boundary = int(np.mean(left_yellow_cluster))

                    # then we find the right yellow one if there are more than 2 clusters
                    if len(yellow_clusters) > 1:
                        right_yellow_cluster = yellow_clusters[1]
                        right_yellow_boundary = int(np.mean(right_yellow_cluster))
                    else:
                        right_yellow_boundary = 360 # if there is no right yellow line, we set it to a large value
            else:
                continue
                    
        # we should only care the while line in btween the two yellow lines
        if left_yellow_boundary is None:
            # unable to find left yellow line
            left_yellow_boundary = 0
            right_yellow_boundary = 360

        row = mask_white[scan_height,left_yellow_boundary:right_yellow_boundary]
        white_indices = np.where(row > 0)[0]
        
        if len(white_indices) > 0:
            white_clusters = self._detect_clusters(white_indices)
            if len(white_clusters) == 1:
                # chose the one nearest to the right yellow line
                right_white_cluster = white_clusters[-1]
                right_white_boundary = int(np.mean(right_white_cluster)) + left_yellow_boundary
            elif len(white_clusters) > 1:
                right_white_cluster = white_clusters[-1]
                right_white_boundary = int(np.mean(right_white_cluster)) + left_yellow_boundary
                if left_yellow_boundary == 0:
                    # this is very likely a color reflection problem, we can then gusee
                    left_yellow_boundary = int(np.mean(white_clusters[0]))

        if self.debug:
            print(f"Left boundary: {left_yellow_boundary}, Right boundary: {right_white_boundary}")
        if self.publish_visualization:
            combined_mask = mask_yellow # optinal which one
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
    IMAGE_PATH = "test/test_img/image04.png"
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        raise FileNotFoundError(f"Cannot load image: {IMAGE_PATH}")

    detector = LaneDetector(debug=True,visual_debug=True)
    detector.image_process(frame)
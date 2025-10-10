import cv2
import numpy as np
from math import hypot, atan2, degrees
class FrontCarDetector:

    def __init__(self):
        
        self.car_pattern_hsv_lower1 = (50, 80, 25)   # allow dark-ish green
        self.car_pattern_hsv_lower2 = (50, 80, 10)
        self.car_pattern_hsv_lower3 = (50, 80, 40) 
        self.car_pattern_hsv_upper = (85, 255, 255) # cover lime → teal-ish green

        self.MIN_DOT_AREA      = 25          # px^2
        self.MIN_CIRCULARITY   = 0.65        # 1.0 = perfect circle
        self.MAX_RADIUS_RATIO  = 1.5         # size similarity

    def _green_mask(self,case,bgr):
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        if case == 1:
            mask = cv2.inRange(hsv, self.car_pattern_hsv_lower1, self.car_pattern_hsv_upper)
        elif case == 3:
            mask = cv2.inRange(hsv, self.car_pattern_hsv_lower3, self.car_pattern_hsv_upper)
        else:
            mask = cv2.inRange(hsv, self.car_pattern_hsv_lower2, self.car_pattern_hsv_upper)
        return cv2.medianBlur(mask, 5)       # keep holes; don't close/dilate

    def detect(self, image_opencv):
        """
        Args:
            image_opencv: BGR image as numpy array (H,W,3), dtype=uint8
        Returns:
            mask_car: binary mask (H,W), dtype=uint8, 0 or 255
        """
        mask_green = self._green_mask(1,image_opencv)
        ok,det = self.find_car_patterns(mask_green)
        if not ok:
            mask_green = self._green_mask(2,image_opencv)
            ok2,det = self.find_car_patterns(mask_green)
            if not ok2:
                ok3,det = self.find_car_patterns(self._green_mask(3,image_opencv))
                if not ok3:
                    return False, None
        return True, det
    
    def _center_and_vertical_diam(self, cnt):
        # returns (cx, cy, d_y)
        if len(cnt) >= 5:
            (cx,cy),(d1,d2),ang = cv2.fitEllipse(cnt)
            A, B = max(d1,d2)/2.0, min(d1,d2)/2.0
            th = np.deg2rad(ang)
            d_y = 2.0*np.sqrt((A*np.sin(th))**2 + (B*np.cos(th))**2)
            return float(cx), float(cy), float(d_y)
        else:
            (cx,cy), r = cv2.minEnclosingCircle(cnt)  # fallback
            return float(cx), float(cy), float(2.0*r)

    def find_car_patterns(self,masked_image):
        H, W = masked_image.shape[:2]
        cnts, hier = cv2.findContours(masked_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if hier is None or len(cnts) == 0: 
            return False, {}
        
        hier = hier[0]
        best = None

        
        for i, c in enumerate(cnts):
            # parent green blob = contour with no parent
            if hier[i][3] != -1:
                continue
            # collect child contours (holes)
            child = hier[i][2]
            holes = []
            while child != -1:
                cc = cnts[child]
                a  = cv2.contourArea(cc)
                if a >= self.MIN_DOT_AREA:
                    p = cv2.arcLength(cc, True) + 1e-6
                    circ = 4*np.pi*a/(p*p)
                    if circ >= self.MIN_CIRCULARITY:
                        (x,y), r = cv2.minEnclosingCircle(cc)
                        holes.append((x, y, r, a, circ))
                child = hier[child][0]  # next sibling

            if len(holes) < 2: 
                continue
            # take two largest by area
            holes.sort(key=lambda t: t[3], reverse=True)
            (x1,y1,r1,a1,c1), (x2,y2,r2,a2,c2) = holes[:2]

            # basic gates
            rratio = max(r1,r2) / max(1e-6, min(r1,r2))
            if rratio > self.MAX_RADIUS_RATIO:
                continue

            s   = hypot(x1-x2, y1-y2)
            ang = degrees(atan2(y2-y1, x2-x1))
            ang = ((ang + 90) % 180) - 90  # [-90,90]

            # simple score: prefer large, round, near image center, near horizontal
            xc   = 0.5*(x1+x2)
            cen_pen = abs(xc - W/2) / W
            score = (a1 + a2) * (c1 + c2) / (1.0 + 0.5*abs(ang)/30.0 + 2.0*cen_pen)

            if (best is None) or (score > best[0]):
                best = (score, (x1,y1,r1), (x2,y2,r2), s, ang, i)

        if best is None:
            return False, {}

        _, d1, d2, s, ang, parent_idx = best
        x1,y1,r1 = d1; x2,y2,r2 = d2
        x,y,w,h = cv2.boundingRect(cnts[parent_idx])
        return True, {
            "centers": ((float(x1),float(y1)), (float(x2),float(y2))),
            "radii":   (float(r1), float(r2)),
            "spacing_px": float(s),
            "angle_deg":  float(ang),
            "parent_bbox": (int(x),int(y),int(w),int(h)),
            "mask": masked_image
        }

    def _test_debug(self,image_opencv):
        mask_car = self._green_mask(1,image_opencv)
        # Create a binary mask where white represents the detected range

        import matplotlib.pyplot as plt

        plt.imshow(mask_car, cmap='gray')
        plt.title("Debug Mask")
        plt.axis('off')
        plt.show()

        overlay = image_opencv.copy()
        overlay[mask_car > 0] = (225,0,255)
        vis = cv2.addWeighted(image_opencv, 0.7, overlay, 0.3, 0)
        plt.figure(); 
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); 
        plt.axis('off'); plt.show()


def draw_detection(bgr, det):
    out = bgr.copy()
    (x1,y1),(x2,y2) = det["centers"]
    r1,r2 = det["radii"]
    cv2.circle(out, (int(x1),int(y1)), int(r1), (0,0,255), 2)
    cv2.circle(out, (int(x2),int(y2)), int(r2), (0,0,255), 2)
    cv2.line(out, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 2)
    x,y,w,h = det["parent_bbox"]
    cv2.rectangle(out, (x,y), (x+w,y+h), (0,255,0), 2)
    return out


if __name__ == "__main__":

    IMAGE_PATH = "test/test_img/acc/image51.png"
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        raise FileNotFoundError(f"Cannot load image: {IMAGE_PATH}")
    
    import matplotlib.pyplot as plt

    def show_bgr(img_bgr, title=""):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(6,4))
        plt.imshow(img_rgb)
        plt.title(title); plt.axis('off'); plt.show()

    def show_mask(mask, title=""):
        plt.figure(figsize=(6,4))
        plt.imshow(mask, cmap='gray', vmin=0, vmax=255)
        plt.title(title); plt.axis('off'); plt.show()

    detector = FrontCarDetector()
    found, det = detector.detect(frame)
    
    if found:
        vis = draw_detection(frame, det)
        show_bgr(vis, title="Detection")   
        rad1, rad2 = det["radii"]
        print(f"Detected car-like pattern with radii: {rad1:.1f}, {rad2:.1f}")
    else:
        print("No car-like pattern detected.")
        detector._test_debug(frame)
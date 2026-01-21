#!/usr/bin/env python3
import subprocess, shlex, time
from collections import deque

import rospy
from sensor_msgs.msg import Image

import cv2
import numpy as np
from cv_bridge import CvBridge

DEV = "/dev/video0"

def sh(cmd: str) -> int:
    return subprocess.call(shlex.split(cmd))

def set_ctrl(ctrls: str) -> int:
    # ctrls example: "auto_exposure=0,auto_exposure_bias=10"
    return sh(f"v4l2-ctl -d {DEV} --set-ctrl={ctrls}")

def get_ctrl(name: str) -> str:
    p = subprocess.check_output(shlex.split(f"v4l2-ctl -d {DEV} --get-ctrl={name}"))
    return p.decode("utf-8", errors="ignore").strip()

# ----- Tuning knobs (start values) -----
BIAS_NORMAL = 12
BIAS_DIM    = 10
BIAS_DARKER = 8

# Hysteresis thresholds based on your data
S_BAD  = 65.0   # enter "whitening" mitigation
S_GOOD = 65.0   # exit back upward
S_GO_UP = 150.0 # need stronger evidence to go back up
# Frame persistence
K_DOWN = 3       # consecutive frames to step down
M_UP   = 10      # consecutive frames to step up

# Lane evidence gate: fraction of pixels with V > 200 in ROI
B_MIN = 0.01     # 1%

# ROI config (bottom strip; optionally exclude center if front car dominates)
ROI_Y0_FRAC = 0.60          # bottom 40% like your offline test
ROI_X0_FRAC = 0.00
ROI_X1_FRAC = 1.00
EXCLUDE_CENTER = False       # set True if front car occupies center bottom
EXC_X0_FRAC = 0.35
EXC_X1_FRAC = 0.65

# Bright mask threshold for "hiV"
V_HI = 200
S_WHITE_THR = 40  # for "white_like" if you want to log it

# Control update rate (don’t hammer v4l2)
UPDATE_HZ = 5.0   # update decision up to 5 Hz (10 fps camera)

# --------------------------------------

bridge = CvBridge()

def compute_metrics(bgr: np.ndarray):
    """Return bright_frac, s_mean_hiV, white_like for the chosen ROI."""
    h, w = bgr.shape[:2]
    y0 = int(h * ROI_Y0_FRAC)
    x0 = int(w * ROI_X0_FRAC)
    x1 = int(w * ROI_X1_FRAC)

    roi = bgr[y0:h, x0:x1].copy()

    if EXCLUDE_CENTER:
        cx0 = int((x1 - x0) * EXC_X0_FRAC)
        cx1 = int((x1 - x0) * EXC_X1_FRAC)
        roi[:, cx0:cx1, :] = 0  # mask out center

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    hiV = v > V_HI
    bright_frac = float(np.mean(hiV))

    if np.any(hiV):
        s_mean_hiV = float(np.mean(s[hiV]))
        white_like = float(np.mean(s[hiV] < S_WHITE_THR))
    else:
        s_mean_hiV = 0.0
        white_like = 0.0

    return bright_frac, s_mean_hiV, white_like

class BiasController:
    def __init__(self):
        self.bias = BIAS_NORMAL
        self.down_count = 0
        self.up_count = 0
        self.last_set_time = 0.0
        self.cooldown_s = 0.6  # avoid rapid toggling

    def _set_bias(self, new_bias: int):
        if new_bias == self.bias:
            return False  # No change made

        # Clamp to [0,24] just in case
        new_bias = max(0, min(24, new_bias))
        set_ctrl(f"auto_exposure_bias={new_bias}")
        self.bias = new_bias
        self.last_set_time = time.time()
        rospy.loginfo(f"[AE] Set auto_exposure_bias={new_bias}")
        return True  # Change made

    def update(self, bright_frac: float, s_mean_hiV: float):
        now = time.time()
        if now - self.last_set_time < self.cooldown_s:
            return False  # cooldown, no change

        # Gate: if no lane evidence, force neutral and reset counters
        if bright_frac < B_MIN:
            self.down_count = 0
            self.up_count = 0
            return self._set_bias(BIAS_NORMAL)

        # Decide down / up with hysteresis
        if s_mean_hiV < S_BAD:
            self.down_count += 1
            self.up_count = 0
        elif s_mean_hiV > S_GO_UP:
            self.up_count += 1
            self.down_count = 0
        else:
            # in the hysteresis band, decay counts gently
            self.down_count = max(0, self.down_count - 1)
            self.up_count = max(0, self.up_count - 1)

        # Step down
        if self.down_count >= K_DOWN:
            if self.bias == BIAS_NORMAL:
                self.down_count = 0
                return self._set_bias(BIAS_DIM)
            elif self.bias == BIAS_DIM:
                self.down_count = 0
                return self._set_bias(BIAS_DARKER)

        # Step up
        if self.up_count >= M_UP:
            if self.bias == BIAS_DARKER:
                self.up_count = 0
                return self._set_bias(BIAS_DIM)
            elif self.bias == BIAS_DIM:
                self.up_count = 0
                return self._set_bias(BIAS_NORMAL)

        return False  # No change made

def main():
    # Ensure AE is on, start neutral
    set_ctrl("auto_exposure=0,auto_exposure_bias=12")

    rospy.init_node("adaptive_ae_bias", anonymous=False)

    ctrl = BiasController()
    last_update = 0.0
    period = 1.0 / UPDATE_HZ

    def cb(msg: Image):
        nonlocal last_update
        now = time.time()
        if now - last_update < period:
            return
        last_update = now

        try:
            bgr = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn(f"[AE] cv_bridge error: {e}")
            return

        bright_frac, s_mean, white_like = compute_metrics(bgr)
        adjustment_made = ctrl.update(bright_frac, s_mean)
        if adjustment_made:
            rospy.loginfo(
                f"[AE] Adjustment made: bright_frac={bright_frac:.3f}, s_mean_hiV={s_mean:.1f}, white_like={white_like*100:.1f}%, new_bias={ctrl.bias}"
            )

    sub = rospy.Subscriber("robot_cam/image_raw", Image, cb, queue_size=1)

    rospy.loginfo("[AE] Adaptive bias node running.")
    rospy.spin()

if __name__ == "__main__":
    main()

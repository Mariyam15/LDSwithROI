import cv2
import numpy as np

class LaneDetector:
    def __init__(self, debug=False):
        self.debug = debug
        self.yellow_lower = np.array([10, 90, 90], dtype=np.uint8)
        self.yellow_upper = np.array([40, 255, 255], dtype=np.uint8)
        self.white_lower  = np.array([0, 0, 200], dtype=np.uint8)
        self.white_upper  = np.array([180, 40, 255], dtype=np.uint8)

    def process_image(self, img):
        debug = {}

        # HSV Mask
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_y = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        mask_w = cv2.inRange(hsv, self.white_lower, self.white_upper)
        mask = cv2.bitwise_or(mask_y, mask_w)

        # Gaussian Blur
        blur = cv2.GaussianBlur(mask, (5, 5), 1.5)
        debug['2_blur'] = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

        # Canny edges
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
        edges = self.canny_custom(gray_blur)
        edges = cv2.bitwise_and(edges, blur)
        debug['3_edges'] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # ROI
        h, w = img.shape[:2]
        roi_mask = np.zeros_like(edges)
        pts = np.array([[
            (int(0.05 * w), h),
            (int(0.45 * w), int(0.6 * h)),
            (int(0.55 * w), int(0.6 * h)),
            (int(0.95 * w), h)
        ]], dtype=np.int32)
        cv2.fillPoly(roi_mask, pts, 255)
        masked_edges = cv2.bitwise_and(edges, roi_mask)
        debug['4_roi_edges'] = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)

        # Hough
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=30,
                                minLineLength=30, maxLineGap=20)
        hough_show = img.copy()
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                cv2.line(hough_show, (x1, y1), (x2, y2), (0, 255, 0), 2)
        debug['5_hough'] = hough_show

        # Fit lane lines
        left_line, right_line = self.fit_lane_lines(lines, w, h)
        final = img.copy()
        if left_line is not None:
            cv2.line(final, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0,0,255), 8)
        if right_line is not None:
            cv2.line(final, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0,0,255), 8)
        cv2.polylines(final, pts, True, (255,0,0), 2)

        return final, debug if self.debug else {}

    def canny_custom(self, gray):
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        mag = (mag / (mag.max() + 1e-6) * 255).astype(np.uint8)
        angle = np.arctan2(gy, gx)

        H, W = mag.shape
        nms = np.zeros_like(mag, dtype=np.uint8)
        ang_deg = angle * 180 / np.pi
        ang_deg[ang_deg < 0] += 180

        for i in range(1, H-1):
            for j in range(1, W-1):
                a = ang_deg[i, j]
                q = r = 255
                if (0 <= a < 22.5) or (157.5 <= a <= 180):
                    q = mag[i, j+1]
                    r = mag[i, j-1]
                elif (22.5 <= a < 67.5):
                    q = mag[i+1, j-1]
                    r = mag[i-1, j+1]
                elif (67.5 <= a < 112.5):
                    q = mag[i+1, j]
                    r = mag[i-1, j]
                elif (112.5 <= a < 157.5):
                    q = mag[i-1, j-1]
                    r = mag[i+1, j+1]
                if mag[i, j] >= q and mag[i, j] >= r:
                    nms[i, j] = mag[i, j]

        high, low = 80, 40
        strong, weak = 255, 75
        res = np.zeros_like(nms)
        strong_i, strong_j = np.where(nms >= high)
        weak_i, weak_j     = np.where((nms < high) & (nms >= low))
        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        for i in range(1, H-1):
            for j in range(1, W-1):
                if res[i,j] == weak:
                    if any(res[i+di, j+dj] == strong for di in [-1,0,1] for dj in [-1,0,1]):
                        res[i,j] = strong
                    else:
                        res[i,j] = 0
        return res

    def fit_lane_lines(self, lines, w, h):
        if lines is None:
            return None, None
        left_pts, right_pts = [], []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            if x1 == x2: continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3: continue
            if slope < 0:
                left_pts += [(x1, y1), (x2, y2)]
            else:
                right_pts += [(x1, y1), (x2, y2)]
        def fit(points):
            if len(points) < 2: return None
            xs = np.array([p[0] for p in points])
            ys = np.array([p[1] for p in points])
            m, b = np.polyfit(xs, ys, 1)
            y1 = h; y2 = int(0.6*h)
            x1 = int((y1-b)/m); x2 = int((y2-b)/m)
            return (x1, y1, x2, y2)
        return fit(left_pts), fit(right_pts)

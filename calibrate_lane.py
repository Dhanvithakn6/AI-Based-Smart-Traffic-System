import cv2
import numpy as np
import json
import argparse
import os

class MetricCalibrator:
    def __init__(self, lane_id, video_path):
        self.lane_id = lane_id
        self.video_path = video_path
        self.image_points = []
        self.world_points = []
        self.frame = None
        
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} not found.")
            exit(1)

    def get_representative_frame(self):
        cap = cv2.VideoCapture(self.video_path)
        # Skip a few frames to ensure a stable image
        for _ in range(10): cap.read()
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Error: Could not read frame.")
            exit(1)
        return frame

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.image_points) < 4:
                # Check for duplicates
                if [x, y] in self.image_points:
                    print("Error: Duplicate point selected.")
                    return
                self.image_points.append([x, y])
                cv2.circle(self.display_frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(self.display_frame, f"P{len(self.image_points)}", (x+10, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Calibration - Step 1: Click 4 Ground Points", self.display_frame)

    def is_degenerate(self, pts):
        """Check if 4 points form a degenerate quadrilateral (e.g., collinear)"""
        if len(pts) < 4: return True
        for i in range(4):
            # Check every combination of 3 points for collinearity using triangle area
            p1, p2, p3 = pts[i], pts[(i+1)%4], pts[(i+2)%4]
            area = 0.5 * abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1]))
            if area < 1e-5: return True
        return False

    def run(self):
        self.frame = self.get_representative_frame()
        self.display_frame = self.frame.copy()

        print(f"\n--- Metric Calibration: Lane {self.lane_id} ---")
        print("Step 1: Select 4 points on the road surface (e.g., BL, BR, TR, TL).")
        
        cv2.namedWindow(
         "Calibration - Step 1: Click 4 Ground Points",
          cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
        )
        cv2.resizeWindow("Calibration - Step 1: Click 4 Ground Points", 1400, 800)
        cv2.setMouseCallback("Calibration - Step 1: Click 4 Ground Points", self.mouse_callback)

        while len(self.image_points) < 4:
            cv2.imshow("Calibration - Step 1: Click 4 Ground Points", self.display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): return

        if self.is_degenerate(self.image_points):
            print("Error: The selected image points form a degenerate quadrilateral.")
            return

        print("\nStep 2: Enter Real-World Coordinates (X, Y) in METERS.")
        for i in range(4):
            while True:
                try:
                    inp = input(f" P{i+1} (pixel {self.image_points[i]}) -> Enter 'X, Y' in meters: ")
                    x_w, y_w = map(float, inp.split(','))
                    self.world_points.append([x_w, y_w])
                    break
                except ValueError:
                    print(" Invalid input. Format: X, Y (e.g., 0, 0 or 12.5, 30)")

        if self.is_degenerate(self.world_points):
            print("Error: The entered world coordinates form a degenerate quadrilateral.")
            return

        # Core Mathematical Calculation
        src_pts = np.float32(self.image_points)
        dst_pts = np.float32(self.world_points)
        
        # H maps (u,v) -> (X,Y) in meters
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        H_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

        # Step 3: Verification & Error Reporting
        print("\n--- Calibration Verification ---")
        errors = []
        for i in range(4):
            px = np.array([self.image_points[i][0], self.image_points[i][1], 1.0])
            mapped = np.dot(H, px)
            mapped /= mapped[2] # Normalize w
            
            actual = np.array(self.world_points[i])
            error = np.linalg.norm(mapped[:2] - actual)
            errors.append(error)
            print(f" P{i+1}: Image{self.image_points[i]} -> Calibrated({mapped[0]:.3f}, {mapped[1]:.3f}) | Error: {error:.4f}m")

        avg_error = np.mean(errors)
        print(f" Average Re-projection Error: {avg_error:.5f} meters")

        # Step 4: Visual Verification (Bird's Eye View)
        # To show BEV, we need temporary display scaling
        # We find the bounds of our world points to define the view
        min_x = min(p[0] for p in self.world_points)
        max_x = max(p[0] for p in self.world_points)
        min_y = min(p[1] for p in self.world_points)
        max_y = max(p[1] for p in self.world_points)
        
        world_w = max_x - min_x
        world_h = max_y - min_y
        
        # Display scale: 20 pixels per meter (only for visualization)
        d_scale = 20 
        disp_w, disp_h = int(world_w * d_scale) + 100, int(world_h * d_scale) + 100
        
        # Offset matrix to center the view in the preview window
        T = np.array([[d_scale, 0, 50 - min_x*d_scale], 
                      [0, -d_scale, disp_h - 50 + min_y*d_scale], 
                      [0, 0, 1]])
        H_viz = np.dot(T, H)
        
        bev_preview = cv2.warpPerspective(self.frame, H_viz, (disp_w, disp_h))
        cv2.imshow("Verification - Bird's Eye View Preview", bev_preview)
        print("\nCheck the BEV window. If road markings are parallel/straight, press any key to save.")
        cv2.waitKey(0)

        # Step 5: Save Configuration
        # Calculate a reference width (Distance between P1 and P2)
        ref_width = np.linalg.norm(np.array(self.world_points[0]) - np.array(self.world_points[1]))

        config_data = {
            "lane_id": self.lane_id,
            "coordinate_unit": "meters",
            "homography_matrix": H.tolist(),
            "inverse_homography_matrix": H_inv.tolist(),
            "image_points": self.image_points,
            "world_points": self.world_points,
            "reference_width_m": ref_width,
            "avg_reprojection_error_m": avg_error
        }

        filename = f"config_lane{self.lane_id}.json"
        with open(filename, 'w') as f:
            json.dump(config_data, f, indent=4)
        
        print(f"\n✅ Configuration successfully saved to {filename}")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metric Traffic Lane Calibrator")
    parser.add_argument("--lane", type=int, required=True, help="Lane index (1-4)")
    args = parser.parse_args()

    video_file = f"lane{args.lane}.mp4"
    calibrator = MetricCalibrator(args.lane, video_file)
    calibrator.run()
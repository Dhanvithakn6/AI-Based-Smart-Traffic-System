import cv2
import numpy as np
import json
from ultralytics import YOLO
import time
import threading
from collections import deque
import os

class LaneProcessor:
    """Processes individual lane video and detects vehicles"""
    
    def __init__(self, video_path, lane_id):
         self.lane_id = lane_id
         self.video_path = video_path
         self.cap = cv2.VideoCapture(video_path)
         self.display_frame = None
         self.is_eof = False

        # YOLO model
         self.model = YOLO("yolov11n.pt")

        # Vehicle class IDs
         self.vehicle_classes = {
            2: "car",
            3: "bike",
            5: "bus",
            7: "truck"
       }

        # Vehicle weights
         self.vehicle_weights = {
            "bike": 1,
            "car": 2,
            "bus": 4,
            "truck": 5
        }

                       # Load lane-specific metric calibration
         config_file = f"config_lane{lane_id}.json"

         if os.path.exists(config_file):
            with open(config_file, "r") as f:
                calibration = json.load(f)

            self.road_width = float(calibration["reference_width_m"])

            # Homography: image pixels -> real-world ground coordinates
            self.H = np.array(
                calibration["homography_matrix"],
                dtype=np.float64
            )

            # Monitoring zone in meters
            zone = calibration.get(
                "monitoring_zone_meters",
                {
                    "min_y": 0.0,
                    "max_y": 20.0
                }
            )

            self.min_y = float(zone["min_y"])
            self.max_y = float(zone["max_y"])

            print(
                f"✅ Lane {lane_id} calibration loaded | "
                f"Width = {self.road_width:.2f} m | "
                f"Zone = {self.min_y:.1f}-{self.max_y:.1f} m"
            )

         else:
            print(f"⚠️ Calibration file not found: {config_file}")
            self.road_width = 1.0
            self.H = np.eye(3)
            self.min_y = 0.0
            self.max_y = 20.0

         self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
         self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

         self.current_frame = None

        # Vehicle statistics
         self.vehicle_count = 0
         self.car_count = 0
         self.bike_count = 0
         self.bus_count = 0
         self.truck_count = 0

         self.weighted_count = 0
         self.queue_length = 0
         self.congestion_score = 0

         self.detection_history = deque(maxlen=10)
    def pixel_to_meters(self, u, v):
        """Convert image pixel coordinates to ground-plane meters."""
        point = np.array(
            [u, v, 1.0],
            dtype=np.float64
        ).reshape(3, 1)

        world_point = self.H @ point

        if abs(world_point[2, 0]) < 1e-9:
            return None, None

        world_point /= world_point[2, 0]

        return (
            float(world_point[0, 0]),
            float(world_point[1, 0])
        )
  
    def detect_vehicles(self):
        """Detect vehicles and count only vehicles inside metric monitoring zone."""
        if self.current_frame is None:
            return

        results = self.model(self.current_frame, verbose=False)

        current_count = 0

        self.car_count = 0
        self.bike_count = 0
        self.bus_count = 0
        self.truck_count = 0

        for result in results:
            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:

                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id not in self.vehicle_classes or conf <= 0.5:
                    continue

                # Bounding box
                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                # Bottom-center = approximate vehicle/road contact point
                u_center = (x1 + x2) / 2.0
                v_bottom = float(y2)

                # Convert pixel -> ground-plane meters
                x_m, y_m = self.pixel_to_meters(
                    u_center,
                    v_bottom
                )

                # If transformation failed
                if x_m is None or y_m is None:
                    continue

                class_name = self.vehicle_classes[cls_id]

                # -------------------------------------------------
                # MONITORING ZONE FILTER
                # Only vehicles between 0m and 20m are counted
                # -------------------------------------------------
                inside_zone = (
                    self.min_y <= y_m <= self.max_y
                )

                if inside_zone:

                    current_count += 1

                    if class_name == "car":
                        self.car_count += 1

                    elif class_name == "bike":
                        self.bike_count += 1

                    elif class_name == "bus":
                        self.bus_count += 1

                    elif class_name == "truck":
                        self.truck_count += 1

                    # Green = vehicle contributes to congestion
                    box_color = (0, 255, 0)

                    cv2.putText(
                        self.current_frame,
                        f"{y_m:.1f}m",
                        (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1
                    )

                else:

                    # Gray = detected but outside monitoring zone
                    box_color = (100, 100, 100)

                cv2.rectangle(
                    self.current_frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2
                )

        # Moving average of vehicles INSIDE monitoring zone
        self.detection_history.append(current_count)

        self.vehicle_count = int(
            np.mean(self.detection_history)
        )

        # Weighted congestion
        weighted_count = (
            self.bike_count * self.vehicle_weights["bike"] +
            self.car_count * self.vehicle_weights["car"] +
            self.bus_count * self.vehicle_weights["bus"] +
            self.truck_count * self.vehicle_weights["truck"]
        )

        self.weighted_count = weighted_count

        # Density relative to calibrated road width
        self.congestion_score = (
            weighted_count / self.road_width
            if self.road_width > 0
            else 0
        )
        
    def read_frame(self, visible=True):
        """Read next frame. If visible=False, analyze but do not update dashboard display."""
        if self.is_eof:
            return True

        ret, frame = self.cap.read()
        if not ret:
            self.is_eof = True
            return True

        self.current_frame = frame.copy()
        
        # Update display_frame only if this lane is currently 'active' visually
        if visible or self.display_frame is None:
            self.display_frame = frame.copy()

        self.detect_vehicles()
        return True

    def reset_video(self):
        """Restart video from frame 0."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.is_eof = False
    
        
        # Lane header
        cv2.rectangle(self.current_frame, (0, 0), (w, 60), (50, 50, 50), -1)
        cv2.putText(self.current_frame, f"LANE {self.lane_id}", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Vehicle count and density
        # Vehicle information
        cv2.putText(
            self.current_frame,
            f"Vehicles : {self.vehicle_count}",
            (10, h-110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        cv2.putText(
        self.current_frame,
        f"Road Width : {self.road_width}",
        (10, h-80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
        )

        cv2.putText(
            self.current_frame,
            f"Congestion : {self.congestion_score:.2f}",
            (10, h-50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,0),
            2
        )
        
        return self.current_frame
    
    def get_frame(self):
        """Return the frame designated for display (stays frozen if not active)."""
        return self.display_frame if self.display_frame is not None else self.current_frame

class TrafficSignalController:
    """Manages 4-lane traffic signal logic"""
    
    def __init__(self):
        self.lanes = None
        self.current_green_lane = 0
        self.signal_start_time = 0
        self.signal_duration = 10
        self.time_remaining = 10
        
        # Fairness tracking (time since last green)
        self.last_green_time = [0] * 4
        self.wait_time = [0] * 4
        self.fairness_threshold = 120  # 2 minutes max wait
        
        # Signal timing based on vehicle count
        self.signal_durations = {
            (0, 10): 10,
            (10, 30): 20,
            (30, float('inf')): 30
        }
    
    def load_lanes(self, video_paths):
        """Initialize 4 lane processors"""
        self.lanes = []
        for i, path in enumerate(video_paths):
            if os.path.exists(path):
                lane = LaneProcessor(path, i+1)
                self.lanes.append(lane)
                print(f"✅ Loaded Lane {i+1}: {path}")
            else:
                print(f"❌ Lane {i+1} video not found: {path}")
                # Create dummy lane if video missing
                self.lanes.append(LaneProcessor(0, i+1))
    
    def get_signal_duration(self, vehicle_count):
        """Calculate green time based on vehicle count"""
        for (min_count, max_count), duration in self.signal_durations.items():
            if min_count <= vehicle_count < max_count:
                return duration
        return 30
    
    def select_next_lane(self):
        """Strict Round-Robin selection. Order: 1 -> 2 -> 3 -> 4."""
        next_lane_idx = (self.current_green_lane + 1) % 4
        # Note: adaptive duration is calculated inside switch_green_lane 
        # using the vehicle_count prepared during the 5s background analysis.
        self.switch_green_lane(next_lane_idx)
    
    def switch_green_lane(self, lane_id):
        """Switch to new green lane"""
        self.current_green_lane = lane_id
        self.last_green_time[lane_id] = time.time()
        
        vehicle_count = self.lanes[lane_id].vehicle_count
        self.signal_duration = self.get_signal_duration(vehicle_count)
        self.signal_start_time = time.time()
        self.time_remaining = self.signal_duration
        
        print(f"\n🟢 GREEN → LANE {lane_id+1} | Vehicles: {vehicle_count} | Time: {self.signal_duration}s")
    
    def update_signal(self):
        """Update signal timer and logic"""
        current_time = time.time()
        self.time_remaining = max(0, self.signal_duration - (current_time - self.signal_start_time))
        
        # Switch lane when time expires
        if self.time_remaining <= 0:
            self.select_next_lane()
    
    def draw_signal_status(self, frame, lane_id):
        """Draw traffic light status on frame"""
        h, w = frame.shape[:2]
        
        # Traffic light
        light_y = 70
        light_x = w // 2
        
        cv2.circle(frame, (light_x, light_y), 30, (0, 0, 100), -1)  # Background
        
        next_lane_idx = (self.current_green_lane + 1) % 4
        
        if lane_id == self.current_green_lane:
            color, status = (0, 255, 0), "GREEN"
        elif lane_id == next_lane_idx and self.time_remaining <= 5:
            color, status = (0, 255, 255), "YELLOW"
        else:
            color, status = (0, 0, 255), "RED"
        
        cv2.circle(frame, (light_x, light_y), 25, color, -1)
        
        # Status text
        cv2.putText(frame, status, (light_x - 40, light_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Global timer for green lane
        if lane_id == self.current_green_lane:
            timer_text = f"TIMER: {self.time_remaining:.1f}s"
            cv2.putText(frame, timer_text, (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return frame

class FourLaneTrafficSystem:
    """Main 4-lane traffic management system"""
    
    def __init__(self, video_paths):
        self.controller = TrafficSignalController()
        self.controller.load_lanes(video_paths)
        self.running = True
        
        # Window names
        self.window_names = [f"Lane_{i+1}" for i in range(4)]
        
    def create_windows(self):
        """Create single dashboard window"""
        cv2.namedWindow("AI Smart Traffic Dashboard", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AI Smart Traffic Dashboard", 1920, 1080)
    
    def process_all_lanes(self):
        """Process all lanes and restart videos together after all reach EOF."""

        # Restart ALL videos only when ALL four have reached EOF
        if all(lane.is_eof for lane in self.controller.lanes):
            print("🔄 Dataset Round Complete: Restarting all lanes from frame 0...")

            for lane in self.controller.lanes:
                lane.reset_video()

        frames = []

        curr_green = self.controller.current_green_lane
        next_lane = (curr_green + 1) % 4
        is_pre_analysis_period = self.controller.time_remaining <= 5

        for i, lane in enumerate(self.controller.lanes):
            if i == curr_green:
                # Active Green: Advance video visually and analyze
                lane.read_frame(visible=True)
            elif i == next_lane and is_pre_analysis_period:
                # Background Yellow: Analyze traffic but keep dashboard video frozen
                lane.read_frame(visible=False)
            
            # Non-active lanes return their last frozen 'display_frame'
            frame = self.controller.draw_signal_status(lane.get_frame(), i)
            frames.append(frame)

        return frames, True
    
    def display_frames(self, frames):
        """Display all 4 lanes in one dashboard"""

        processed = []

        for i, frame in enumerate(frames):

            if frame is None:
                frame = np.zeros((450, 800, 3), dtype=np.uint8)

            # Resize FIRST
            frame = cv2.resize(frame, (800, 450))

             # Draw lane header
            cv2.rectangle(frame, (0, 0), (800, 45), (45,45,45), -1)
            cv2.putText(frame,
                        f"LANE {i+1}",
                        (15,30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255,255,255),
                        2)

            lane = self.controller.lanes[i]

            # Bottom information panel
            cv2.rectangle(frame, (0,390), (800,450), (30,30,30), -1)

            cv2.putText(frame,
                        f"Vehicles : {lane.vehicle_count}",
                        (20,415),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,255,0),
                        2)

            cv2.putText(frame,
                        f"Width : {lane.road_width}",
                        (260,415),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255,255,255),
                        2)

            cv2.putText(frame,
                        f"Congestion : {lane.congestion_score:.2f}",
                        (420,415),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,255,255),
                        2)

            next_lane = (self.controller.current_green_lane + 1) % 4

            if i == self.controller.current_green_lane:
                color = (0,255,0)
                status = "GREEN"
            elif i == next_lane and self.controller.time_remaining <= 5:
                color = (0,255,255)
                status = "YELLOW"
            else:
                color = (0,0,255)
                status = "RED"

            cv2.circle(frame,(740,22),14,color,-1)

            cv2.putText(frame,
                        status,
                        (610,32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2)

            if i == self.controller.current_green_lane:
                cv2.putText(frame,
                            f"{self.controller.time_remaining:.1f}s",
                            (650,440),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,255,255),
                            2)

            processed.append(frame)

        top = np.hstack((processed[0], processed[1]))
        bottom = np.hstack((processed[2], processed[3]))

        dashboard = np.vstack((top, bottom))

        cv2.imshow("AI Smart Traffic Dashboard", dashboard)
    
    def run(self):
        """Main processing loop"""
        print("🚦 4-LANE ADAPTIVE TRAFFIC SIGNAL SYSTEM")
        print("=" * 50)
        self.create_windows()
        
                # Analyze all four lanes before assigning the first green signal
        for lane in self.controller.lanes:
            lane.read_frame(visible=True)

        initial_lane = max(
            range(4),
            key=lambda i: self.controller.lanes[i].congestion_score
        )

        self.controller.switch_green_lane(initial_lane)

        print(
            f"🚦 Initial priority: Lane {initial_lane + 1} "
            f"(Congestion: {self.controller.lanes[initial_lane].congestion_score:.2f})"
        )

        fps_counter = 0
        fps_time = time.time()
        
        while self.running:
            frames, all_active = self.process_all_lanes()
            
            # Update signal logic
            self.controller.update_signal()
            
            # Display frames
            self.display_frames(frames)
            
            # FPS counter
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps = 30 / (time.time() - fps_time)
                print(f"FPS: {fps:.1f} | Green: L{self.controller.current_green_lane+1} | "
                      f"T: {self.controller.time_remaining:.1f}s")
                fps_time = time.time()
            
            # Handle key events (works across all windows)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('r'):
                for lane in self.controller.lanes:
                    lane.vehicle_count = 0
                    lane.detection_history.clear()
                print("🔄 All counts reset!")
            elif key == ord('n'):
                self.controller.switch_green_lane((self.controller.current_green_lane + 1) % 4)
        
        self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        for lane in self.controller.lanes:
            lane.cap.release()
        cv2.destroyAllWindows()
        print("\n📊 Final Statistics:")
        for i, lane in enumerate(self.controller.lanes):
            print(
                 f"Lane {i+1}: "
                 f"{lane.vehicle_count} vehicles | "
                 f"Congestion = {lane.congestion_score:.2f}"
                )

def main():
    """Main function with default video paths"""
    video_paths = [
        "lane1.mp4",
        "lane2.mp4", 
        "lane3.mp4",
        "lane4.mp4"
    ]
    
    # Check if videos exist, use webcam if files missing
    final_paths = []
    for path in video_paths:
        if os.path.exists(path):
            final_paths.append(path)
        else:
            print(f"📹 Using webcam for {path}")
            final_paths.append(0)  # Webcam
    
    system = FourLaneTrafficSystem(final_paths)
    system.run()

if __name__ == "__main__":
    main()
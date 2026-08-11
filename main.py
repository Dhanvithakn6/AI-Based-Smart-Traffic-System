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
        self.cap = cv2.VideoCapture(video_path)

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
        
    def detect_vehicles(self):
        """Detect vehicles in current frame"""
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
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_id in self.vehicle_classes and conf > 0.5:
                        class_name = self.vehicle_classes[cls_id]

                        current_count += 1

                        if class_name == "car":
                            self.car_count += 1
                        elif class_name == "bike":
                            self.bike_count += 1
                        elif class_name == "bus":
                            self.bus_count += 1
                        elif class_name == "truck":
                            self.truck_count += 1

                        # Draw bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        cv2.rectangle(
                            self.current_frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 255),
                            2
                        )
                        
               # Update count using moving average
        self.detection_history.append(current_count)
        self.vehicle_count = int(np.mean(self.detection_history))

        # Weighted congestion score
        weighted_count = (
            self.bike_count * 1 +
            self.car_count * 2 +
            self.bus_count * 4 +
            self.truck_count * 5
        )

        self.weighted_count = weighted_count
        self.congestion_score = weighted_count / self.road_width
        
    def read_frame(self):
        """Read next frame and loop video when it ends"""

        ret, frame = self.cap.read()

        # If video ended, restart it
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        if ret:
            self.current_frame = frame.copy()
            self.detect_vehicles()

        return ret
    
        
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
        """Get processed frame with annotations"""
        return self.current_frame

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
        """Intelligent lane selection with fairness"""
        current_time = time.time()
        
        # Calculate wait times
        for i in range(4):
            self.wait_time[i] = current_time - self.last_green_time[i]
        
        # At 80% of current signal time, evaluate other lanes
        elapsed = current_time - self.signal_start_time
        if elapsed > 0.8 * self.signal_duration:
            
            # Candidates: all lanes except current green
            candidates = [
                (
                    i,
                    lane.congestion_score,
                    lane.vehicle_count,
                    self.wait_time[i]
                )
                for i, lane in enumerate(self.lanes)
                if i != self.current_green_lane
            ]
            
            if candidates:
                # Priority: density (70%) + fairness (30%)
                candidates.sort(key=lambda x: (
                    x[1] * 0.8 +  # Density priority
                    (x[3] / self.fairness_threshold) * 0.3  # Fairness boost
                ), reverse=True)
                
                next_lane = candidates[0][0]
            else:
                next_lane = (self.current_green_lane + 1) % 4
            
            self.switch_green_lane(next_lane)
    
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
        
        if lane_id == self.current_green_lane:
            color = (0, 255, 0)  # Green
            status = "GREEN"
        else:
            color = (0, 0, 255)  # Red
            status = "RED"
        
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
        """Process one frame from each lane"""
        frames = []
        all_active = True
        
        for lane in self.controller.lanes:
            ret = lane.read_frame()
            if ret:
                # Add signal status
                frame = self.controller.draw_signal_status(lane.get_frame(), lane.lane_id - 1)
                frames.append(frame)
            else:
                all_active = False
                frames.append(None)
        
        return frames, all_active
    
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

            color = (0,255,0) if i == self.controller.current_green_lane else (0,0,255)
            status = "GREEN" if i == self.controller.current_green_lane else "RED"

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
        self.process_all_lanes()

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
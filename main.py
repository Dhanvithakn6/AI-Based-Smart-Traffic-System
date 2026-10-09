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
    
    def __init__(
        self,
        video_path,
        lane_id,
        emergency_model=None,
        emergency_state_lock=None
    ):
         self.lane_id = lane_id
         self.video_path = video_path
         self.cap = cv2.VideoCapture(video_path)
         self.traffic_cap = cv2.VideoCapture(video_path)
         self.display_frame = None
         self.is_eof = False

         # Normal traffic model
         self.model = YOLO("yolov11n.pt")

# Emergency model is passed from the controller
         self.emergency_model = emergency_model
         self.emergency_state_lock = (
             emergency_state_lock or threading.Lock()
         )

# Emergency detection state
         self.ambulance_detected = False
         self.consecutive_ambulance_frames = 0
         self.clearance_counter = 0
         self.clearance_started_at = None
         self.last_emergency_diagnostic_at = 0.0

         self.REQUIRED_CONFIRMATION_FRAMES = 5
         self.AMBULANCE_CLEARANCE_FRAMES = 15

# Separate video stream for background ambulance monitoring
         self.emergency_cap = cv2.VideoCapture(video_path)

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
         self.statistics_lock = threading.Lock()
         self._analysis_requests = threading.Semaphore(0)
         self._analysis_request_lock = threading.Lock()
         self._pending_analysis_request_ids = deque()
         self._next_analysis_request_id = 0
         self._completed_analysis_request_id = 0
         self._analysis_stop_event = threading.Event()
         self._inference_error = None
         self.detection_overlay = []
         self.inference_count = 0
         self.inference_elapsed = 0.0
         self.traffic_measurement_ready = False
         self.analysis_revision = 0
         self._inference_thread = None

    def start_traffic_analysis(self):
        if self._inference_thread is not None:
            return

        self._inference_thread = threading.Thread(
            target=self._vehicle_inference_loop,
            name=f"vehicle-inference-lane-{self.lane_id}",
            daemon=True
        )
        self._inference_thread.start()

    def request_traffic_analysis(self):
        """Request one background analysis without waiting for the worker."""
        if self._inference_error is not None:
            self.check_traffic_analysis_error()
        if self._inference_thread is None:
            raise RuntimeError(
                f"Traffic analysis has not started for lane {self.lane_id}"
            )
        with self._analysis_request_lock:
            self._next_analysis_request_id += 1
            request_id = self._next_analysis_request_id
            self._pending_analysis_request_ids.append(request_id)
        self._analysis_requests.release()
        return request_id

    def is_traffic_analysis_complete(self, request_id):
        with self._analysis_request_lock:
            return self._completed_analysis_request_id >= request_id

    def _set_traffic_analysis_error(self, error):
        self._inference_error = error

    def _vehicle_inference_loop(self):
        while not self._analysis_stop_event.is_set():
            if not self._analysis_requests.acquire(timeout=0.1):
                continue
            if self._analysis_stop_event.is_set():
                return

            with self._analysis_request_lock:
                request_id = self._pending_analysis_request_ids.popleft()

            started_at = time.perf_counter()
            try:
                ret, frame = self.traffic_cap.read()
                if not ret:
                    self.traffic_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.traffic_cap.read()
                if not ret:
                    raise RuntimeError(
                        f"Unable to read traffic-analysis video for "
                        f"lane {self.lane_id}"
                    )
                self.current_frame = frame
                annotations = self.detect_vehicles()
            except Exception as error:
                self._set_traffic_analysis_error(error)
                return

            elapsed = time.perf_counter() - started_at
            with self.statistics_lock:
                self.detection_overlay = annotations
                self.inference_count += 1
                self.inference_elapsed += elapsed
                self.traffic_measurement_ready = True
                self.analysis_revision += 1
            with self._analysis_request_lock:
                self._completed_analysis_request_id = request_id

    def stop_vehicle_detection(self):
        self._analysis_stop_event.set()
        self._analysis_requests.release()
        if self._inference_thread is not None:
            self._inference_thread.join()

    def check_traffic_analysis_error(self):
        if self._inference_error is not None:
            raise RuntimeError(
                f"Traffic analysis failed for lane {self.lane_id}"
            ) from self._inference_error

    def detect_emergency(self, frame):
        """Background ambulance detection using the dedicated ambulance model."""
        if frame is None or self.emergency_model is None:
             return False

        results = self.emergency_model(
        frame,
        conf=0.70,
        classes=[0],
        verbose=False
       )

        confidence = max(
            (
                float(box.conf[0])
                for result in results
                if result.boxes is not None
                for box in result.boxes
            ),
            default=0.0
        )
        found = confidence >= 0.70

        clearance_message = None
        if found:
            print(
                f"🚨 Ambulance candidate: Lane {self.lane_id}, "
                f"confidence = {confidence:.2f}"
            )

        with self.emergency_state_lock:
            if found:
                self.consecutive_ambulance_frames += 1
                self.clearance_counter = 0
                self.clearance_started_at = None

                if (
                    self.consecutive_ambulance_frames
                    >= self.REQUIRED_CONFIRMATION_FRAMES
                ):
                    if not self.ambulance_detected:
                        print(
                            f"✅ Ambulance confirmed: Lane {self.lane_id}"
                        )
                    self.ambulance_detected = True
            else:
                self.consecutive_ambulance_frames = 0

            if self.ambulance_detected and not found:
                if self.clearance_counter == 0:
                    self.clearance_started_at = time.monotonic()
                self.clearance_counter += 1

                now = time.monotonic()
                if (
                    self.clearance_counter
                    >= self.AMBULANCE_CLEARANCE_FRAMES
                ):
                    elapsed = now - self.clearance_started_at
                    clearance_message = (
                        f"\n🏥 Emergency cleared | "
                        f"Lane {self.lane_id} | "
                        f"clear_count={self.clearance_counter}/"
                        f"{self.AMBULANCE_CLEARANCE_FRAMES} | "
                        f"elapsed={elapsed:.2f}s"
                    )
                    self.ambulance_detected = False
                    self.clearance_counter = 0
                    self.clearance_started_at = None
                elif now - self.last_emergency_diagnostic_at >= 1.0:
                    elapsed = now - self.clearance_started_at
                    print(
                        f"Emergency Lane {self.lane_id} | "
                        f"ambulance confidence={confidence:.2f} | "
                        f"clear_count={self.clearance_counter}/"
                        f"{self.AMBULANCE_CLEARANCE_FRAMES} | "
                        f"clear_elapsed={elapsed:.2f}s"
                    )
                    self.last_emergency_diagnostic_at = now
            elif self.ambulance_detected:
                now = time.monotonic()
                if now - self.last_emergency_diagnostic_at >= 1.0:
                    print(
                        f"Emergency Lane {self.lane_id} | "
                        f"ambulance confidence={confidence:.2f} | "
                        f"clear_count=0/"
                        f"{self.AMBULANCE_CLEARANCE_FRAMES}"
                    )
                    self.last_emergency_diagnostic_at = now
            else:
                self.clearance_counter = 0

            is_detected = self.ambulance_detected

        if clearance_message is not None:
            print(clearance_message)

        return is_detected
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
            return []

        results = self.model(
            self.current_frame,
            conf=0.25,
            imgsz=960,
            classes=[2, 3, 5, 7],
            verbose=False
        )

        current_count = 0
        car_count = 0
        bike_count = 0
        bus_count = 0
        truck_count = 0
        annotations = []

        for result in results:
            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:

                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id not in self.vehicle_classes or conf < 0.25:
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

                class_name = self.vehicle_classes[cls_id]
                x_m, y_m = self.pixel_to_meters(u_center, v_bottom)
                mapped = (
                    x_m is not None
                    and y_m is not None
                    and np.isfinite(x_m)
                    and np.isfinite(y_m)
                )
                inside_zone = (
                    mapped and self.min_y <= y_m <= self.max_y
                )

                if inside_zone:
                    current_count += 1

                    if class_name == "car":
                        car_count += 1
                    elif class_name == "bike":
                        bike_count += 1
                    elif class_name == "bus":
                        bus_count += 1
                    elif class_name == "truck":
                        truck_count += 1

                    box_color = (0, 255, 0)
                    label = f"{class_name} {conf:.2f} | {y_m:.1f}m"
                elif mapped:
                    box_color = (100, 100, 100)
                    label = f"{class_name} {conf:.2f} | outside zone"
                else:
                    box_color = (0, 165, 255)
                    label = f"{class_name} {conf:.2f} | unmapped"

                cv2.rectangle(
                    self.current_frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2
                )
                cv2.putText(
                    self.current_frame,
                    label,
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box_color,
                    1
                )
                annotations.append(
                    (x1, y1, x2, y2, box_color, label)
                )

        weighted_count = (
            bike_count * self.vehicle_weights["bike"] +
            car_count * self.vehicle_weights["car"] +
            bus_count * self.vehicle_weights["bus"] +
            truck_count * self.vehicle_weights["truck"]
        )

        with self.statistics_lock:
            self.detection_history.append(current_count)

            self.vehicle_count = int(
                np.mean(self.detection_history)
            )

            self.car_count = car_count
            self.bike_count = bike_count
            self.bus_count = bus_count
            self.truck_count = truck_count
            self.weighted_count = weighted_count

            self.congestion_score = (
                weighted_count / self.road_width
                if self.road_width > 0
                else 0
            )

        return annotations
        
    def read_frame(self, visible=True):
        """Read the next playback frame and save it when this lane is active."""
        if self.is_eof:
            return True

        ret, frame = self.cap.read()
        if not ret:
            self.is_eof = True
            return True

        # Update display_frame only if this lane is currently 'active' visually
        if visible or self.display_frame is None:
            self.display_frame = frame

        return True

    def prime_display_frame(self):
        """Load a still frame so every dashboard panel is visible at startup."""
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(
                f"Unable to read initial display frame for lane {self.lane_id}"
            )
        self.display_frame = frame

    def reset_video(self):
        """Restart the normal video stream."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.is_eof = False
        self.current_frame = None
        self.display_frame = None
    
    def get_frame(self):
        """Return this lane's persistent display frame without detection boxes."""
        return self.display_frame

class TrafficSignalController:
    """Manages 4-lane traffic signal logic"""
    
    def __init__(self):
        self.lanes = None
        self.current_green_lane = 0
        self.signal_start_time = 0
        self.signal_duration = 10
        self.time_remaining = 10
        # One shared emergency model for all four lanes
        self.emergency_model = YOLO("ambulance_best.pt")
        self.emergency_state_lock = threading.Lock()

# Emergency priority state
        self.emergency_mode = False
        self.emergency_lane_id = -1
        self.signal_initialized = False
        self.next_normal_lane = None
        
        self.preanalysis_lane = None
        self.preanalysis_duration = None
        self.last_preanalysis_log = None
        self.preanalysis_request_id = None
        self.preanalysis_completed_phase = None
        self.preanalysis_weighted_count = None
        self.preanalysis_congestion_score = None

        # Green duration based on the selected lane's weighted vehicle count
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
                lane = LaneProcessor(
                    path,
                    i + 1,
                    self.emergency_model,
                    self.emergency_state_lock
                )
                self.lanes.append(lane)
                print(f"✅ Loaded Lane {i+1}: {path}")
            else:
                print(f"❌ Lane {i+1} video not found: {path}")
                # Create dummy lane if video missing
                self.lanes.append(
    LaneProcessor(
        0,
        i + 1,
        self.emergency_model,
        self.emergency_state_lock
    )
)
    
    def get_signal_duration(self, weighted_count):
        """Calculate green time from the selected lane's weighted traffic."""
        for (min_count, max_count), duration in self.signal_durations.items():
            if min_count <= weighted_count < max_count:
                return duration
        return 30
    
    def select_initial_lane(self):
        """Choose the startup lane with the highest calibrated congestion."""
        def get_congestion(lane_id):
            lane = self.lanes[lane_id]
            with lane.statistics_lock:
                return lane.congestion_score

        return max(
            range(len(self.lanes)),
            key=get_congestion
        )

    def select_next_lane(self):
        """Advance normal control in round-robin order."""
        if self.next_normal_lane is not None:
            self.switch_green_lane(self.next_normal_lane)

    def prepare_next_lane(self):
        lane_id = self.next_normal_lane
        if lane_id is None:
            return
        lane = self.lanes[lane_id]
        phase = (self.current_green_lane, self.signal_start_time)

        if self.last_preanalysis_log != phase:
            self.preanalysis_weighted_count = None
            self.preanalysis_congestion_score = None
            self.preanalysis_lane = lane_id
            self.preanalysis_duration = None
            self.preanalysis_completed_phase = None
            self.preanalysis_request_id = lane.request_traffic_analysis()
            self.last_preanalysis_log = phase
            print(
                f"PRE-ANALYSIS STARTED: Lane {lane_id + 1} "
                f"(next round-robin lane; "
                f"{self.time_remaining:.1f}s remaining)"
            )

        if (
            lane.is_traffic_analysis_complete(self.preanalysis_request_id)
            and self.preanalysis_completed_phase != phase
        ):
            with lane.statistics_lock:
                weighted_count = lane.weighted_count
                congestion_score = lane.congestion_score
            self.preanalysis_weighted_count = weighted_count
            self.preanalysis_congestion_score = congestion_score
            self.preanalysis_duration = self.get_signal_duration(
                weighted_count
            )
            print(
                f"PRE-ANALYSIS COMPLETE: Lane {lane_id + 1} | "
                f"weighted traffic = {weighted_count} | "
                f"congestion = {congestion_score:.2f} | "
                f"prepared green time = "
                f"{self.preanalysis_duration} seconds"
            )
            self.preanalysis_completed_phase = phase
    
    def switch_green_lane(self, lane_id, initial=False):
        """Switch to new green lane"""
        if self.emergency_mode:
            self.current_green_lane = self.emergency_lane_id
            return

        previous_phase = (self.current_green_lane, self.signal_start_time)
        prepared_duration = (
            self.preanalysis_duration
            if (
                self.preanalysis_lane == lane_id
                and self.preanalysis_completed_phase == previous_phase
            )
            else None
        )

        self.current_green_lane = lane_id
        self.signal_initialized = True
        self.next_normal_lane = (lane_id + 1) % len(self.lanes)
        self.preanalysis_lane = None
        self.preanalysis_duration = None
        self.last_preanalysis_log = None
        self.preanalysis_request_id = None
        self.preanalysis_completed_phase = None
        self.preanalysis_weighted_count = None
        self.preanalysis_congestion_score = None

        lane_traffic = []
        for lane in self.lanes:
            with lane.statistics_lock:
                lane_traffic.append(
                    (lane.weighted_count, lane.congestion_score)
                )

        weighted_count, congestion_score = lane_traffic[lane_id]
        self.signal_duration = (
            prepared_duration
            if prepared_duration is not None
            else self.get_signal_duration(weighted_count)
        )
        self.signal_start_time = time.time()
        self.time_remaining = self.signal_duration

        print("\nNORMAL SIGNAL DECISION")
        for index, (traffic, congestion) in enumerate(lane_traffic):
            print(
                f"Lane {index + 1}: traffic/congestion = "
                f"{traffic} weighted vehicles / {congestion:.2f}"
            )
        if initial:
            print("INITIAL TRAFFIC")
            for index, (traffic, congestion) in enumerate(lane_traffic):
                print(
                    f"Lane {index + 1} = weighted {traffic}, "
                    f"congestion {congestion:.2f}"
                )
            print(f"INITIAL SELECTED LANE = Lane {lane_id + 1}")
            print(f"GREEN TIME = {self.signal_duration} seconds")
        print(f"Selected lane = Lane {lane_id + 1}")
        print(f"Weighted traffic = {weighted_count}")
        print(f"Green time = {self.signal_duration} seconds")
        print(
            f"\n🟢 GREEN → LANE {lane_id + 1} | "
            f"Weighted vehicles: {weighted_count} | "
            f"Congestion: {congestion_score:.2f} | "
            f"Time: {self.signal_duration}s"
        )
    
    def update_signal(self):
        """Update normal signal timing and emergency priority."""

        # Check whether any lane has a confirmed ambulance
        emergency_lane = -1

        with self.emergency_state_lock:
            emergency_lane = next(
                (
                    i for i, lane in enumerate(self.lanes)
                    if lane.ambulance_detected
                ),
                -1
            )

    # Emergency priority
        if emergency_lane != -1:
            if (
                not self.emergency_mode
                or self.emergency_lane_id != emergency_lane
            ):
                print(
                    f"\n🚦 EMERGENCY OVERRIDE → "
                    f"Lane {emergency_lane + 1}"
                )

                self.emergency_mode = True
                self.emergency_lane_id = emergency_lane

                # Keep emergency lane green.
                self.signal_duration = 999999
                self.signal_start_time = time.time()
                self.time_remaining = 999999

            self.current_green_lane = emergency_lane
            return

    # Emergency cleared
        if self.emergency_mode:
            self.emergency_mode = False
            self.emergency_lane_id = -1

            print("🏥 Emergency cleared; resuming normal adaptive control.")
            self.select_next_lane()
            return

# Normal signal timing
        current_time = time.time()

        self.time_remaining = max(
        0,
        self.signal_duration -
        (current_time - self.signal_start_time)
    )

        if self.time_remaining <= 5:
            self.prepare_next_lane()

        if self.time_remaining <= 0:
            current_phase = (
                self.current_green_lane,
                self.signal_start_time
            )
            if (
                self.preanalysis_lane != self.next_normal_lane
                or self.preanalysis_completed_phase != current_phase
                or self.preanalysis_duration is None
            ):
                return
            self.select_next_lane()
    
class FourLaneTrafficSystem:
    """Main 4-lane traffic management system"""
    
    def __init__(self, video_paths):
        self.controller = TrafficSignalController()
        self.controller.load_lanes(video_paths)
        self.running = True
        self.emergency_stop_event = threading.Event()
        self.emergency_thread = None
        
        # Window names
        self.window_names = [f"Lane_{i+1}" for i in range(4)]
        
    def create_windows(self):
        """Create single dashboard window"""
        cv2.namedWindow("AI Smart Traffic Dashboard", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AI Smart Traffic Dashboard", 1920, 1080)
    
    def process_all_lanes(self):
        """Process normal traffic frames for the four-lane dashboard."""

        if len(self.controller.lanes) != 4:
            raise RuntimeError(
                f"Expected 4 lane processors, found {len(self.controller.lanes)}"
            )

        for lane in self.controller.lanes:
            lane.check_traffic_analysis_error()

        frames = []
        green_lane = self.controller.current_green_lane
        lane = self.controller.lanes[green_lane]
        if lane.is_eof:
            lane.reset_video()
        lane.read_frame(visible=True)
        if lane.is_eof:
            lane.reset_video()
            lane.read_frame(visible=True)

        for lane in self.controller.lanes:
            frame = lane.get_frame()
            if frame is None:
                frame = np.zeros((450, 800, 3), dtype=np.uint8)

            frames.append(frame)

        if len(frames) != 4:
            raise RuntimeError(
                f"Expected exactly 4 lane frames, received {len(frames)}"
            )

        return frames, True

    def monitor_emergencies(self):
        """Scan every dedicated emergency stream outside the display loop."""
        while not self.emergency_stop_event.is_set():
            for lane in self.controller.lanes:
                if self.emergency_stop_event.is_set():
                    break

                ret, frame = lane.emergency_cap.read()
                if not ret:
                    lane.emergency_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = lane.emergency_cap.read()

                if ret:
                    lane.detect_emergency(frame)
                elif self.emergency_stop_event.wait(0.01):
                    break
    
    def display_frames(self, frames):
        """Display all 4 lanes in one dashboard"""
        if len(frames) != 4:
            raise ValueError(
                f"Expected exactly 4 lane frames, received {len(frames)}"
            )

        processed = []

        for i, frame in enumerate(frames):

            if frame is None:
                frame = np.zeros((450, 800, 3), dtype=np.uint8)

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
            with lane.statistics_lock:
                vehicle_count = lane.vehicle_count
                road_width = lane.road_width
                congestion_score = lane.congestion_score

            # Bottom information panel
            cv2.rectangle(frame, (0,390), (800,450), (30,30,30), -1)

            cv2.putText(frame,
                        f"Vehicles : {vehicle_count}",
                        (20,415),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,255,0),
                        2)

            cv2.putText(frame,
                        f"Width : {road_width}",
                        (260,415),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255,255,255),
                        2)

            cv2.putText(frame,
                        f"Congestion : {congestion_score:.2f}",
                        (420,415),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,255,255),
                        2)

            if (
                self.controller.emergency_mode
                and i == self.controller.current_green_lane
            ):
                color = (0,255,0)
                status = "GREEN"
            elif not self.controller.signal_initialized:
                color = (0,0,255)
                status = "RED"
            elif i == self.controller.current_green_lane:
                color = (0,255,0)
                status = "GREEN"
            elif (
                not self.controller.emergency_mode
                and self.controller.time_remaining <= 5
                and i == self.controller.preanalysis_lane
            ):
                color = (0,255,255)
                status = "NEXT GREEN"
            else:
                color = (0,0,255)
                status = "RED"

            cv2.circle(frame,(740,22),14,color,-1)

            cv2.putText(frame,
                        status,
                        (565 if status == "NEXT GREEN" else 610,32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6 if status == "NEXT GREEN" else 0.7,
                        color,
                        2)

            if (
                self.controller.emergency_mode
                and i == self.controller.current_green_lane
            ):
                timer_text = "EMERGENCY"
                cv2.putText(frame,
                            timer_text,
                            (620,440),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0,255,255),
                            2)
            elif (
                self.controller.signal_initialized
                and i == self.controller.current_green_lane
            ):
                timer_text = (
                    f"{self.controller.time_remaining:.1f}s"
                )
                cv2.putText(frame,
                            timer_text,
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
        self.emergency_thread = threading.Thread(
            target=self.monitor_emergencies,
            name="emergency-monitor",
            daemon=True
        )
        self.emergency_thread.start()

        try:
            for lane in self.controller.lanes:
                lane.prime_display_frame()

            for lane in self.controller.lanes:
                lane.start_traffic_analysis()
                lane.request_traffic_analysis()

            while (
                not all(
                    lane.traffic_measurement_ready
                    for lane in self.controller.lanes
                )
                or self.controller.emergency_mode
            ):
                for lane in self.controller.lanes:
                    lane.check_traffic_analysis_error()
                self.controller.update_signal()

                if self.controller.emergency_mode:
                    frames, _ = self.process_all_lanes()
                else:
                    frames = [
                        lane.get_frame()
                        for lane in self.controller.lanes
                    ]
                self.display_frames(frames)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
        except Exception:
            self.cleanup()
            raise

        if not self.running:
            self.cleanup()
            return

        if not self.controller.signal_initialized:
            initial_lane = self.controller.select_initial_lane()
            self.controller.switch_green_lane(initial_lane, initial=True)

        fps_counter = 0
        fps_time = time.time()
        
        while self.running:
            # Update signal logic
            self.controller.update_signal()
            
            # Display frames
            frames, all_active = self.process_all_lanes()
            self.display_frames(frames)
            
            # FPS counter
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps = 30 / (time.time() - fps_time)
                timer_status = (
                    "EMERGENCY"
                    if self.controller.emergency_mode
                    else f"T: {self.controller.time_remaining:.1f}s"
                )
                print(f"FPS: {fps:.1f} | Green: L{self.controller.current_green_lane+1} | "
                      f"{timer_status}")
                fps_time = time.time()
            
            # Handle key events (works across all windows)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('r'):
                for lane in self.controller.lanes:
                    with lane.statistics_lock:
                        lane.vehicle_count = 0
                        lane.detection_history.clear()
                print("🔄 All counts reset!")
            elif key == ord('n'):
                self.controller.select_next_lane()
        
        self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.emergency_stop_event.set()
        if self.emergency_thread is not None:
            self.emergency_thread.join()

        for lane in self.controller.lanes:
            lane.stop_vehicle_detection()
            lane.cap.release()
            lane.traffic_cap.release()
            lane.emergency_cap.release()
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
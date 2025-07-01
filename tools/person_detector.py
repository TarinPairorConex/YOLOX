# --- Refactored Person Detection and Tracking Script ---
import time
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import os
import argparse

# YOLOX specific imports
from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess

# Import our refactored modules
from tracker import PersonTracker, assign_trackers, analyze_entrance_exit_pattern, get_direction_name
from evaluator import VideoFrameEvaluator
from motion_detector import MotionBrightnessDetector
from bed_occupancy_kalman_filter import BedOccupancyKalmanFilter

from trend_analyzer import MovementTrendAnalyzer

class YOLOXPredictor:
    """YOLOX model for object detection"""
    
    def __init__(self, model_path, model_name="yolox-nano", conf_threshold=0.3, 
                 nms_threshold=0.45, input_size_wh=(416, 416), device="cpu"):
        """
        Initialize YOLOX predictor.
        
        Args:
            model_path: Path to the model weights
            model_name: Name of the YOLOX model
            conf_threshold: Confidence threshold for detections
            nms_threshold: NMS threshold for detections
            input_size_wh: Input size (width, height) for the model
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        # input_size_for_preproc is (HEIGHT, WIDTH) as expected by ValTransform target_size
        self.input_size_for_preproc = (input_size_wh[1], input_size_wh[0])
        self.device = torch.device(device)
        
        # Initialize YOLOX model
        self.exp = get_exp(exp_file=None, exp_name=model_name)
        self.exp.num_classes = 1
        self.exp.class_names = ["person"]
        self.exp.test_size = self.input_size_for_preproc
        self.exp.test_conf = conf_threshold
        self.exp.nmsthre = nms_threshold
        
        self.model = self.exp.get_model()
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model weights
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        model_state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
        self.model.load_state_dict(model_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize preprocessing
        self.preproc = ValTransform(legacy=False)
        print(f"YOLOXPredictor initialized. Target preproc size (H,W): {self.input_size_for_preproc}")

        self.contour_area_trend_analyzer = MovementTrendAnalyzer(window_size=30)  # Initialize trend analyzer with a window size of 30 frames
    
    def inference(self, frame):
        """
        Run inference on a frame.
        
        Args:
            frame: Input frame
            
        Returns:
            tuple: (detections, ratio)
                detections: YOLOX detections tensor or None
                ratio: Preprocessing ratio for scaling bounding boxes
        """
        if frame is None or frame.size == 0:
            print("ERROR: Input frame is empty or invalid.")
            return None, 1.0
        
        # Preprocess frame
        img_for_preproc, ratio_from_preproc = self.preproc(frame, None, self.input_size_for_preproc)
        
        # Validate ratio
        if not isinstance(ratio_from_preproc, (float, tuple)):
            # Calculate ratio manually if preproc output is bad
            original_h, original_w = frame.shape[:2]
            target_h, target_w = self.input_size_for_preproc
            r = min(target_h / original_h, target_w / original_w)
            ratio_to_use = r
        elif isinstance(ratio_from_preproc, tuple):
            ratio_to_use = ratio_from_preproc[0]
        else:
            ratio_to_use = float(ratio_from_preproc)
        
        if ratio_to_use == 0:
            print("ERROR: Calculated ratio is zero. Defaulting to 1.0.")
            ratio_to_use = 1.0
        
        # Run inference
        img_tensor = torch.from_numpy(img_for_preproc).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            processed_outputs = postprocess(
                outputs, self.exp.num_classes, self.conf_threshold,
                self.nms_threshold, class_agnostic=False
            )
        
        return processed_outputs[0] if processed_outputs else None, ratio_to_use
    
    def process_detections(self, detections_tensor, ratio):
        """
        Process YOLOX detections into a format suitable for tracking.
        
        Args:
            detections_tensor: YOLOX detections tensor
            ratio: Preprocessing ratio for scaling bounding boxes
            
        Returns:
            list: List of detection dictionaries with 'bbox' and 'centroid' keys
        """
        if detections_tensor is None:
            return []
        
        detections_np = detections_tensor.cpu().numpy()
        
        if ratio == 0:
            bboxes_scaled = detections_np[:, 0:4].copy()
        else:
            bboxes_scaled = detections_np[:, 0:4] / ratio
        
        scores = detections_np[:, 4]
        cls_ids = detections_np[:, 6] if detections_np.shape[1] > 6 else np.zeros(detections_np.shape[0], dtype=int)
        
        if detections_np.shape[1] > 5:
            scores = detections_np[:, 4] * detections_np[:, 5]  # obj_conf * cls_conf
        
        processed_detections = []

        
        for i in range(len(bboxes_scaled)):
            if cls_ids[i] == 0 and scores[i] >= self.conf_threshold:  # Class 0 for person
                bbox = tuple(map(int, bboxes_scaled[i]))  # (x1, y1, x2, y2)
                centroid = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
                processed_detections.append({
                    'bbox': bbox,
                    'centroid': centroid,
                    'score': float(scores[i]),
                    'class_id': int(cls_ids[i])
                })
        
        return processed_detections
    
    def visualize(self, frame, detections, draw_scores=True):
        """
        Visualize detections on a frame.
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            draw_scores: Whether to draw confidence scores
            
        Returns:
            numpy.ndarray: Frame with visualized detections
        """
        result_frame = frame.copy()
        
        # Define colors for visualization
        colors = np.array([
            [0.000, 0.447, 0.741],
            [0.850, 0.325, 0.098],
            [0.929, 0.694, 0.125],
            [0.494, 0.184, 0.556],
            [0.466, 0.674, 0.188],
            [0.301, 0.745, 0.933]
        ]) * 255
        
        for det in detections:
            bbox = det['bbox']
            score = det.get('score', 1.0)
            class_id = det.get('class_id', 0)
            
            # Get color for this class
            color = colors[class_id % len(colors)].astype(np.uint8).tolist()
            
            # Draw bounding box
            cv2.rectangle(result_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label with score
            if draw_scores:
                label = f"person: {score:.2f}"
                txt_color = (0, 0, 0) if np.mean(colors[class_id % len(colors)]) > 0.5 else (255, 255, 255)
                txt_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                
                cv2.rectangle(
                    result_frame, 
                    (bbox[0], bbox[1] - txt_size[1] - 4), 
                    (bbox[0] + txt_size[0] + 4, bbox[1]), 
                    color, 
                    -1
                )
                cv2.putText(
                    result_frame, 
                    label, 
                    (bbox[0] + 2, bbox[1] - 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, 
                    txt_color, 
                    1
                )
        
        return result_frame


class PersonDetectorTracker:
    """Main class for person detection and tracking"""
    
    def __init__(self, config):
        """
        Initialize the person detector and tracker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize YOLOX predictor
        self.predictor = YOLOXPredictor(
            model_path=config['model_path'],
            model_name=config['model_name'],
            conf_threshold=config['conf_threshold'],
            nms_threshold=config['nms_threshold'],
            input_size_wh=config['input_size_wh'],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize bed polygons
        self.bed_polygons = {
            'outer': np.array(config['outer_bed_points'], dtype=np.int32),
            'inner': np.array(config['inner_bed_points'], dtype=np.int32)
        }
        
        # Initialize trackers
        self.active_trackers = {}
        self.next_tracker_id = 0
        
        # Initialize evaluator
        self.evaluator = VideoFrameEvaluator()
        
        # Tracking method
        self.tracking_method = config['tracking_method']  # 'euclidean' or 'iou'
        self.bed_detection_method = config['bed_detection_method']  # 'center_point' or 'iou'
        
        # Initialize motion and brightness detector
        motion_config = {
            'bed_rectangle_points': config['bed_rectangle_points'],
            'brightness_threshold': config.get('brightness_threshold', 200),
            'darkness_threshold': config.get('darkness_threshold', 30),
            'min_blob_area': config.get('min_blob_area', 50),
            'min_dark_blob_area': config.get('min_dark_blob_area', 50),
            'very_bright_threshold': config.get('very_bright_threshold', 250),
            'movement_threshold': config.get('movement_threshold', 400),
            'movement_countdown_frames': config.get('movement_countdown_frames', 30)
        }
        self.motion_detector = MotionBrightnessDetector(motion_config)
        
        # Detection method for determining if person is in bed
        self.detection_method = config.get('detection_method', 'tracking')  # 'tracking', 'motion', or 'combined'
        
        # Initialize bed occupancy counter
        self.bed_occupancy_count = 0
        self.entrance_tracker = {}  # Track which person IDs have triggered entrance events
        self.exit_tracker = {}      # Track which person IDs have triggered exit events

        self.did_exit_occur_recently = False  # Track if an exit event occurred recently
        self.did_entrance_occur_recently = False  # Track if an entrance event occurred recently
        self.last_exit_frame = None  # Track the last frame when an exit event occurred
        self.last_entrance_frame = None  # Track the last frame when an entrance event occurred
        
        # Initialize Kalman filter for bed occupancy estimation
        self.kalman_filter = BedOccupancyKalmanFilter(
            process_noise=config.get('kalman_process_noise', 0.01),
            measurement_noise_person=config.get('kalman_measurement_noise_person', 0.1),
            measurement_noise_motion=config.get('kalman_measurement_noise_motion', 0.001)
        )

        # Initialize MovementTrendAnalyzer for contour area trend analysis
        self.contour_area_trend_analyzer = MovementTrendAnalyzer(window_size=30)
        
        self.motion_display_counter = 0  # Add this line
        self.motion_display_duration = 30  # Number of frames to keep motion "on screen"

        # Bed occupancy probability from Kalman filter
        self.bed_occupancy_probability = 0.0
        
        print(f"PersonDetectorTracker initialized with:")
        print(f"  - Tracking method: {self.tracking_method}")
        print(f"  - Bed detection method: {self.bed_detection_method}")
        print(f"  - Detection method: {self.detection_method}")
    
    def process_frame(self, frame, frame_number, video_path, flip_frame=True):
        """
        Process a single frame.
        
        Args:
            frame: Input frame
            frame_number: Frame number
            video_path: Path to the video file
            flip_frame: Whether to flip the frame vertically
            
        Returns:
            tuple: (display_frame, people_in_bed_count, ground_truth)
                display_frame: Frame with visualizations
                people_in_bed_count: Number of people detected on the bed
                ground_truth: Ground truth value (1 if person should be on bed, 0 otherwise)
        """
        # Get ground truth
        _, _, ground_truth = self.evaluator.is_person_on_bed(video_path, frame_number)
        
        # Make a copy of the original frame for display
        display_frame = frame.copy()
        
        # Run YOLOX inference on the original frame
        yolox_detections_tensor, ratio = self.predictor.inference(display_frame)
        
        # Process detections
        detections = self.predictor.process_detections(yolox_detections_tensor, ratio)
        
        # def get_thermal_peak_in_bbox(frame, bbox):
        #         x1, y1, x2, y2 = bbox
        #         roi = frame[y1:y2, x1:x2]
        #         thermal_peak = np.max(roi)
        #         return thermal_peak
        

        # for det in detections:
        #     thermal_peak = get_thermal_peak_in_bbox(display_frame, det['bbox'])
        #     if thermal_peak < 254:
        #         detections.remove(det)
                

        # Flip the frame if needed
        if flip_frame:
            display_frame = cv2.flip(display_frame, 0)
            
            # Adjust detection coordinates for flipped frame
            frame_height = display_frame.shape[0]
            for det in detections:
                # Flip bounding box coordinates
                x1, y1, x2, y2 = det['bbox']
                det['bbox'] = (x1, frame_height - y2, x2, frame_height - y1)
                
                # Flip centroid coordinates
                cx, cy = det['centroid']
                det['centroid'] = (cx, frame_height - cy)
        
        # Update trackers
        current_detections = [det for det in detections]
        
        # Assign detections to trackers
        if current_detections:
            assignments, self.next_tracker_id = assign_trackers(
                current_detections,
                self.active_trackers,
                self.tracking_method,
                max_distance=self.config['max_association_distance'],
                min_iou=self.config['min_iou_association'],
                next_id=self.next_tracker_id,
                history_length=self.config['history_length']
            )
            
            # Update assigned trackers
            for det in current_detections:
                centroid = det['centroid']
                assignment_key = tuple(map(int, centroid))
                
                if assignment_key in assignments:
                    tracker_id = assignments[assignment_key]
                    if tracker_id in self.active_trackers:
                        # Update tracker with detection
                        self.active_trackers[tracker_id].update_state(
                            det,
                            self.bed_polygons,
                            self.config['movement_threshold'],
                            self.config['min_movement']
                        )
                        
                        # Analyze entrance/exit patterns and update bed occupancy count
                        entrance_detected, exit_detected = analyze_entrance_exit_pattern(self.active_trackers[tracker_id])
                        
                        # Update bed occupancy count based on entrance/exit events
                        self._update_bed_occupancy_count(tracker_id, entrance_detected, exit_detected, frame_number)
        
        # Update unassigned trackers
        current_frame_assigned_ids = set(assignments.values()) if 'assignments' in locals() else set()
        

        
        for tracker_id in list(self.active_trackers.keys()):
            if tracker_id not in current_frame_assigned_ids:
                self.active_trackers[tracker_id].update_state(
                    None,
                    self.bed_polygons,
                    self.config['movement_threshold'],
                    self.config['min_movement']
                )

        if self.last_exit_frame is not None:
            self.did_exit_occur_recently = (frame_number - self.last_exit_frame) <= 30  # or your threshold
        else:
            self.did_exit_occur_recently = False

        if self.last_entrance_frame is not None:
            self.did_entrance_occur_recently = (frame_number - self.last_entrance_frame) <= 30
        else:
            self.did_entrance_occur_recently = False


        # Remove trackers that haven't been detected for too long
        trackers_to_remove = []
        for tid, tracker in self.active_trackers.items():
            if tracker.frames_since_detected > self.config['max_undetected_frames']:
                trackers_to_remove.append(tid)
        
        for tid in trackers_to_remove:
            if tid in self.active_trackers:
                del self.active_trackers[tid]
                
        # Clean up entrance/exit trackers for removed trackers
        for tracker_id in list(self.entrance_tracker.keys()):
            if tracker_id not in self.active_trackers:
                if tracker_id in self.entrance_tracker:
                    del self.entrance_tracker[tracker_id]
                if tracker_id in self.exit_tracker:
                    del self.exit_tracker[tracker_id]
        

        # Process frame with motion and brightness detector
        motion_frame = display_frame.copy()
        motion_person_in_bed, motion_display_frame, motion_debug_info = self.motion_detector.process_frame(motion_frame)
        cv2.imshow("Motion Detection", motion_display_frame)
        
        #================ TREND ANALYSIS ====================
        def get_grayscale_roi_and_mask(frame, rectangle_points):
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.array(rectangle_points), 255)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)
            return roi, mask
        
        def extract_blob_data(binary_roi, min_area):
            """Extract blob data from connected components analysis"""
            if binary_roi is None:
                return []
            num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary_roi, connectivity=4)
            blob_data = []
            for i in range(1, num_labels):  # Start from 1 to skip background
                x, y, w, h, area = stats[i]
                cx, cy = centroids[i]
                if area > min_area:
                    blob_data.append({
                        'centroid': (cx, cy),
                        'area': area,
                        'bbox': (x, y, w, h),
                        'stats': stats[i],
                    })
            return blob_data
        roi, _ = get_grayscale_roi_and_mask(display_frame, self.config['outer_bed_points'])
        _, binary = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)
        bright_blob_data = extract_blob_data(binary, 50)
        area = 0
        if bright_blob_data:
            area = sum(blob['area'] for blob in bright_blob_data)
        self.contour_area_trend_analyzer.add_data_point(frame_number, area)
        contour_area_info = self.contour_area_trend_analyzer.get_trend_info()
 
        direction = contour_area_info['trend_direction']
        strength = contour_area_info['trend_strength']
        norm_variance = contour_area_info['normalized_variance']

        is_motion_detected = motion_debug_info.get('is_motion_detected', False)
        if is_motion_detected:
            self.contour_area_trend_analyzer = MovementTrendAnalyzer(window_size=30)
        
        if is_motion_detected:
            self.motion_display_counter = self.motion_display_duration  # Reset counter if motion detected
        elif self.motion_display_counter > 0:
            self.motion_display_counter -= 1  # Decrement if not detected

        # Reset motion counter if exit occurred recently
        if self.did_exit_occur_recently:
            self.motion_display_counter = 0

        motion_measured = self.motion_display_counter > 0  # True if motion was detected in the last few frames


        # Determine if it's a person under blanket or silhouette
        person_under_blanket = (direction == 'increasing' and strength < 10 and norm_variance < 0.05)
        silhouette_detected = (direction == 'decreasing' and strength < 10 and norm_variance < 0.05)

        trend_frame  = display_frame.copy()
        cv2.putText(trend_frame, f"Trend: {direction} ({strength:.1f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(trend_frame, f"Norm Variance: {norm_variance:.3f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(trend_frame, f"Motion Detected: {motion_measured}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Toggle display for silhouette or person under blanket
        if person_under_blanket:
            status_text = "Person Under Blanket"
            status_color = (0, 255, 0)
        elif silhouette_detected:
            status_text = "Silhouette Detected"
            status_color = (0, 165, 255)
        else:
            status_text = "No Blanket/Silhouette"
            status_color = (255, 255, 255)
        cv2.putText(trend_frame, status_text, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        cv2.imshow("Contour Area Trend", trend_frame)




        # ============ BED OCCUPANCY DETECTION ==============

        # Determine if a person is in bed based on the selected detection method
        tracking_person_in_bed = sum(1 for t in self.active_trackers.values() if t.is_relevant_to_bed) > 0


        # --- ALERT for bbox overlapping bed but not classified as in bed ---
        overlapping_bboxes_not_in_bed = []
        for det in detections:
            bbox = det['bbox']
            # Compute IOU with outer bed polygon's bbox
            bed_bbox = (
                min([p[0] for p in self.config['outer_bed_points']]),
                min([p[1] for p in self.config['outer_bed_points']]),
                max([p[0] for p in self.config['outer_bed_points']]),
                max([p[1] for p in self.config['outer_bed_points']])
            )
            # Calculate IOU
            x1_inter = max(bbox[0], bed_bbox[0])
            y1_inter = max(bbox[1], bed_bbox[1])
            x2_inter = min(bbox[2], bed_bbox[2])
            y2_inter = min(bbox[3], bed_bbox[3])
            width_inter = max(0, x2_inter - x1_inter)
            height_inter = max(0, y2_inter - y1_inter)
            area_inter = width_inter * height_inter
            area_bbox = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            area_bed = (bed_bbox[2] - bed_bbox[0]) * (bed_bbox[3] - bed_bbox[1])
            area_union = area_bbox + area_bed - area_inter
            iou = area_inter / area_union if area_union > 0 else 0.0

            # If IOU > 0 but not classified as in bed, collect this bbox
            if iou > 0:
                # Check if this detection is NOT tracked as in bed
                is_tracked_in_bed = False
                for t in self.active_trackers.values():
                    if t.last_bbox == bbox and t.is_relevant_to_bed:
                        is_tracked_in_bed = True
                        break
                if not is_tracked_in_bed:
                    overlapping_bboxes_not_in_bed.append(bbox)

        is_one_person_overlapping_bed = len(overlapping_bboxes_not_in_bed) == 1
        # Trigger alert only if exactly one such bbox
        if is_one_person_overlapping_bed:
            cv2.putText(    
                display_frame,
                "ALERT: Person overlaps bed but not classified as in bed!",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )


        # update the motion in bed
        motion_person_in_bed = motion_person_in_bed and not is_one_person_overlapping_bed

        if self.detection_method == 'tracking':
            # Use only tracking-based detection
            people_in_bed_count = 1 if tracking_person_in_bed else 0
        elif self.detection_method == 'motion':
            # Use only motion-based detection
            people_in_bed_count = 1 if (motion_person_in_bed) else 0
        else:  # 'combined'
            # Use both methods (OR logic)
            people_in_bed_count = 1 if (tracking_person_in_bed and not silhouette_detected) or motion_person_in_bed else 0
                                        # (person_under_blanket and not silhouette_detected)) else 0
        # Update Kalman filter with measurements
        person_measurement = 1.0 if (tracking_person_in_bed and not silhouette_detected) else 0.0
        # person_measurement =  1.0 if self.bed_occupancy_count > 0 else 0.0

        # Convert motion detector debug info to a probability
        motion_measurement = 0.0
        brightness_measurement = 0.0

        if motion_debug_info:
            # Normalize movement level to 0-1 range
            # motion_measurement = motion_debug_info.get('is_motion_detected', 0)
            motion_measurement = motion_measured
    
            # Normalize very bright pixels to 0-1 range
            very_bright_pixels = motion_debug_info.get('very_bright_pixels', 0)
            very_bright_threshold = 600  # Threshold used in motion_detector.py
            normalized_brightness = min(1.0, very_bright_pixels / very_bright_threshold)

            # Combine movement and brightness
            brightness_measurement = normalized_brightness

        did_exit_occur_recently_measurement = 1.0 if self.did_exit_occur_recently else 0.0
        did_entrance_occur_recently_measurement = 1.0 if self.did_entrance_occur_recently else 0.0
        
        self.kalman_filter.predict()
        # Use custom measurement noises and importance order:
        # Highest importance: motion_measurement (almost guarantees occupancy)
        # Then: person_measurement, did_exit_occur_recently_measurement, brightness_measurement
        self.bed_occupancy_probability = self.kalman_filter.update_multi(
            [
            motion_measurement,                
            person_measurement,                
            1 - did_exit_occur_recently_measurement,
            did_entrance_occur_recently_measurement, 
            brightness_measurement             
            ],
            measurement_noises=[
            0.01,   
            0.1,
            0.1,
            0.1,
            0.1
            ]
        )

        # reset bed occupancy count if probability is low
        if self.bed_occupancy_probability < 0.5:
            self.bed_occupancy_count = 0

        # people_in_bed_count = int(self.bed_occupancy_probability > 0.8)  # Convert probability to binary count
        
        # Draw visualizations
        self._draw_visualizations(display_frame, frame_number, people_in_bed_count, 
                                 tracking_person_in_bed, motion_person_in_bed)
        
        return display_frame, people_in_bed_count, ground_truth
    
    def _update_bed_occupancy_count(self, tracker_id, entrance_detected, exit_detected, frame_number):
        """
        Update the bed occupancy count based on entrance/exit events
        
        Args:
            tracker_id: ID of the tracker
            entrance_detected: Whether an entrance was detected
            exit_detected: Whether an exit was detected
            
        Returns:
            bool: Whether the count was updated
        """
        count_updated = False
        
        # Check for entrance event
        if entrance_detected and tracker_id not in self.entrance_tracker:
            self.entrance_tracker[tracker_id] = True
            self.bed_occupancy_count += 1
            count_updated = True
            self.last_entrance_frame = frame_number
            print(f"Person {tracker_id} entered the bed. Count: {self.bed_occupancy_count}")
        
        # Check for exit event
        if exit_detected and tracker_id not in self.exit_tracker:
            self.exit_tracker[tracker_id] = True
            if self.bed_occupancy_count > 0:  # Prevent negative counts
                self.bed_occupancy_count -= 1
            count_updated = True

            self.last_exit_frame = frame_number
            print(f"Person {tracker_id} exited the bed. Count: {self.bed_occupancy_count}")
        
        return count_updated
    
    def _draw_visualizations(self, frame, frame_number, people_in_bed_count, 
                            tracking_person_in_bed=False, motion_person_in_bed=False):
        """
        Draw visualizations on the frame.
        
        Args:
            frame: Frame to draw on
            frame_number: Current frame number
            people_in_bed_count: Number of people detected on the bed
            tracking_person_in_bed: Whether tracking detected a person in bed
            motion_person_in_bed: Whether motion detection detected a person in bed
        """
        # Draw bed polygons
        cv2.polylines(frame, [self.bed_polygons['outer']], True, (0, 255, 0), 2)
        cv2.polylines(frame, [self.bed_polygons['inner']], True, (255, 0, 0), 2)
        
        # Draw frame info
        total_people_count = len(self.active_trackers)
        people_outside_bed_count = total_people_count - (1 if tracking_person_in_bed else 0)
        
        cv2.putText(frame, f"Frame: {int(frame_number)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Total People: {total_people_count}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw detection method results
        detection_text = f"In Bed: {people_in_bed_count}"
        if self.detection_method == 'combined':
            detection_text += f" (T:{int(tracking_person_in_bed)},M:{int(motion_person_in_bed)})"
        cv2.putText(frame, detection_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw bed occupancy count and Kalman filter estimate
        # Draw a semi-transparent background for the count
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 150), (350, 200), (0, 0, 0), -1)
        alpha = 0.6  # Transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw the count text
        # cv2.putText(
        #     frame,
        #     f"BED OCCUPANCY COUNT: {self.bed_occupancy_count}",
        #     (20, 170),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.6,
        #     (0, 255, 255),  # Yellow color
        #     1
        # )
        
        # Draw the Kalman filter estimate
        percentage = self.kalman_filter.get_occupancy_percentage()
        color = (0, 0, 255) if percentage < 50 else (0, 255, 0)  # Red if >50, else green
        cv2.putText(
            frame,
            f"BED OCCUPANCY PROBABILITY: {percentage}%",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1
        )
        
        # Draw tracker info
        y_offset_text = 85
        for tracker_id, tracker in self.active_trackers.items():
            if tracker.is_relevant_to_bed and tracker.last_valid_position:
                direction_name = get_direction_name(tracker.smoothed_direction)
                cv2.putText(
                    frame,
                    f"Person {tracker_id}: {direction_name} ({tracker.movement_magnitude:.1f}px)",
                    (10, y_offset_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1
                )
                y_offset_text += 15
        
        # Draw entrance/exit messages
        self._draw_entrance_exit_messages(frame)
        
        # Draw trackers
        for tracker in self.active_trackers.values():
            self._draw_tracker(frame, tracker)
    
    def _draw_entrance_exit_messages(self, frame):
        """Draw entrance/exit messages on the frame"""
        y_offset = 230
        for tracker in self.active_trackers.values():
            if tracker.state_message_timer > 0:
                message = ""
                msg_color = (255, 255, 255)
                
                if tracker.entrance_detected:
                    message = f"ENTRANCE DETECTED - Person {tracker.id}"
                    msg_color = (0, 255, 0)
                elif tracker.exit_detected:
                    message = f"EXIT DETECTED - Person {tracker.id}"
                    msg_color = (0, 0, 255)
                
                if message:
                    cv2.putText(
                        frame,
                        message,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        msg_color,
                        2
                    )
                    y_offset += 25
    
    def _draw_tracker(self, frame, tracker):
        """Draw tracker visualization on the frame"""
        if not tracker.last_valid_position:
            return
        
        pos = tracker.last_valid_position
        color = tracker.color
        
        if tracker.is_relevant_to_bed:
            circle_size = 12
            line_thickness = 3
            text_color = (0, 255, 0)
        else:
            circle_size = 8
            line_thickness = 2
            text_color = (255, 255, 255)
        
        # Draw position circle
        cv2.circle(frame, (int(pos[0]), int(pos[1])), circle_size, color, -1)
        cv2.circle(frame, (int(pos[0]), int(pos[1])), circle_size + 3, (255, 255, 255), 2)
        
        # Draw ID
        cv2.putText(
            frame,
            f"P{tracker.id}",
            (int(pos[0]) + 15, int(pos[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1
        )
        
        # Draw direction arrow
        if tracker.smoothed_direction is not None and tracker.movement_magnitude > self.config['min_movement']:
            arrow_length = min(max(tracker.movement_magnitude * 2, 15), 50)
            end_x = pos[0] + arrow_length * np.cos(np.radians(tracker.smoothed_direction))
            end_y = pos[1] - arrow_length * np.sin(np.radians(tracker.smoothed_direction))
            
            cv2.arrowedLine(
                frame,
                (int(pos[0]), int(pos[1])),
                (int(end_x), int(end_y)),
                color,
                line_thickness,
                tipLength=0.3
            )
        
        # Draw trail
        if len(tracker.position_history) > 1:
            trail_points = list(tracker.position_history)
            for i in range(1, len(trail_points)):
                if trail_points[i-1] and trail_points[i]:
                    alpha = i / len(trail_points)
                    trail_color = (
                        int(color[0] * alpha * 0.7),
                        int(color[1] * alpha * 0.7),
                        int(color[2] * alpha * 0.7)
                    )
                    
                    cv2.line(
                        frame,
                        (int(trail_points[i-1][0]), int(trail_points[i-1][1])),
                        (int(trail_points[i][0]), int(trail_points[i][1])),
                        trail_color,
                        line_thickness
                    )
        
        # Draw bounding box if available
        if tracker.last_bbox is not None:
            x1, y1, x2, y2 = tracker.last_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    
    def process_video(self, video_path, output_path=None, show_display=True, save_results=True):
        """
        Process a video file.
        
        Args:
            video_path: Path to the video file
            output_path: Path to save the output video
            show_display: Whether to show the display window
            save_results: Whether to save evaluation results
            
        Returns:
            dict: Evaluation results
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer if output path is provided
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Reset trackers
        self.active_trackers = {}
        self.next_tracker_id = 0
        
        # Start evaluation
        self.evaluator.start_evaluation(video_path)
        
        # Process frames
        frame_number = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            
            # Process frame
            display_frame, people_in_bed_count, ground_truth = self.process_frame(
                frame, frame_number, video_path
            )
            
            # Add frame result to evaluator
            prediction = 1 if people_in_bed_count > 0 else 0
            self.evaluator.add_frame_result(frame_number, prediction, ground_truth)
            
            # Write frame to output video
            if video_writer:
                video_writer.write(display_frame)
            
            # Show display
            if show_display:
                cv2.imshow("Person Detection and Tracking", display_frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Print progress
            if frame_number % 100 == 0 or frame_number == total_frames:
                print(f"Processed {frame_number}/{total_frames} frames ({frame_number/total_frames:.1%})")
        
        # End evaluation
        results = self.evaluator.end_evaluation()
        
        # Print summary
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"Video {video_name} processed:")
        print(f"  - Total frames: {frame_number}")
        print(f"  - Correct frames: {sum(results['correct'])}")
        print(f"  - Accuracy: {results['accuracy']:.2%}")
        
        # Save results
        if save_results:
            # Create results directory if it doesn't exist
            results_dir = os.path.join(os.path.dirname(video_path), "results")
            os.makedirs(results_dir, exist_ok=True)
            
            # Save visualization
            fig_path = os.path.join(results_dir, f"{video_name}_results.png")
            self.evaluator.visualize_results(video_name, fig_path, show=False)
            
            # Save frame errors visualization
            errors_path = os.path.join(results_dir, f"{video_name}_errors.png")
            self.evaluator.visualize_frame_errors(video_name, errors_path, show=False)
            
            print(f"Results saved to {results_dir}")
        
        # Release resources
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        return results
    
    def process_videos(self, video_paths, output_dir=None, show_display=True, save_results=True):
        """
        Process multiple video files.
        
        Args:
            video_paths: List of paths to video files
            output_dir: Directory to save output videos
            show_display: Whether to show the display window
            save_results: Whether to save evaluation results
            
        Returns:
            dict: Summary of evaluation results
        """
        all_results = {}
        
        for video_path in video_paths:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            print(f"\nProcessing video: {video_name}")
            
            # Create output path if output directory is provided
            output_path = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{video_name}_output.avi")
            
            # Process video
            results = self.process_video(
                video_path, output_path, show_display, save_results
            )
            
            if results:
                all_results[video_name] = results
        
        # Save overall results
        if save_results and all_results:
            # Create results directory if it doesn't exist
            results_dir = os.path.join(os.path.dirname(video_paths[0]), "results")
            os.makedirs(results_dir, exist_ok=True)
            
            # Save overall visualization
            fig_path = os.path.join(results_dir, "all_results.png")
            self.evaluator.visualize_results(None, fig_path, show=False)
            
            # Save results to file
            results_path = os.path.join(results_dir, "evaluation_results.json")
            self.evaluator.save_results(results_path)
            
            print(f"\nOverall results saved to {results_dir}")
        
        # Get summary
        summary = self.evaluator.get_summary()
        
        if summary:
            print("\nEvaluation Summary:")
            print(f"Total frames: {summary['overall']['total_frames']}")
            print(f"Total correct: {summary['overall']['total_correct']}")
            print(f"Overall accuracy: {summary['overall']['accuracy']:.2%}")
            
            print("\nAccuracy by video:")
            for video, metrics in summary['videos'].items():
                print(f"  - {video}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['frames']} frames)")
        
        return summary


    


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Person Detection and Tracking")
    parser.add_argument("--video", type=str, help="Path to a single video file")
    parser.add_argument("--videos", type=str, nargs="+", help="Paths to multiple video files")
    parser.add_argument("--output_dir", type=str, help="Directory to save output videos")
    parser.add_argument("--no_display", action="store_true", help="Don't show display window")
    parser.add_argument("--no_save", action="store_true", help="Don't save results")
    parser.add_argument("--tracking_method", type=str, default="iou", choices=["euclidean", "iou"], 
                        help="Tracking method (euclidean or iou)")
    parser.add_argument("--bed_detection_method", type=str, default="center_point", choices=["center_point", "iou"], 
                        help="Bed detection method (center_point or iou)")
    parser.add_argument("--detection_method", type=str, default="combined", choices=["tracking", "motion", "combined"],
                        help="Method to determine if person is in bed (tracking, motion, or combined)")
    parser.add_argument("--suppression_frames", type=int, default=90, 
                        help="Number of frames to suppress center point detection after exit")
    parser.add_argument("--movement_threshold", type=int, default=400,
                        help="Threshold for movement detection")
    parser.add_argument("--movement_countdown_frames", type=int, default=30,
                        help="Number of frames to maintain movement detection after movement stops")
    parser.add_argument("--kalman_process_noise", type=float, default=0.01,
                        help="Process noise for Kalman filter")
    parser.add_argument("--kalman_measurement_noise_person", type=float, default=0.1,
                        help="Measurement noise for person detector in Kalman filter")
    parser.add_argument("--kalman_measurement_noise_motion", type=float, default=0.2,
                        help="Measurement noise for motion detector in Kalman filter")
    args = parser.parse_args()
    
    # Define configuration
    config = {
        # YOLOX model configuration
        'model_path': r"C:\Users\Tairin Pairor\Documents\Github\Tarin%20Project\person-detection\models\best_ckpt 1.pth",
        'model_name': "yolox-nano",
        'conf_threshold': 0.6,
        'nms_threshold': 0.45,
        'input_size_wh': (416, 416),
        
        # Bed polygons
        'outer_bed_points': [(152, 4), (145, 145), (208, 151), (240, 10)],
        'inner_bed_points': [(173, 40), (214, 43), (202, 116), (162, 115)],
        'bed_rectangle_points': [(152, 4), (145, 145), (208, 151), (240, 10)],
        
        # Tracking configuration
        'tracking_method': args.tracking_method,
        'bed_detection_method': args.bed_detection_method,
        'detection_method': args.detection_method,
        'movement_threshold': 400,
        'min_movement': 3,
        'max_association_distance': 150,
        'min_iou_association': 0.3,
        'max_undetected_frames': 10,
        'history_length': 30,
        'center_point_suppression_frames': args.suppression_frames,
        
        # Motion detection configuration
        'brightness_threshold': 200,
        'darkness_threshold': 30,
        'min_blob_area': 50,
        'min_dark_blob_area': 50,
        'very_bright_threshold': 250,
        'movement_threshold': args.movement_threshold,
        'movement_countdown_frames': args.movement_countdown_frames,
        
        # Kalman filter configuration
        'kalman_process_noise': args.kalman_process_noise,
        'kalman_measurement_noise_person': args.kalman_measurement_noise_person,
        'kalman_measurement_noise_motion': args.kalman_measurement_noise_motion,
    }
    
    # Initialize detector and tracker
    detector_tracker = PersonDetectorTracker(config)
    
    # Get video paths
    video_paths = []
    
    if args.video:
        video_paths = [args.video]
    elif args.videos:
        video_paths = args.videos
    else:
        # Default video paths from the original script
        video_paths = [
            r"F:\NTUC-P-57_202506031303_202506031304.webm",
            r"F:\NTUC-P-57_202506031442_202506031445.webm",
            r"F:\NTUC-P-57_202506031631_202506031634.webm",
            r"F:\ruoxuan.webm",
            r"F:\NTUC-P-57_202506031838_202506031840.webm",
            r"F:\NTUC-P-57_202506041801_202506041803.webm",
            r"F:\NTUC-P-57_202506051744_202506051747.webm",
            r"F:\NTUC-P-57_202506061546_202506061553.webm",
            r"F:\NTUC-P-57_202506111624_202506111636.webm",
            r"F:\NTUC-P-57_202506111749_202506111750.webm",
            r"F:\NTUC-P-57_202506111759_202506111802.webm",
            r"F:\NTUC-P-57_202506111811_202506111813.webm",
            r"F:\NTUC-P-57_202506111814_202506111816.webm",
            r"F:\NTUC-P-57_202506121531_202506121533.webm",
        ]
    
    # Process videos
    detector_tracker.process_videos(
        video_paths,
        output_dir=args.output_dir,
        show_display=not args.no_display,
        save_results=not args.no_save
    )


if __name__ == "__main__":
    main()

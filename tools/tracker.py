# --- Refactored Object Tracker Module ---
import math
from collections import deque
import numpy as np
import cv2
import time
from shapely.geometry import box

class TrackerBase:
    """Base class for all trackers"""
    def __init__(self, tracker_id, tracker_type="person", history_length=10):
        self.id = tracker_id
        self.tracker_type = tracker_type
        
        self.position_history = deque(maxlen=history_length)
        self.direction_history = deque(maxlen=history_length)
        
        self.last_valid_position = None
        self.current_direction = None
        self.movement_magnitude = 0
        self.smoothed_direction = None
        self.frames_since_detected = 0
        
        self.is_relevant_to_bed = False
        self.color = self._generate_color()
        
        # Tracking state
        self.location_state_history = deque(maxlen=history_length) 
        self.raw_location_state_history = deque(maxlen=history_length) 
        self.entrance_detected = False
        self.exit_detected = False
        self.state_message_timer = 0
        
        # Bounding box information
        self.bbox = None
        self.last_bbox = None
        self.bbox_history = deque(maxlen=history_length)
        
        # Center point suppression timer
        self.center_point_suppression_timer = 0
        self.center_point_suppression_frames = 0  # Will be set by config

        

    def _generate_color(self):
        """Generate a random color for visualization"""
        np.random.seed(self.id * 42) 
        return (int(np.random.randint(50, 255)), 
                int(np.random.randint(50, 255)), 
                int(np.random.randint(50, 255)))
    
    def update_state(self, current_data_item, bed_polygons, 
                     specific_movement_threshold, specific_min_movement):
        """Base update method to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement update_state")
    
    def is_in_bed(self, bed_polygons, method="center_point"):
        """Determine if the tracker is in bed using the specified method"""
        raise NotImplementedError("Subclasses must implement is_in_bed")


class PersonTracker(TrackerBase):
    """Tracker for person objects with support for both Euclidean and IOU tracking"""
    
    def __init__(self, tracker_id, history_length=10, tracking_method="euclidean", 
                 bed_detection_method="center_point", center_point_suppression_frames=90):
        super().__init__(tracker_id, "person", history_length)
        self.tracking_method = tracking_method  # "euclidean" or "iou"
        self.bed_detection_method = bed_detection_method  # "center_point" or "iou"
    
    def update_state(self, current_data_item, bed_polygons, 
                     specific_movement_threshold, specific_min_movement):
        """
        Updates the tracker's state with new data.
        current_data_item: For center point tracking, a tuple (x,y) or None.
                          For bbox tracking, a dict with 'bbox' and 'centroid' keys or None.
        """
        current_position = None
        current_bbox = None
        
        if current_data_item is None:
            self.frames_since_detected += 1
            if self.state_message_timer > 0:
                self.state_message_timer -= 1
            return
        
        # Extract data based on tracking method
        if isinstance(current_data_item, tuple):
            # Center point only
            current_position = current_data_item
        elif isinstance(current_data_item, dict):
            # Bounding box with centroid
            current_position = current_data_item.get('centroid')
            current_bbox = current_data_item.get('bbox')
        
        if current_position is None:
            self.frames_since_detected += 1
            return
        
        # Validate movement based on tracking method
        if self.last_valid_position and self.tracking_method == "euclidean":
            distance = calculate_distance(current_position, self.last_valid_position)
            if distance > specific_movement_threshold:
                # Movement too large, revert to last valid position
                current_position = self.last_valid_position
                if current_bbox is not None and self.last_bbox is not None:
                    current_bbox = self.last_bbox
        elif self.last_bbox is not None and current_bbox is not None and self.tracking_method == "iou":
            iou_value = calculate_iou(self.last_bbox, current_bbox)
            if iou_value < 0.1:  # Very low IOU indicates potentially wrong association
                # Movement too large, revert to last valid position
                current_position = self.last_valid_position
                current_bbox = self.last_bbox
        
        # Update position and bbox
        self.last_valid_position = current_position
        if current_bbox is not None:
            self.last_bbox = current_bbox
            self.bbox_history.append(current_bbox)
        self.frames_since_detected = 0
        
        # Calculate direction and movement
        if len(self.position_history) > 0:
            previous_position = self.position_history[-1]
            self.current_direction, self.movement_magnitude = calculate_movement_direction(
                current_position, previous_position, specific_min_movement
            )
            self.direction_history.append(self.current_direction)
            self.smoothed_direction = smooth_direction(self.current_direction, list(self.direction_history))
        
        self.position_history.append(current_position)
        
        # Determine bed status
        self.update_bed_status(bed_polygons)
    
    def update_bed_status(self, bed_polygons):
        """Update whether the person is in bed based on the configured method"""
        if self.last_valid_position is None:
            return
        
        # Get raw location state
        is_in_inner_bed, is_in_outer_bed = self.is_in_bed(bed_polygons, self.bed_detection_method)
        
        if is_in_inner_bed:
            raw_state = "inner_bed"
            current_location_state_str = "inner_bed"
            self.is_relevant_to_bed = True
            
        elif is_in_outer_bed:
            raw_state = "outer_bed"
            current_location_state_str = "outer_bed"
            self.is_relevant_to_bed = True
            
        else:
            raw_state = "outside"
            current_location_state_str = "outside"
            self.is_relevant_to_bed = False
        
        # Update state histories
        self.location_state_history.append(current_location_state_str)
        self.raw_location_state_history.append(raw_state)
        
        # Countdown message timer if active
        if self.state_message_timer > 0:
            self.state_message_timer -= 1
        
    
    def is_in_bed(self, bed_polygons, method=None):
        """
        Determine if the tracker is in bed using the specified method.
        Returns (is_in_inner_bed, is_in_outer_bed)
        """
        if method is None:
            method = self.bed_detection_method
            
        if method == "center_point":
            return self._is_in_bed_center_point(bed_polygons)
        elif method == "iou":
            return self._is_in_bed_iou(bed_polygons)
        else:
            # Default to center point method
            return self._is_in_bed_center_point(bed_polygons)
    
    def _is_in_bed_center_point(self, bed_polygons):
        """Check if center point is in bed polygons"""
        if self.last_valid_position is None:
            return False, False
            
        is_in_inner_bed = is_point_in_polygon(self.last_valid_position, bed_polygons['inner'])
        is_in_outer_bed = is_point_in_polygon(self.last_valid_position, bed_polygons['outer'])
        
        return is_in_inner_bed, is_in_outer_bed
    
    def _is_in_bed_iou(self, bed_polygons):
        """Check if bounding box has significant overlap with bed polygons"""
        if self.last_bbox is None:
            return False, False
            
        # Convert bed polygons to bounding boxes
        inner_bed_bbox = polygon_to_bbox(bed_polygons['inner'])
        outer_bed_bbox = polygon_to_bbox(bed_polygons['outer'])
        
        # Calculate IOU with bed regions
        inner_iou = calculate_iou(self.last_bbox, inner_bed_bbox)
        outer_iou = calculate_iou(self.last_bbox, outer_bed_bbox)
        
        # Define thresholds for "in bed" determination
        INNER_BED_IOU_THRESHOLD = 0.3
        OUTER_BED_IOU_THRESHOLD = 0.2
        
        is_in_inner_bed = inner_iou >= INNER_BED_IOU_THRESHOLD
        is_in_outer_bed = outer_iou >= OUTER_BED_IOU_THRESHOLD
        
        return is_in_inner_bed, is_in_outer_bed
    
    def _is_mostly_inside_bed(self, bed_polygons):
        """Check if the tracker's position is mostly inside the bed area"""
        if self.last_valid_position is None:
            return False
            
        inner_bed_bbox = polygon_to_bbox(bed_polygons['inner'])
        outer_bed_bbox = polygon_to_bbox(bed_polygons['outer'])
    


# --- Tracking Assignment Functions ---

def assign_trackers_euclidean(current_detections, active_trackers, max_distance, next_id, history_length):
    """Assign detections to trackers using Euclidean distance"""
    assignments = {}
    used_tracker_ids = set()
    current_next_id = next_id
    
    for detection in current_detections:
        if isinstance(detection, tuple):
            current_pos = detection
        elif isinstance(detection, dict) and 'centroid' in detection:
            current_pos = detection['centroid']
        else:
            continue
        
        if current_pos is None:
            continue
            
        best_match_id = None
        min_dist = float('inf')
        
        for tracker_id, tracker in active_trackers.items():
            if tracker_id in used_tracker_ids or tracker.last_valid_position is None:
                continue
                
            distance = calculate_distance(current_pos, tracker.last_valid_position)
            if distance < max_distance and distance < min_dist:
                min_dist = distance
                best_match_id = tracker_id
        
        assignment_key = tuple(map(int, current_pos))
        
        if best_match_id is not None:
            assignments[assignment_key] = best_match_id
            used_tracker_ids.add(best_match_id)
        else:
            new_id = current_next_id
            active_trackers[new_id] = PersonTracker(
                new_id, 
                history_length=history_length,
                tracking_method="euclidean"
            )
            assignments[assignment_key] = new_id
            current_next_id += 1
    
    return assignments, current_next_id

def assign_trackers_iou(current_detections, active_trackers, min_iou, next_id, history_length):
    """Assign detections to trackers using IOU of bounding boxes"""
    assignments = {}
    used_tracker_ids = set()
    current_next_id = next_id
    
    for detection in current_detections:
        if not isinstance(detection, dict) or 'bbox' not in detection or 'centroid' not in detection:
            continue
            
        current_bbox = detection['bbox']
        current_pos = detection['centroid']
        
        if current_bbox is None or current_pos is None:
            continue
            
        best_match_id = None
        max_iou = min_iou  # Minimum IOU threshold
        
        for tracker_id, tracker in active_trackers.items():
            if tracker_id in used_tracker_ids or tracker.last_bbox is None:
                continue
                
            iou = calculate_iou(current_bbox, tracker.last_bbox)
            if iou > max_iou:
                max_iou = iou
                best_match_id = tracker_id
        
        assignment_key = tuple(map(int, current_pos))
        
        if best_match_id is not None:
            assignments[assignment_key] = best_match_id
            used_tracker_ids.add(best_match_id)
        else:
            new_id = current_next_id
            active_trackers[new_id] = PersonTracker(
                new_id, 
                history_length=history_length,
                tracking_method="iou"
            )
            assignments[assignment_key] = new_id
            current_next_id += 1
    
    return assignments, current_next_id

def assign_trackers(current_detections, active_trackers, tracking_method, 
                   max_distance=150, min_iou=0.3, next_id=0, history_length=10):
    """
    Assign detections to trackers using the specified method.
    
    Args:
        current_detections: List of detections (center points or dicts with 'bbox' and 'centroid')
        active_trackers: Dictionary of active trackers
        tracking_method: 'euclidean' or 'iou'
        max_distance: Maximum Euclidean distance for association (for 'euclidean' method)
        min_iou: Minimum IOU for association (for 'iou' method)
        next_id: Next available tracker ID
        history_length: History length for new trackers
        
    Returns:
        assignments: Dictionary mapping detection keys to tracker IDs
        next_id: Updated next available tracker ID
    """
    if tracking_method == "euclidean":
        return assign_trackers_euclidean(
            current_detections, active_trackers, max_distance, next_id, history_length
        )
    elif tracking_method == "iou":
        return assign_trackers_iou(
            current_detections, active_trackers, min_iou, next_id, history_length
        )
    else:
        # Default to Euclidean
        return assign_trackers_euclidean(
            current_detections, active_trackers, max_distance, next_id, history_length
        )


# --- Entrance/Exit Analysis ---

def group_consecutive_states(history):
    """Group consecutive identical states"""
    if not history:
        return []
    grouped = [history[0]]
    for state in history[1:]:
        if state != grouped[-1]:
            grouped.append(state)
    return grouped

def sequence_contains_pattern(sequence, pattern):
    """Check if sequence contains the given pattern"""
    if len(sequence) < len(pattern):
        return False
    for i in range(len(sequence) - len(pattern) + 1):
        if sequence[i:i+len(pattern)] == pattern:
            return True
    return False

def analyze_entrance_exit_pattern(tracker):
    """Analyze tracker history for entrance/exit patterns"""
    if len(tracker.location_state_history) < 3:
        return False, False
    
    history = list(tracker.raw_location_state_history)
    grouped_sequence = group_consecutive_states(history)
    
    entrance_pattern = ["outside", "outer_bed"]
    exit_pattern = ["outer_bed", "outside"]
    
    entrance_detected = sequence_contains_pattern(grouped_sequence, entrance_pattern)
    exit_detected = sequence_contains_pattern(grouped_sequence, exit_pattern)
    
    # Update tracker state if patterns are detected
    if entrance_detected and not tracker.entrance_detected:
        tracker.entrance_detected = True
        tracker.exit_detected = False
        tracker.state_message_timer = 60
    
    if exit_detected and not tracker.exit_detected:
        tracker.exit_detected = True
        tracker.entrance_detected = False
        tracker.state_message_timer = 60
    
    return entrance_detected, exit_detected


# --- Helper Functions ---

def calculate_distance(pos1, pos2):
    """Calculate Euclidean distance between two positions"""
    if pos1 is None or pos2 is None:
        return float('inf')
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def calculate_movement_direction(current_pos, previous_pos, min_movement_val):
    """Calculate movement direction and magnitude"""
    if current_pos is None or previous_pos is None:
        return None, 0
    dx = current_pos[0] - previous_pos[0]
    dy = current_pos[1] - previous_pos[1]
    magnitude = math.sqrt(dx*dx + dy*dy)
    if magnitude < min_movement_val:
        return None, magnitude
    angle = math.degrees(math.atan2(-dy, dx))
    if angle < 0:
        angle += 360
    return angle, magnitude

def get_direction_name(angle):
    """Convert angle to direction name"""
    if angle is None:
        return "Stationary"
    angle = angle % 360
    if 337.5 <= angle or angle < 22.5:
        return "Right"
    elif 22.5 <= angle < 67.5:
        return "Up-Right"
    elif 67.5 <= angle < 112.5:
        return "Up"
    elif 112.5 <= angle < 157.5:
        return "Up-Left"
    elif 157.5 <= angle < 202.5:
        return "Left"
    elif 202.5 <= angle < 247.5:
        return "Down-Left"
    elif 247.5 <= angle < 292.5:
        return "Down"
    elif 292.5 <= angle < 337.5:
        return "Down-Right"
    return "Unknown"

def smooth_direction(new_direction, history):
    """Smooth direction using history"""
    if new_direction is None:
        return None
    valid_directions = [d for d in history if d is not None]
    if len(valid_directions) < 2:
        return new_direction
    x_sum = sum(math.cos(math.radians(d)) for d in valid_directions)
    y_sum = sum(math.sin(math.radians(d)) for d in valid_directions)
    smoothed = math.degrees(math.atan2(y_sum, x_sum))
    if smoothed < 0:
        smoothed += 360
    return smoothed

def is_point_in_polygon(point, polygon_points):
    """Check if point is inside polygon"""
    if point is None:
        return False
    return cv2.pointPolygonTest(
        np.array(polygon_points, dtype=np.int32), 
        (float(point[0]), float(point[1])), 
        False
    ) >= 0

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IOU) between two bounding boxes.
    Each box should be in format (x1, y1, x2, y2).
    """
    if box1 is None or box2 is None:
        return 0.0
        
    # Get coordinates of intersection
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Calculate area of intersection
    width_inter = max(0, x2_inter - x1_inter)
    height_inter = max(0, y2_inter - y1_inter)
    area_inter = width_inter * height_inter
    
    # Calculate area of both boxes
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    area_union = area_box1 + area_box2 - area_inter
    
    # Calculate IOU
    if area_union <= 0:
        return 0.0
    return area_inter / area_union

def polygon_to_bbox(polygon_points):
    """Convert polygon points to bounding box (x1, y1, x2, y2)"""
    if not polygon_points or len(polygon_points) < 3:
        return None
        
    x_coords = [p[0] for p in polygon_points]
    y_coords = [p[1] for p in polygon_points]
    
    x1 = min(x_coords)
    y1 = min(y_coords)
    x2 = max(x_coords)
    y2 = max(y_coords)
    
    return (x1, y1, x2, y2)


def is_inner_bbox_inside_outer_bbox(inner_bbox, outer_bbox):
    """Check if inner bounding box is completely inside outer bounding box using shapely."""
    if inner_bbox is None or outer_bbox is None:
        return False
    inner = box(*inner_bbox)
    outer = box(*outer_bbox)
    return outer.contains(inner)

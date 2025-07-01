# --- Motion and Brightness Detection Module ---
import cv2
import numpy as np
import time
from collections import deque
from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box

class MotionBrightnessDetector:
    """Detects motion and brightness in video frames"""
    
    def __init__(self, config):
        """
        Initialize the motion and brightness detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Bed polygons
        self.bed_rectangle_points = np.array(config['bed_rectangle_points'], dtype=np.int32)
        
        # Thresholds
        self.brightness_threshold = config.get('brightness_threshold', 200)
        self.darkness_threshold = config.get('darkness_threshold', 30)
        self.min_blob_area = config.get('min_blob_area', 50)
        self.min_dark_blob_area = config.get('min_dark_blob_area', 50)
        self.very_bright_threshold = config.get('very_bright_threshold', 250)
        self.movement_threshold = config.get('movement_threshold', 400)
        
        # State tracking
        self.prev_binary = None
        self.prev_dark_binary = None
        
        # Movement detection countdown
        self.movement_countdown = 0
        self.movement_countdown_frames = config.get('movement_countdown_frames', 30)

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=True, history=1)
        self.bg_subtractor.setShadowThreshold(0.5)
    def process_frame(self, frame):
        """
        Process a frame to detect motion and brightness.
        
        Args:
            frame: Input frame
            
        Returns:
            tuple: (is_person_in_bed, display_frame, debug_info)
                is_person_in_bed: Whether a person is detected in bed
                display_frame: Frame with visualizations
                debug_info: Dictionary with debug information
        """
        # Make a copy of the frame for display
        display_frame = frame.copy()
        
        # Get grayscale ROI and mask
        gray_roi, mask, opacity_map = self._get_grayscale_roi_and_mask(frame)
        
        # Create binary masks for brightness and darkness
        binary = self._create_brightness_binary(gray_roi, mask)
        dark_binary = self._create_darkness_binary(gray_roi, mask)
        
    
        
        detections, _ = self.get_detections(self.bg_subtractor, frame, bbox_thresh=50, nms_thresh=0.1, kernel=np.ones((9,9), dtype=np.uint8))
        is_motion_detected = len(detections) > 0
        # bright_area = sum(blob['area'] for blob in bright_blob_data) if bright_blob_data else 0
        # for blob in bright_blob_data:
        #     x1, y1, x2, y2 = blob
        #     cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Count very bright pixels
        very_bright_pixels = self._get_very_bright_pixels(frame, self.very_bright_threshold)
        def draw_bboxes(frame, detections, color=(0,255,0)):
            for (x1, y1, x2, y2) in detections:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Calculate total bright area
        draw_bboxes(display_frame, detections)
        # # Determine if a person is in bed based on motion and brightness
        # is_counting = False
        # combined_contour_movement_level = contour_movement_level
        # # if combined_contour_movement_level > self.movement_threshold:
        # if combined_contour_movement_level > 350:
        #     is_counting = True

        is_counting = is_motion_detected
        
        
        is_person_in_bed = False
        if very_bright_pixels > 600:
            is_person_in_bed = True
        if is_counting:
            is_person_in_bed = True

        
        # Draw visualizations
        # self._draw_blob_info(display_frame, bright_blob_data, (0, 255, 0), "Bright ")
        # self._draw_blob_info(display_frame, dark_blob_data, (0, 0, 255), "Dark ")
        
        # Draw bed rectangle
        cv2.polylines(display_frame, [self.bed_rectangle_points], True, (0, 255, 255), 2)
        
        # Draw movement detection alerts
        if is_counting:
            cv2.putText(display_frame, "Movement Detected!", (10, 270), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
        
        if very_bright_pixels > 600:
            cv2.putText(display_frame, "Very Bright Pixels Detected!", (10, 285), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
        
        # Update previous states
        self.prev_binary = binary.copy() if binary is not None else None
        self.prev_dark_binary = dark_binary.copy() if dark_binary is not None else None
        
        # Prepare debug info
        debug_info = {
            # 'bright_area': bright_area,
            # 'contour_movement_level': contour_movement_level,
            # 'dark_contour_movement_level': dark_contour_movement_level,
            'very_bright_pixels': very_bright_pixels,
            # 'bright_blob_count': len(bright_blob_data),
            # 'dark_blob_count': len(dark_blob_data),
            # 'combined_contour_movement_level': combined_contour_movement_level,
            'is_counting': is_counting,
            'is_person_in_bed': is_person_in_bed,
            'is_motion_detected': is_motion_detected,
        }
        
        return is_person_in_bed, display_frame, debug_info
    
    def _get_grayscale_roi_and_mask(self, frame):
        """
        Get grayscale ROI and mask for the bed region.
        
        Args:
            frame: Input frame
            
        Returns:
            tuple: (roi, mask, opacity_map)
        """
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, self.bed_rectangle_points, 255)
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)
        
        # Placeholder for opacity_map to maintain function signature
        opacity_map = None
        
        return roi, mask, opacity_map
    
    def _create_brightness_binary(self, roi, mask):
        """Create binary mask for bright pixels"""
        _, binary = cv2.threshold(roi, self.brightness_threshold, 255, cv2.THRESH_BINARY)
        return binary
    
    def _create_darkness_binary(self, roi, mask):
        """Create binary mask for dark pixels"""
        dark_binary = np.zeros_like(roi, dtype=np.uint8)
        dark_binary[(roi > 0) & (roi < self.darkness_threshold)] = 255
        return dark_binary
    
    
    def _extract_blob_data(self, binary_roi, min_area):
        """
        Extract blob data from connected components analysis.
        
        Args:
            binary_roi: Binary ROI
            min_area: Minimum area to consider a valid blob
            
        Returns:
            list: List of blob data dictionaries
        """
        if binary_roi is None:
            return []
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_roi, connectivity=4)
        blob_data = []
        
        for i in range(1, num_labels):  # Start from 1 to skip background
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]
            
            # Filter by minimum area
            if area > min_area:
                blob_data.append({
                    'centroid': (cx, cy),
                    'area': area,
                    'bbox': (x, y, w, h),
                    'stats': stats[i],
                    'labels': labels,  # Include labels for debugging
                })
        
        return blob_data
    
    def _calculate_movement_levels(self, binary, dark_binary):
        """
        Calculate movement levels between current and previous frames.
        
        Returns:
            tuple: (contour_movement_level, dark_contour_movement_level)
        """
        contour_movement_level = 0
        dark_contour_movement_level = 0
        
        # Calculate contour movement levels
        if self.prev_binary is not None and binary is not None and binary.shape == self.prev_binary.shape:
            contour_movement_level = np.sum(cv2.absdiff(binary, self.prev_binary) > 0)
        
        if self.prev_dark_binary is not None and dark_binary is not None and dark_binary.shape == self.prev_dark_binary.shape:
            dark_contour_movement_level = np.sum(cv2.absdiff(dark_binary, self.prev_dark_binary) > 0)
        
        return contour_movement_level, dark_contour_movement_level
    
    def _get_very_bright_pixels(self, frame, threshold):
        """
        Get the number of very bright pixels in the bed region.
        
        Args:
            frame: Input frame
            threshold: Brightness threshold
            
        Returns:
            int: Number of very bright pixels
        """
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, self.bed_rectangle_points, 255)
        
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply mask
        masked_frame = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)
        
        # Count very bright pixels
        very_bright_pixels = np.sum(masked_frame > threshold)
        
        return very_bright_pixels
    
    
    def _draw_blob_info(self, frame, blob_data, color, label_prefix):
        """
        Draw blob information on frame.
        
        Args:
            frame: Frame to draw on
            blob_data: List of blob data dictionaries
            color: Color for visualization
            label_prefix: Prefix for labels
        """
        for blob in blob_data:
            x, y, w, h = blob['bbox']
            cx, cy = blob['centroid']
            area = blob['area']
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw centroid
            cv2.circle(frame, (int(cx), int(cy)), 3, color, -1)
            
            # Draw area and centroid info
            info_text = f"{label_prefix}Area:{area} C:({int(cx)},{int(cy)})"
            cv2.putText(frame, info_text, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            
            # Draw stats info (width x height)
            stats_text = f"W:{w} H:{h}"
            cv2.putText(frame, stats_text, (x, y + h + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
        
    # Motion detection functions
    def get_motion_mask(self, fg_mask, min_thresh=0, kernel=np.ones((9,9), dtype=np.uint8)):
        """Obtains a clean motion mask from the foreground mask."""
        _, thresh = cv2.threshold(fg_mask, min_thresh, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.medianBlur(thresh, 3)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return motion_mask

    def get_contour_detections(self, mask, bbox_thresh=100 ,bed_rectangle=None):
        """Finds bounding boxes from contours in the mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area > bbox_thresh and self.is_box_over_lap_polygon([x, y, x + w, y + h], self.bed_rectangle_points):
                detections.append([x, y, x + w, y + h, area])
        return np.array(detections) if detections else np.zeros((0, 5))

    def non_max_suppression(self, boxes, scores, threshold=0.1):
        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=int)
        boxes = boxes.astype(np.float32)
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]
        return boxes[keep].astype(int)

    def draw_bboxes(self, frame, detections, color=(0,255,0)):
        for (x1, y1, x2, y2) in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    def get_detections(self, backSub, frame, bbox_thresh=100, nms_thresh=0.1, kernel=np.ones((9,9), dtype=np.uint8)):
        fg_mask = backSub.apply(frame, learningRate=0.01)
        motion_mask = self.get_motion_mask(fg_mask, kernel=kernel)
        detections = self.get_contour_detections(motion_mask, bbox_thresh)
        if len(detections) == 0:
            return np.zeros((0, 4), dtype=int), fg_mask
        bboxes = detections[:, :4]
        scores = detections[:, -1]
        return self.non_max_suppression(bboxes, scores, nms_thresh), fg_mask

    def is_box_over_lap_polygon(self, box, polygon):
        """
        Check if a bounding box overlaps with a polygon.
        :param box: [x1, y1, x2, y2]
        :param polygon: Nx2 np.ndarray of polygon points
        :return: True if there is an overlap, False otherwise
        """
        poly = Polygon(polygon)
        rect = shapely_box(box[0], box[1], box[2], box[3])
        return not poly.intersection(rect).is_empty


import cv2
import numpy as np
# import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
from collections import deque


class MovementTrendAnalyzer:
    """
    Analyzes movement trends in video data, specifically focusing on detecting
    whether the movement (area) is increasing or decreasing over time.
    Also calculates variance to distinguish between consistent trends with small
    variations and trends with large jumps.
    """
    
    def __init__(self, window_size=30, outlier_threshold=2.5, min_trend_strength=0.1):
        """
        Initialize the trend analyzer.
        
        Args:
            window_size: Number of frames to analyze for trend detection
            outlier_threshold: Z-score threshold for outlier detection
            min_trend_strength: Minimum slope to consider as a significant trend
        """
        self.window_size = window_size
        self.outlier_threshold = outlier_threshold
        self.min_trend_strength = min_trend_strength
        
        # Data storage
        self.data_history = deque(maxlen=window_size * 2)  # Store more for stability
        self.frame_numbers = deque(maxlen=window_size * 2)
        
        # Trend state
        self.current_trend = "stable"  # "increasing", "decreasing", "stable"
        self.trend_strength = 0.0
        self.trend_confidence = 0.0
        self.trend_variance = 0.0
        self.normalized_variance = 0.0  # Variance relative to the trend magnitude
        
    def add_data_point(self, frame_number, value):
        """Add new data point"""
        self.data_history.append(value)
        self.frame_numbers.append(frame_number)
        
        if len(self.data_history) >= self.window_size:
            self._update_trend()
    
    def _filter_outliers(self, data):
        """Remove outliers using modified Z-score method"""
        if len(data) < 5:
            return data
        
        data_array = np.array(data)
        
        # Use median-based outlier detection (more robust than mean)
        median = np.median(data_array)
        mad = np.median(np.abs(data_array - median))  # Median Absolute Deviation
        
        if mad == 0:
            return data_array  # No variation, return as-is
        
        # Modified Z-score
        modified_z_scores = 0.6745 * (data_array - median) / mad
        
        # Filter outliers
        mask = np.abs(modified_z_scores) < self.outlier_threshold
        
        # If too many outliers, use a more lenient threshold
        if np.sum(mask) < len(data) * 0.5:
            mask = np.abs(modified_z_scores) < self.outlier_threshold * 1.5
        
        return data_array[mask]
    
    def _smooth_data(self, data):
        """Apply smoothing to reduce noise"""
        if len(data) < 5:
            return data
        
        window_length = min(len(data) // 3, 7)
        if window_length % 2 == 0:
            window_length += 1  # Must be odd for Savitzky-Golay filter
        
        if window_length < 3:
            return data
        
        try:
            # Savitzky-Golay filter for smooth trends
            smoothed = savgol_filter(data, window_length, 2)
            return smoothed
        except:
            # Fallback to simple moving average
            return np.convolve(data, np.ones(3)/3, mode='same')
    
    def _calculate_trend_slope(self, data, frames):
        """Calculate trend slope using robust linear regression"""
        if len(data) < 3:
            return 0.0, 0.0
        
        # Normalize frame numbers to start from 0
        x = np.array(frames) - frames[0]
        y = np.array(data)
        
        # Use robust regression (Theil-Sen estimator)
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            confidence = abs(r_value)  # Correlation coefficient as confidence
            return slope, confidence
        except:
            return 0.0, 0.0
    
    def _calculate_variance(self, data, trend_line):
        """
        Calculate variance of data points from the trend line.
        This helps distinguish between consistent trends with small variations
        and trends with large jumps.
        """
        if len(data) < 3 or len(trend_line) != len(data):
            return 0.0, 0.0
        
        # Calculate residuals (differences between actual data and trend line)
        residuals = data - trend_line
        
        # Calculate variance of residuals
        variance = np.var(residuals)
        
        # Calculate normalized variance (relative to the magnitude of the data)
        data_range = np.max(data) - np.min(data) if np.max(data) != np.min(data) else 1.0
        normalized_variance = variance / (data_range ** 2)
        
        return variance, normalized_variance
    
    def _update_trend(self):
        """Update trend analysis"""
        if len(self.data_history) < self.window_size:
            return
        
        # Get recent data
        recent_data = list(self.data_history)[-self.window_size:]
        recent_frames = list(self.frame_numbers)[-self.window_size:]
        
        # Filter outliers
        filtered_data = self._filter_outliers(recent_data)
        
        if len(filtered_data) < self.window_size * 0.5:
            # Too many outliers, trend uncertain
            self.current_trend = "uncertain"
            self.trend_confidence = 0.0
            self.trend_variance = 0.0
            self.normalized_variance = 0.0
            return
        
        # Smooth the filtered data
        smoothed_data = self._smooth_data(filtered_data)
        
        # Calculate trend slope
        slope, confidence = self._calculate_trend_slope(
            smoothed_data, 
            recent_frames[-len(smoothed_data):]
        )
        
        # Calculate trend line based on slope
        x = np.array(recent_frames[-len(smoothed_data):]) - recent_frames[-len(smoothed_data)]
        trend_line = slope * x + smoothed_data[0]
        
        # Calculate variance from trend line
        variance, normalized_variance = self._calculate_variance(smoothed_data, trend_line)
        self.trend_variance = variance
        self.normalized_variance = normalized_variance
        
        # Determine trend direction
        self.trend_strength = abs(slope)
        self.trend_confidence = confidence
        
        if abs(slope) < self.min_trend_strength:
            self.current_trend = "stable"
        elif slope > 0:
            self.current_trend = "increasing"
        else:
            self.current_trend = "decreasing"
    
    def get_trend_info(self):
        """Get current trend information"""
        return {
            'trend_direction': self.current_trend,
            'trend_strength': self.trend_strength,
            'confidence': self.trend_confidence,
            'variance': self.trend_variance,
            'normalized_variance': self.normalized_variance,
            'data_points': len(self.data_history),
            'is_reliable': self.trend_confidence > 0.3 and len(self.data_history) >= self.window_size
        }
    
    def get_trend_summary(self):
        """Get human-readable trend summary"""
        info = self.get_trend_info()
        
        if not info['is_reliable']:
            return "Insufficient data for reliable trend"
        
        direction = info['trend_direction']
        strength = info['trend_strength']
        confidence = info['confidence']
        norm_variance = info['normalized_variance']
        
        # Describe strength
        if strength < 0.5:
            strength_desc = "slightly"
        elif strength < 2.0:
            strength_desc = "moderately"
        else:
            strength_desc = "strongly"
        
        # Describe confidence
        if confidence < 0.5:
            confidence_desc = "low confidence"
        elif confidence < 0.7:
            confidence_desc = "medium confidence"
        else:
            confidence_desc = "high confidence"
        
        # Describe variance
        if norm_variance < 0.01:
            variance_desc = "very consistent"
        elif norm_variance < 0.05:
            variance_desc = "consistent"
        elif norm_variance < 0.15:
            variance_desc = "with some variance"
        else:
            variance_desc = "with high variance"
        
        if direction == "stable":
            return f"Movement is stable {variance_desc} ({confidence_desc})"
        else:
            return f"Movement is {strength_desc} {direction} {variance_desc} ({confidence_desc})"


def analyze_video_movement_trend(video_path, roi_points=None, threshold=250, window_size=30):
    """
    Analyze a video to determine if movement (area) is increasing or decreasing over time.
    Also calculates variance to distinguish between consistent trends with small variations
    and trends with large jumps.
    
    Args:
        video_path: Path to the video file
        roi_points: Region of interest points as a list of (x,y) tuples. If None, uses the entire frame.
        threshold: Brightness threshold for detecting movement
        window_size: Number of frames to analyze for trend detection
        
    Returns:
        A dictionary containing trend analysis results including variance metrics
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Could not open video file: {video_path}"}
    
    # Initialize trend analyzer
    trend_analyzer = MovementTrendAnalyzer(window_size=window_size)
    
    # Previous frame for movement detection
    prev_binary = None
    
    # Process video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get frame number
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        # Create mask for ROI
        if roi_points is None:
            # Use entire frame if no ROI specified
            roi_points = [(0, 0), (frame.shape[1], 0), 
                         (frame.shape[1], frame.shape[0]), (0, frame.shape[0])]
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.array(roi_points), 255)
        
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = cv2.bitwise_and(gray, gray, mask=mask)
        _, binary = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY)
        
        # Extract blob data
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4)
        
        # Calculate total area of all blobs (excluding background)
        total_area = 0
        for i in range(1, num_labels):  # Start from 1 to skip background
            area = stats[i, cv2.CC_STAT_AREA]
            total_area += area
        
        # Add data point to trend analyzer
        trend_analyzer.add_data_point(frame_number, total_area)
        
        # Update previous frame
        prev_binary = binary.copy()
    
    # Release video capture
    cap.release()
    
    # Get final trend analysis
    trend_info = trend_analyzer.get_trend_info()
    trend_summary = trend_analyzer.get_trend_summary()
    
    return {
        "trend_direction": trend_info["trend_direction"],
        "trend_strength": trend_info["trend_strength"],
        "confidence": trend_info["confidence"],
        "variance": trend_info["variance"],
        "normalized_variance": trend_info["normalized_variance"],
        "is_reliable": trend_info["is_reliable"],
        "summary": trend_summary
    }


if __name__ == "__main__":
    # Example usage
    # video_path = r"C:\Users\Tairin Pairor\Downloads\NTUC-P-57_202506111624_202506111636.webm"
    video_path = r"F:\NTUC-P-57_202506061546_202506061553.webm"
    roi_points = [(135, 149), (211, 153), (247, 5), (148, 0)]  # Example ROI
    
    result = analyze_video_movement_trend(video_path, roi_points)
    # print(result)

import numpy as np


class BedOccupancyKalmanFilter:
    """Kalman filter for bed occupancy estimation"""
    
    def __init__(self, process_noise=0.01, measurement_noise_person=0.1, measurement_noise_motion=0.2):
        """
        Initialize the Kalman filter for bed occupancy estimation.
        
        Args:
            process_noise: Process noise covariance
            measurement_noise_person: Measurement noise for person detector
            measurement_noise_motion: Measurement noise for motion detector
        """
        # State vector: [bed_occupancy_probability, bed_occupancy_change_rate]
        self.state = np.zeros((2, 1), dtype=np.float32)
        
        # State transition matrix
        self.A = np.array([[1, 1], [0, 1]], dtype=np.float32)
        
        # Measurement matrix - maps state to measurements
        self.H = np.array([[1, 0], [1, 0]], dtype=np.float32)
        
        # Process noise covariance
        self.Q = np.eye(2, dtype=np.float32) * process_noise
        
        # Measurement noise covariance
        self.R = np.array([
            [measurement_noise_person, 0],
            [0, measurement_noise_motion]
        ], dtype=np.float32)
        
        # Error covariance matrix
        self.P = np.eye(2, dtype=np.float32)
        
        # Identity matrix
        self.I = np.eye(2, dtype=np.float32)
        
        # Kalman gain
        self.K = np.zeros((2, 2), dtype=np.float32)
        
        # History of estimates
        self.history = []
        
        # Smoothed estimate (exponential moving average)
        self.smoothed_estimate = 0.0
        self.alpha = 0.3  # Smoothing factor
    
    def predict(self):
        """Predict the next state"""
        # Project the state ahead
        self.state = np.dot(self.A, self.state)
        
        # Project the error covariance ahead
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        
        return self.state[0, 0]  # Return the bed occupancy probability
    
    def update(self, person_measurement, motion_measurement):
        """
        Update the state with new measurements.
        
        Args:
            person_measurement: Measurement from person detector (0 to 1)
            motion_measurement: Measurement from motion detector (0 to 1)
            
        Returns:
            float: Updated bed occupancy probability
        """
        # Measurement vector
        Z = np.array([[person_measurement], [motion_measurement]], dtype=np.float32)
        
        # Compute the Kalman gain
        PHT = np.dot(self.P, self.H.T)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        self.K = np.dot(PHT, np.linalg.inv(S))
        
        # Update the estimate with measurement
        y = Z - np.dot(self.H, self.state)
        self.state = self.state + np.dot(self.K, y)
        
        # Update the error covariance
        self.P = np.dot((self.I - np.dot(self.K, self.H)), self.P)
        
        # Ensure the state is within bounds
        self.state[0, 0] = max(0, min(1, self.state[0, 0]))
        
        # Update smoothed estimate
        self.smoothed_estimate = self.alpha * self.state[0, 0] + (1 - self.alpha) * self.smoothed_estimate
        
        # Store history
        self.history.append(self.smoothed_estimate)
        if len(self.history) > 100:
            self.history.pop(0)
        
        return self.smoothed_estimate
    
    def update_multi(self, measurements, measurement_noises=None):
        """
        Update the state with new measurements (supports multiple metrics).
        
        Args:
            measurements: List or array of measurements (each 0 to 1)
            measurement_noises: Optional list/array of measurement noise values (same length as measurements)
            
        Returns:
            float: Updated bed occupancy probability
        """
        measurements = np.array(measurements, dtype=np.float32).reshape(-1, 1)
        num_metrics = measurements.shape[0]

        # Dynamically build H and R if needed
        H = np.ones((num_metrics, 2), dtype=np.float32)
        H[:, 1] = 0  # Only first state variable is measured

        if measurement_noises is not None:
            R = np.diag(measurement_noises)
        else:
            # Default: use original R for first two, 0.2 for extras
            default_noises = [0.1, 0.2] + [0.2] * (num_metrics - 2)
            R = np.diag(default_noises[:num_metrics])

        # Kalman gain
        PHT = np.dot(self.P, H.T)
        S = np.dot(np.dot(H, self.P), H.T) + R
        K = np.dot(PHT, np.linalg.inv(S))

        # Update estimate
        y = measurements - np.dot(H, self.state)
        self.state = self.state + np.dot(K, y)

        # Update error covariance
        self.P = np.dot((self.I - np.dot(K, H)), self.P)

        # Ensure state is within bounds
        self.state[0, 0] = max(0, min(1, self.state[0, 0]))

        # Update smoothed estimate
        self.smoothed_estimate = self.alpha * self.state[0, 0] + (1 - self.alpha) * self.smoothed_estimate

        # Store history
        self.history.append(self.smoothed_estimate)
        if len(self.history) > 100:
            self.history.pop(0)

        return self.smoothed_estimate
    
    def get_occupancy_probability(self):
        """Get the current bed occupancy probability"""
        return self.smoothed_estimate
    
    def get_occupancy_percentage(self):
        """Get the current bed occupancy as a percentage"""
        return int(self.smoothed_estimate * 100)

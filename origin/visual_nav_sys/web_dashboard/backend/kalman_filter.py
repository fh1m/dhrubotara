"""
Simple Kalman Filter for smoothing IMU and position data
Provides stability without lag
"""
import numpy as np

class KalmanFilter1D:
    """1D Kalman filter for single variable smoothing"""
    def __init__(self, process_variance=0.01, measurement_variance=0.1, initial_value=0.0):
        self.process_variance = process_variance  # Q - how much we trust the model
        self.measurement_variance = measurement_variance  # R - how much we trust the measurement
        self.estimate = initial_value
        self.estimate_error = 1.0
        
    def update(self, measurement):
        """Update with new measurement"""
        # Prediction
        prediction = self.estimate
        prediction_error = self.estimate_error + self.process_variance
        
        # Update
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.estimate_error = (1 - kalman_gain) * prediction_error
        
        return self.estimate

class AttitudeKalmanFilter:
    """Kalman filter for attitude (roll, pitch, yaw)"""
    def __init__(self):
        # Low variance for stable attitude reading
        self.roll_filter = KalmanFilter1D(process_variance=0.001, measurement_variance=0.05)
        self.pitch_filter = KalmanFilter1D(process_variance=0.001, measurement_variance=0.05)
        self.yaw_filter = KalmanFilter1D(process_variance=0.005, measurement_variance=0.1)
        
    def update(self, roll, pitch, yaw):
        """Update with new attitude measurements"""
        return (
            self.roll_filter.update(roll),
            self.pitch_filter.update(pitch),
            self.yaw_filter.update(yaw)
        )

class PositionKalmanFilter:
    """Kalman filter for 3D position"""
    def __init__(self):
        # Higher variance for position (more dynamic)
        self.x_filter = KalmanFilter1D(process_variance=0.01, measurement_variance=0.2)
        self.y_filter = KalmanFilter1D(process_variance=0.01, measurement_variance=0.2)
        self.z_filter = KalmanFilter1D(process_variance=0.01, measurement_variance=0.2)
        
    def update(self, x, y, z):
        """Update with new position measurements"""
        return (
            self.x_filter.update(x),
            self.y_filter.update(y),
            self.z_filter.update(z)
        )

class VelocityKalmanFilter:
    """Kalman filter for velocity with drift compensation"""
    def __init__(self):
        # Medium variance, with drift decay
        self.vx_filter = KalmanFilter1D(process_variance=0.02, measurement_variance=0.3)
        self.vy_filter = KalmanFilter1D(process_variance=0.02, measurement_variance=0.3)
        self.vz_filter = KalmanFilter1D(process_variance=0.02, measurement_variance=0.3)
        self.drift_decay = 0.98  # Decay towards zero to prevent drift
        
    def update(self, vx, vy, vz):
        """Update with drift compensation"""
        vx_filtered = self.vx_filter.update(vx) * self.drift_decay
        vy_filtered = self.vy_filter.update(vy) * self.drift_decay
        vz_filtered = self.vz_filter.update(vz) * self.drift_decay
        return (vx_filtered, vy_filtered, vz_filtered)


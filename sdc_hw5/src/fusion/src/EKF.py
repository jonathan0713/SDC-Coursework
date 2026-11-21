import numpy as np
from math import cos, sin, atan2

class ExtendedKalmanFilter:
    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        # Define the state to be estimated: [x, y, yaw]
        self.pose = np.array([x, y, yaw], dtype=float)
        
        # Transition matrix (Jacobian of motion model, updated in predict)
        self.A = np.identity(3)
        self.B = np.identity(3)
        
        # State covariance matrix
        self.S = np.identity(3) * 1.0
        
        # Observation matrix for GPS (only observing x and y)
        self.C = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])
        
        # State transition error covariance (Process noise)
        self.R = np.identity(3) * 0.1
        
        # Measurement error covariance (Observation noise for x, y)
        self.Q = np.identity(2) * 1.0
        print("Initialize Kalman Filter")
    
    def predict(self, u):
        # u = [dx_local, dy_local, dyaw]
        dx, dy, dyaw = u[0], u[1], u[2]
        theta = self.pose[2]
        
        # 1. State Prediction based on local odometry changes
        self.pose[0] += dx * cos(theta) - dy * sin(theta)
        self.pose[1] += dx * sin(theta) + dy * cos(theta)
        self.pose[2] += dyaw
        
        # Normalize yaw to strictly fall within [-pi, pi]
        self.pose[2] = (self.pose[2] + np.pi) % (2 * np.pi) - np.pi
        
        # 2. Calculate Jacobian matrix of the motion model w.r.t state (self.A)
        self.A = np.identity(3)
        self.A[0, 2] = -dx * sin(theta) - dy * cos(theta)
        self.A[1, 2] =  dx * cos(theta) - dy * sin(theta)
        
        # 3. Covariance Prediction
        self.S = self.A @ self.S @ self.A.T + self.R
        
    def update(self, z):
        # z = [gps_x, gps_y]
        
        # 1. Measurement Prediction
        z_pred = np.array([self.pose[0], self.pose[1]])
        residual = z - z_pred
        
        # 2. Calculate Kalman Gain
        # S_meas = C * S * C^T + Q
        S_meas = self.C @ self.S @ self.C.T + self.Q
        K = self.S @ self.C.T @ np.linalg.inv(S_meas)
        
        # 3. State Update
        self.pose = self.pose + K @ residual
        self.pose[2] = (self.pose[2] + np.pi) % (2 * np.pi) - np.pi
        
        # 4. Covariance Update
        self.S = (np.identity(3) - K @ self.C) @ self.S
        
        return self.pose, self.S
import numpy as np

class KalmanFilter:
    def __init__(self, x=0, y=0, yaw=0):
        # 狀態矩陣 [x, y, yaw]
        self.state = np.array([x, y, yaw], dtype=float)
        
        # 狀態轉移矩陣 (Transition matrix)
        self.A = np.identity(3, dtype=float)
        # 控制矩陣 (Control matrix)
        self.B = np.identity(3, dtype=float)
        
        # 狀態共變異數矩陣 (State covariance matrix)
        self.S = np.identity(3, dtype=float) * 1.0
        
        # 觀測矩陣 (Observation matrix)
        # 由於我們只測量到 x 與 y，yaw 並未直接測量，因此為 2x3 矩陣
        self.C = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        
        # 狀態轉移誤差 (Process Noise Covariance)
        # 依據 filtered_path.py 中的 control_var = 1 設定
        self.R = np.identity(3, dtype=float) * 1000
        
        # 測量誤差 (Measurement Noise Covariance)
        # 依據 filtered_path.py 中的 measurement_var = 3 設定
        self.Q = np.identity(2, dtype=float) * 0.001

    def predict(self, u):
        # 預測狀態: x_{k|k-1} = A * x_{k-1|k-1} + B * u_k
        self.state = np.dot(self.A, self.state) + np.dot(self.B, u)
        
        # 預測共變異數: S_{k|k-1} = A * S_{k-1|k-1} * A^T + R
        self.S = np.dot(np.dot(self.A, self.S), self.A.T) + self.R
        
        return self.state, self.S

    def update(self, z):
        # 計算殘差 (Innovation): y = z - C * x_{k|k-1}
        y = z - np.dot(self.C, self.state)
        
        # 計算殘差共變異數: S_inv = C * S_{k|k-1} * C^T + Q
        S_inv = np.dot(np.dot(self.C, self.S), self.C.T) + self.Q
        
        # 計算卡爾曼增益 (Kalman Gain): K = S_{k|k-1} * C^T * S_inv^-1
        K = np.dot(np.dot(self.S, self.C.T), np.linalg.inv(S_inv))
        
        # 更新狀態: x_{k|k} = x_{k|k-1} + K * y
        self.state = self.state + np.dot(K, y)
        
        # 更新共變異數: S_{k|k} = (I - K * C) * S_{k|k-1}
        self.S = np.dot(np.identity(3) - np.dot(K, self.C), self.S)
        
        return self.state, self.S
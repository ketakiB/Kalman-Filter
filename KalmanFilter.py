import numpy as np

class KalmanFilter:
    def __init__(self, F=None, B=None, H=None, Q=None, R=None, P=None, x=None):
        """
        Initialize the Kalman Filter
        :param F: State transition matrix
        :param B: Control input matrix
        :param H: Observation matrix
        :param Q: Process noise covariance
        :param R: Measurement noise covariance
        :param P: Estimate error covariance
        :param x: Initial state estimate
        """
        self.F = F  # State transition matrix
        self.B = B  # Control input matrix, optional
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = P  # Estimate error covariance
        self.x = x  # Initial state estimate

    def predict(self, u=None):
        """
        Predict the next state
        :param u: Control vector
        """
        # Predict the state
        if u is not None and self.B is not None:
            self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        else:
            self.x = np.dot(self.F, self.x)
        
        # Predict the estimate covariance
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        """
        Update the state based on measurement
        :param z: Measurement
        """
        # Measurement update
        y = z - np.dot(self.H, self.x)  # Measurement residual
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # Residual covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman gain
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.F.shape[0])  # Identity matrix
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

    def get_state(self):
        """
        Return the current state estimate.
        """
        return self.x
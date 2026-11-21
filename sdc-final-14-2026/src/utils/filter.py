from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Filter(Protocol):
    def predict(self) -> None:
        """Advance internal state by one time step."""
        ...

    def correct(self, z: np.ndarray) -> None:
        """Update state with observation z (shape: [x, y])."""
        ...

    @property
    def state(self) -> np.ndarray:
        """Current best position estimate, shape (2,): [x, y]."""
        ...

    @property
    def covariance(self) -> np.ndarray:
        """Current 2×2 position covariance matrix."""
        ...


class KalmanFilter:
    """
    Constant-velocity Kalman Filter.

    State vector:    [x, y, vx, vy]  (4D)
    Observation:     [x, y, ṙ]       (3D augmented, H' linearised at predicted state)

    Args:
        initial_pos:  [x, y] initial position
        initial_vel:  [vx, vy] initial velocity; if None, defaults to [0, 0]
        dt:           time step between frames (default 1)
        q_var:        process noise variance (position uncertainty growth per step)
        r_var:        measurement noise variance (radar position noise)
        p_var:        initial state covariance diagonal value
        r_var_vel:    measurement noise variance for range-rate observation
    """

    def __init__(
        self,
        initial_pos: np.ndarray,
        initial_vel: np.ndarray | None = None,
        dt: float = 1.0,
        q_var: float = 1.0,
        r_var: float = 1.0,
        p_var: float = 10.0,
        r_var_vel: float = 1.0,
        q_var_lat: float | None = None,
    ) -> None:
        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=float)

        # Observation matrix: observe [x, y] from [x, y, vx, vy]
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        # Lateral process noise: if not specified, equal to longitudinal.
        # Set q_var_lat < q_var to penalise lateral (y) velocity and displacement,
        # encoding the prior that objects move primarily in the x (forward) direction.
        _q_lat = q_var if q_var_lat is None else float(q_var_lat)

        # Process noise covariance — anisotropic: x/vx use q_var, y/vy use _q_lat
        self.Q = np.diag([q_var, _q_lat, q_var, _q_lat]).astype(float)

        # Measurement noise covariance
        self.R = np.eye(2, dtype=float) * r_var
        self.r_var_vel = r_var_vel

        vx = float(initial_vel[0]) if initial_vel is not None else 0.0
        vy = float(initial_vel[1]) if initial_vel is not None else 0.0
        self.x = np.array([initial_pos[0], initial_pos[1], vx, vy], dtype=float)

        # State covariance: high initial uncertainty on velocity.
        # vy initial uncertainty is scaled proportionally to q_var_lat / q_var so
        # that tighter lateral process noise also means tighter lateral velocity prior.
        _lat_ratio = _q_lat / max(q_var, 1e-12)
        self.P = np.eye(4, dtype=float) * p_var
        self.P[2, 2] = p_var * 10
        self.P[3, 3] = p_var * 10 * _lat_ratio

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def correct(self, z: np.ndarray) -> None:
        if len(z) == 3:
            # Augmented: z = [x, y, v_r]. Linearise H at predicted state (EKF style).
            r = float(np.linalg.norm(self.x[:2]))
            cos_t, sin_t = (float(self.x[0]) / r, float(self.x[1]) / r) if r > 1e-6 else (1.0, 0.0)
            H = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, cos_t, sin_t]], dtype=float)
            R = np.diag([self.R[0, 0], self.R[1, 1], self.r_var_vel])
        else:
            H, R = self.H, self.R
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P

    @property
    def state(self) -> np.ndarray:
        return self.H @ self.x  # [x, y]

    @property
    def covariance(self) -> np.ndarray:
        return self.H @ self.P @ self.H.T  # 2×2


class IdentityFilter:
    """
    No-op filter: returns raw detection positions without smoothing.
    Useful as a baseline and for keeping existing tests passing.
    """

    _HIGH_COV = np.eye(2, dtype=float) * 1e6

    def __init__(self, initial_pos: np.ndarray) -> None:
        self._state = np.array(initial_pos[:2], dtype=float)

    def predict(self) -> None:
        pass  # no motion model

    def correct(self, z: np.ndarray) -> None:
        self._state = np.array(z[:2], dtype=float)

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self._HIGH_COV.copy()

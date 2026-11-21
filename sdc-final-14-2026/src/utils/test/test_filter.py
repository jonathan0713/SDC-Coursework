import numpy as np
import pytest

from utils.filter import Filter, IdentityFilter, KalmanFilter


# ── Protocol conformance ──────────────────────────────────────────────────────

def test_kalman_satisfies_filter_protocol():
    kf = KalmanFilter(np.array([1.0, 2.0]))
    assert isinstance(kf, Filter)


def test_identity_satisfies_filter_protocol():
    f = IdentityFilter(np.array([1.0, 2.0]))
    assert isinstance(f, Filter)


# ── KalmanFilter ──────────────────────────────────────────────────────────────

def test_kf_initial_state():
    kf = KalmanFilter(np.array([3.0, 4.0]))
    np.testing.assert_allclose(kf.state, [3.0, 4.0])


def test_kf_covariance_shape():
    kf = KalmanFilter(np.array([0.0, 0.0]))
    assert kf.covariance.shape == (2, 2)


def test_kf_predict_shifts_position_with_velocity():
    kf = KalmanFilter(np.array([0.0, 0.0]))
    # Manually set velocity
    kf.x[2] = 2.0   # vx = 2
    kf.x[3] = 1.0   # vy = 1
    kf.predict()
    np.testing.assert_allclose(kf.state, [2.0, 1.0], atol=1e-9)


def test_kf_predict_expands_covariance():
    kf = KalmanFilter(np.array([0.0, 0.0]))
    p_before = kf.covariance.copy()
    kf.predict()
    # Covariance should grow after predict (no correction yet)
    assert np.all(np.diag(kf.covariance) >= np.diag(p_before))


def test_kf_correct_moves_state_toward_observation():
    kf = KalmanFilter(np.array([0.0, 0.0]))
    kf.predict()
    obs = np.array([5.0, 5.0])
    kf.correct(obs)
    # State should shift toward observation
    assert kf.state[0] > 0.0
    assert kf.state[1] > 0.0


def test_kf_correct_reduces_covariance():
    kf = KalmanFilter(np.array([0.0, 0.0]))
    kf.predict()
    p_before = kf.covariance.copy()
    kf.correct(np.array([0.1, 0.1]))
    # Correction should reduce uncertainty
    assert np.all(np.diag(kf.covariance) <= np.diag(p_before))


def test_kf_repeated_same_observation_converges():
    kf = KalmanFilter(np.array([0.0, 0.0]))
    obs = np.array([10.0, 10.0])
    for _ in range(50):
        kf.predict()
        kf.correct(obs)
    np.testing.assert_allclose(kf.state, obs, atol=0.5)


# ── IdentityFilter ────────────────────────────────────────────────────────────

def test_identity_initial_state():
    f = IdentityFilter(np.array([1.0, 2.0]))
    np.testing.assert_allclose(f.state, [1.0, 2.0])


def test_identity_predict_does_not_change_state():
    f = IdentityFilter(np.array([3.0, 4.0]))
    f.predict()
    np.testing.assert_allclose(f.state, [3.0, 4.0])


def test_identity_correct_updates_to_observation():
    f = IdentityFilter(np.array([0.0, 0.0]))
    f.correct(np.array([7.0, 8.0]))
    np.testing.assert_allclose(f.state, [7.0, 8.0])


def test_identity_covariance_is_large():
    f = IdentityFilter(np.array([0.0, 0.0]))
    assert np.all(np.diag(f.covariance) > 1e4)

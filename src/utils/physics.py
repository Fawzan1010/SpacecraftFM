"""Physics utilities for quaternion and attitude dynamics."""

import numpy as np
from typing import Tuple


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (q1 * q2).
    
    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]
    
    Returns:
        Product quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Return conjugate of quaternion.
    
    Args:
        q: Quaternion [w, x, y, z]
    
    Returns:
        Conjugate quaternion [w, -x, -y, -z]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_norm(q: np.ndarray) -> float:
    """Calculate norm of quaternion.
    
    Args:
        q: Quaternion [w, x, y, z]
    
    Returns:
        Quaternion norm
    """
    return np.linalg.norm(q)


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit norm.
    
    Args:
        q: Quaternion [w, x, y, z]
    
    Returns:
        Normalized quaternion
    """
    return q / quaternion_norm(q)


def quaternion_to_dcm(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to Direction Cosine Matrix (DCM).
    
    Args:
        q: Quaternion [w, x, y, z] (normalized)
    
    Returns:
        3x3 DCM (rotation matrix)
    """
    q = quaternion_normalize(q)
    w, x, y, z = q
    
    dcm = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)],
    ])
    
    return dcm


def dcm_to_quaternion(dcm: np.ndarray) -> np.ndarray:
    """Convert DCM to quaternion using Davenport's method.
    
    Args:
        dcm: 3x3 rotation matrix
    
    Returns:
        Quaternion [w, x, y, z]
    """
    trace = np.trace(dcm)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (dcm[2, 1] - dcm[1, 2]) * s
        y = (dcm[0, 2] - dcm[2, 0]) * s
        z = (dcm[1, 0] - dcm[0, 1]) * s
    elif dcm[0, 0] > dcm[1, 1] and dcm[0, 0] > dcm[2, 2]:
        s = 2.0 * np.sqrt(1.0 + dcm[0, 0] - dcm[1, 1] - dcm[2, 2])
        w = (dcm[2, 1] - dcm[1, 2]) / s
        x = 0.25 * s
        y = (dcm[0, 1] + dcm[1, 0]) / s
        z = (dcm[0, 2] + dcm[2, 0]) / s
    elif dcm[1, 1] > dcm[2, 2]:
        s = 2.0 * np.sqrt(1.0 + dcm[1, 1] - dcm[0, 0] - dcm[2, 2])
        w = (dcm[0, 2] - dcm[2, 0]) / s
        x = (dcm[0, 1] + dcm[1, 0]) / s
        y = 0.25 * s
        z = (dcm[1, 2] + dcm[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + dcm[2, 2] - dcm[0, 0] - dcm[1, 1])
        w = (dcm[1, 0] - dcm[0, 1]) / s
        x = (dcm[0, 2] + dcm[2, 0]) / s
        y = (dcm[1, 2] + dcm[2, 1]) / s
        z = 0.25 * s
    
    return quaternion_normalize(np.array([w, x, y, z]))


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to Euler angles (roll, pitch, yaw).
    
    Args:
        q: Quaternion [w, x, y, z]
    
    Returns:
        Euler angles [roll, pitch, yaw] in radians
    """
    dcm = quaternion_to_dcm(q)
    
    # Extract Euler angles from DCM
    pitch = np.arcsin(-dcm[2, 0])
    
    if np.cos(pitch) > 1e-6:
        roll = np.arctan2(dcm[2, 1], dcm[2, 2])
        yaw = np.arctan2(dcm[1, 0], dcm[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-dcm[0, 1], dcm[1, 1])
    
    return np.array([roll, pitch, yaw])


def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """Convert Euler angles to quaternion.
    
    Args:
        euler: Euler angles [roll, pitch, yaw] in radians
    
    Returns:
        Quaternion [w, x, y, z]
    """
    roll, pitch, yaw = euler
    
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])


def quaternion_error(q_true: np.ndarray, q_estimated: np.ndarray) -> float:
    """Calculate geodesic quaternion error.
    
    Args:
        q_true: True quaternion [w, x, y, z]
        q_estimated: Estimated quaternion [w, x, y, z]
    
    Returns:
        Geodesic error in radians
    """
    q_true = quaternion_normalize(q_true)
    q_estimated = quaternion_normalize(q_estimated)
    
    dot_product = np.clip(np.dot(q_true, q_estimated), -1.0, 1.0)
    error = 2.0 * np.arccos(np.abs(dot_product))
    
    return error


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Create skew-symmetric matrix from 3-vector.
    
    Args:
        v: 3-element vector
    
    Returns:
        3x3 skew-symmetric matrix
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def cross_product_matrix(v: np.ndarray) -> np.ndarray:
    """Alias for skew_symmetric."""
    return skew_symmetric(v)


def quaternion_kinematics(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Propagate quaternion using kinematic equation: dq/dt = 0.5 * Omega(omega) * q
    
    Args:
        q: Current quaternion [w, x, y, z]
        omega: Angular velocity [wx, wy, wz]
        dt: Time step
    
    Returns:
        Updated quaternion
    """
    # Omega matrix form
    omega_matrix = 0.5 * np.array([
        [0, -omega[0], -omega[1], -omega[2]],
        [omega[0], 0, omega[2], -omega[1]],
        [omega[1], -omega[2], 0, omega[0]],
        [omega[2], omega[1], -omega[0], 0],
    ])
    
    q_dot = omega_matrix @ q
    q_new = q + q_dot * dt
    
    return quaternion_normalize(q_new)


def rotate_vector(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector using quaternion.
    
    Args:
        v: 3-element vector to rotate
        q: Quaternion [w, x, y, z]
    
    Returns:
        Rotated 3-element vector
    """
    # Convert vector to quaternion form
    v_quat = np.array([0, v[0], v[1], v[2]])
    
    # Rotate: q * v * q*
    q_conj = quaternion_conjugate(q)
    v_rotated_quat = quaternion_multiply(quaternion_multiply(q, v_quat), q_conj)
    
    return v_rotated_quat[1:]

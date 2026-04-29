"""Geometry helpers for the air-combat demo."""
from __future__ import annotations

import math
import numpy as np


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = norm(v)
    if n < eps:
        return np.zeros_like(v, dtype=float)
    return v / n


def clamp(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))


def angle_of(v: np.ndarray) -> float:
    return math.atan2(float(v[1]), float(v[0]))


def wrap_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def rotate(v: np.ndarray, rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=float)


def heading_to_vector(theta: float) -> np.ndarray:
    return np.array([math.cos(theta), math.sin(theta)], dtype=float)


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    ua, ub = unit(a), unit(b)
    dot = clamp(float(np.dot(ua, ub)), -1.0, 1.0)
    return math.acos(dot)

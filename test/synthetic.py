# -*- coding: utf-8 -*-
"""
Synthetic but realistic time series for exercising change-point detection.

The series is an AR(1) process with Gaussian innovations whose parameters
change abruptly at known points: level shifts and a change of the noise
variance, the kinds of regime change ChangeFinder is designed for (Yamanishi
& Takeuchi).  Isolated outliers are injected so that a detector has to cope
with short spikes as well as genuine changes.  The AR process is continued
across segment boundaries, so the series looks like one measurement that
changes regime rather than a concatenation of unrelated signals.
"""
import numpy as np

# (length, mean, innovation std, AR(1) coefficient)
DEFAULT_SEGMENTS = [
    (400, 10.0, 1.0, 0.4),  # baseline
    (400, 15.0, 1.0, 0.4),  # level shift up (5 sigma)
    (400, 15.0, 3.0, 0.4),  # volatility increase (sigma x3)
    (400, 0.0, 3.0, 0.4),   # level shift down (5 sigma of the current regime)
]
# Isolated spikes, one per segment, away from the change points.
DEFAULT_OUTLIERS = [150, 620, 1050, 1350]


def generate_series(seed=0, segments=DEFAULT_SEGMENTS, outliers=DEFAULT_OUTLIERS,
                    outlier_sigma=4.0):
    """
    Returns ``(x, change_points, outlier_points)``.

    ``change_points`` are the first index of every segment after the first,
    ``outlier_points`` the indices of the injected spikes.  Spikes have a
    magnitude of ``outlier_sigma`` innovation standard deviations of the
    segment they fall in, with alternating sign.
    """
    rng = np.random.RandomState(seed)
    n = sum(length for length, _, _, _ in segments)
    x = np.zeros(n)
    sigma_at = np.zeros(n)
    change_points = []
    t = 0
    prev_dev = 0.0  # deviation from the segment mean at t-1
    for length, mean, std, phi in segments:
        if t > 0:
            change_points.append(t)
        e = rng.normal(scale=std, size=length)
        for i in range(length):
            prev_dev = phi * prev_dev + e[i]
            x[t + i] = mean + prev_dev
        sigma_at[t:t + length] = std
        t += length
    outliers = list(outliers or [])
    for k, i in enumerate(outliers):
        x[i] += outlier_sigma * sigma_at[i] * (1.0 if k % 2 == 0 else -1.0)
    return x, change_points, outliers

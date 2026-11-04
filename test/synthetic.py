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


# ---------------------------------------------------------------------------
# Daily business metric with weekly seasonality, trend and regime changes.
# ---------------------------------------------------------------------------

WEEKLY_PATTERN = np.array([-60.0, -80.0, -70.0, -50.0, 20.0, 130.0, 110.0])  # Mon .. Sun


def generate_daily_series(seed=0, n=730, level=1000.0, slope=0.3, phi=0.5, sigma=25.0,
                          weekly=WEEKLY_PATTERN, annual_amplitude=80.0,
                          level_shift=(200, 250.0), slope_change=(380, -0.7),
                          seasonal_change=(500, 0.4), variance_change=(600, 60.0),
                          outliers=((100, 400.0), (300, -500.0), (450, 350.0), (660, -300.0))):
    """
    A daily e-commerce-like metric: base level, weekly pattern (weekends
    high), a mild annual cycle, linear growth and AR(1) noise, with the
    regime changes a monitoring team would want to catch:

    * ``level_shift=(t, delta)``: a campaign or new channel adds ``delta``.
    * ``slope_change=(t, delta)``: growth changes by ``delta`` per day.
    * ``seasonal_change=(t, factor)``: the weekly pattern is scaled by ``factor``.
    * ``variance_change=(t, sigma)``: innovation std becomes ``sigma``.
    * ``outliers``: isolated spikes ``(t, delta)`` (flash sale, outage, holiday).

    Returns ``(x, changes, outlier_points)`` where ``changes`` maps the change
    type to its index.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    trend = level + slope * t
    if slope_change is not None:
        t0, delta = slope_change
        trend = trend + delta * np.clip(t - t0, 0, None)
    if level_shift is not None:
        t0, delta = level_shift
        trend = trend + delta * (t >= t0)
    season = weekly[t % 7].astype(np.float64).copy()
    if seasonal_change is not None:
        t0, factor = seasonal_change
        season[t >= t0] *= factor
    annual = annual_amplitude * np.sin(2.0 * np.pi * (t - 60) / 365.0)
    sig = np.full(n, sigma, dtype=np.float64)
    if variance_change is not None:
        t0, s = variance_change
        sig[t >= t0] = s
    e = rng.normal(size=n) * sig
    noise = np.zeros(n)
    for k in range(1, n):
        noise[k] = phi * noise[k - 1] + e[k]
    x = trend + season + annual + noise
    outlier_points = []
    for i, delta in (outliers or ()):
        x[i] += delta
        outlier_points.append(i)
    changes = {}
    if level_shift is not None:
        changes["level"] = level_shift[0]
    if slope_change is not None:
        changes["slope"] = slope_change[0]
    if seasonal_change is not None:
        changes["seasonal"] = seasonal_change[0]
    if variance_change is not None:
        changes["variance"] = variance_change[0]
    return x, changes, outlier_points

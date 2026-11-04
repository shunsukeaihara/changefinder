# -*- coding: utf-8 -*-
"""
OnlineSTL: online seasonal-trend decomposition.

Mishra, Sriharsha & Zhong, "OnlineSTL: Scaling Time Series Decomposition by
100x", PVLDB 15(7), 2022.  Non-symmetric tri-cube trend filters and
exponentially smoothed cyclic subseries are applied alternately to a sliding
window of the latest ``4 * m`` points (``m`` the largest period).
"""
import numpy as np


def tricube(u):
    """Tri-cube kernel W(u) = (1 - u^3)^3 on [0, 1), 0 elsewhere."""
    u = np.asarray(u, dtype=np.float64)
    w = np.where((u >= 0) & (u < 1), (1.0 - u ** 3) ** 3, 0.0)
    return w


def _nonsym_weights(lam):
    """Weights for the non-symmetric trend filter over the latest ``lam`` points.

    The newest point gets W(0) = 1, the oldest W((lam - 1) / lam)."""
    k = np.arange(1, lam + 1)
    w = tricube(np.abs(lam - k) / float(lam))
    return w / w.sum()


def trend_filter(x, weights):
    """Non-symmetric tri-cube trend estimate of the newest point of ``x``."""
    x = np.asarray(x, dtype=np.float64)
    lam = len(weights)
    return float(np.dot(weights, x[-lam:]))


def symmetric_trend_filter(x, half):
    """Symmetric tri-cube smoother used during initialization.

    Each point is smoothed over the ``half - 1`` points on either side
    (a window of ``2 * half - 1``), truncated at the ends of the array."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        left = max(0, i - half + 1)
        right = min(n - 1, i + half - 1)
        lam = right - left + 1
        w = tricube(np.abs(np.arange(left, right + 1) - i) / float(lam))
        out[i] = np.dot(w, x[left:right + 1]) / w.sum()
    return out


def seasonality_filter(d, period, gamma, phase0=0):
    """Exponentially smooth every cyclic subseries of ``d`` in time order.

    Element ``j`` belongs to the subseries ``(phase0 + j) % period``.
    Returns the smoothed series (same length as ``d``) and the last smoothed
    value of every subseries indexed by phase."""
    d = np.asarray(d, dtype=np.float64)
    out = np.empty(len(d))
    last = np.zeros(period)
    seen = np.zeros(period, dtype=bool)
    for j, v in enumerate(d):
        r = (phase0 + j) % period
        if seen[r]:
            last[r] = gamma * v + (1.0 - gamma) * last[r]
        else:
            last[r] = v
            seen[r] = True
        out[j] = last[r]
    return out, last


class OnlineSTL(object):
    """
    Online additive decomposition ``x = trend + sum(seasonal) + residual``.

    Parameters
    ----------
    periods : int or sequence of int
        Seasonal periods ``m_1 .. m_k``.
    gamma : float
        Smoothing factor of the seasonality filter (paper default 0.7).
        Closer to 1 makes the seasonal estimates follow the latest point.

    Call ``initialize(series)`` with at least ``4 * max(periods)`` points,
    then ``update(x)`` for every new point.  ``forecast()`` returns the
    one-step-ahead prediction ``trend + seasonal`` from the current state
    without looking at the next point.
    """

    def __init__(self, periods, gamma=0.7):
        if np.isscalar(periods):
            periods = [periods]
        self.periods = [int(p) for p in periods]
        assert len(self.periods) > 0 and all(p >= 2 for p in self.periods), "periods must be >= 2."
        assert 0.0 < gamma <= 1.0, "gamma must be in (0, 1]."
        self.gamma = float(gamma)
        self.m = max(self.periods)
        self.window = 4 * self.m
        self._w_trend1 = [_nonsym_weights(4 * p) for p in self.periods]
        self._w_trend4 = _nonsym_weights(3 * self.m)
        self._w_final = _nonsym_weights(self.m)
        self._n = 0  # number of points consumed
        self.initialized = False
        self.trend = 0.0

    def initialize(self, series):
        """Offline initialization on the latest ``4 * m`` points of ``series``."""
        series = np.asarray(series, dtype=np.float64)
        assert len(series) >= self.window, "initialization needs at least 4 * max(periods) points."
        A = series[-self.window:].copy()
        n0 = len(series)
        phase0 = n0 - self.window  # global index of A[0]
        self.A = A
        D = A.copy()
        deseasonalized = A.copy()
        self.K = []
        self.E_S = []
        self.E_T = []
        for p, w1 in zip(self.periods, self._w_trend1):
            trend1 = symmetric_trend_filter(A, p)
            T1 = deseasonalized - trend1
            K, last_S = seasonality_filter(T1, p, self.gamma, phase0)
            trend2 = symmetric_trend_filter(K, max(2, int(round(1.5 * p))))
            D5 = T1 - trend2
            Q, last_T = seasonality_filter(D5, p, self.gamma, phase0)
            self.K.append(K)
            self.E_S.append(last_S)
            self.E_T.append(last_T)
            deseasonalized = deseasonalized - Q
            D = D - Q
        self.D = D[-self.m:].copy()
        self._n = n0
        self.trend = trend_filter(self.D, self._w_final)
        self.initialized = True
        return self

    def _phase(self, p, index):
        return index % p

    def forecast(self):
        """One-step-ahead prediction (trend + seasonal) before seeing the next point."""
        assert self.initialized, "call initialize() first."
        return self.trend + sum(E[self._phase(p, self._n)] for p, E in zip(self.periods, self.E_T))

    def update(self, x):
        """Consume ``x`` and return ``(trend, seasonal, residual)``.

        ``seasonal`` is a numpy array with one entry per period."""
        assert self.initialized, "call initialize() first."
        x = float(x)
        self.A = np.append(self.A[1:], x)
        b = x
        S = np.empty(len(self.periods))
        for i, (p, w1) in enumerate(zip(self.periods, self._w_trend1)):
            t1 = trend_filter(self.A, w1)
            d1 = b - t1
            r = self._phase(p, self._n)
            self.E_S[i][r] = self.gamma * d1 + (1.0 - self.gamma) * self.E_S[i][r]
            self.K[i] = np.append(self.K[i][1:], self.E_S[i][r])
            t4 = trend_filter(self.K[i], self._w_trend4)
            d5 = b - t1 - t4
            self.E_T[i][r] = self.gamma * d5 + (1.0 - self.gamma) * self.E_T[i][r]
            b = b - self.E_T[i][r]
            S[i] = self.E_T[i][r]
        self.D = np.append(self.D[1:], b)
        self.trend = trend_filter(self.D, self._w_final)
        self._n += 1
        residual = x - self.trend - S.sum()
        return self.trend, S, residual

# -*- coding: utf-8 -*-
import math
import warnings

import numpy as np
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA

__version__ = "0.2"

# Lower bound for variances. Keeps log-likelihood computations finite when a
# series becomes (nearly) perfectly predictable and the estimated variance
# collapses to zero.
_EPS = 1e-12


def LevinsonDurbin(r, lpcOrder):
    """
    Solve the Yule-Walker equations with the Levinson-Durbin recursion.

    Returns ``(a, e)`` where ``a`` is the AR coefficient vector of length
    ``lpcOrder + 1`` with ``a[0] == 1`` and ``e`` is the final prediction
    error power.

    The recursion is guarded against degenerate inputs: a zero lag-0
    autocovariance returns the trivial predictor, and the recursion stops
    early (keeping the lower-order solution) when the prediction error
    vanishes or the reflection coefficient leaves ``(-1, 1)``.  The latter can
    happen because the SDAR autocovariance estimates are not guaranteed to
    form a positive definite Toeplitz matrix.

    Original implementation from
    http://aidiary.hatenablog.com/entry/20120415/1334458954
    """
    r = np.asarray(r, dtype=np.float64)
    assert lpcOrder >= 1, "lpcOrder must be 1 or more."
    assert len(r) >= lpcOrder + 1, "r must have lpcOrder + 1 elements or more."

    a = np.zeros(lpcOrder + 1, dtype=np.float64)
    a[0] = 1.0
    if not np.isfinite(r[0]) or r[0] <= _EPS:
        return a, 0.0

    lam = - r[1] / r[0]
    if not abs(lam) < 1.0:
        return a, r[0]
    a[1] = lam
    e = r[0] * (1.0 - lam * lam)

    for k in range(1, lpcOrder):
        if not e > _EPS:
            break
        lam = 0.0
        for j in range(k + 1):
            lam -= a[j] * r[k + 1 - j]
        lam /= e
        if not abs(lam) < 1.0:
            break

        prev = a[:k + 1].copy()
        a[1:k + 2] = prev[1:k + 1].tolist() + [0.0]
        a[1:k + 2] += lam * prev[k::-1]
        e *= (1.0 - lam * lam)

    return a, e


class _SDAR_1Dim(object):
    def __init__(self, r, order):
        assert 0.0 < r < 1.0, "r must be in (0, 1)."
        self._r = r
        self._mu = 0.0
        self._sigma = 0.0
        self._order = order
        self._c = np.zeros(self._order + 1, dtype=np.float64)
        self._initialized = False

    def update(self, x, term):
        assert len(term) >= self._order, "term must be order or more"
        term = np.asarray(term, dtype=np.float64)[-self._order:]
        x = float(x)
        if not self._initialized:
            # Start from the observed level instead of a random guess so that
            # results are deterministic and the burn-in transient is small.
            self._mu = float(np.mean(np.append(term, x)))
            self._initialized = True
        self._mu = (1.0 - self._r) * self._mu + self._r * x
        self._c[0] = (1.0 - self._r) * self._c[0] + self._r * (x - self._mu) * (x - self._mu)
        for i in range(1, self._order + 1):
            self._c[i] = (1.0 - self._r) * self._c[i] + self._r * (x - self._mu) * (term[-i] - self._mu)
        what, _ = LevinsonDurbin(self._c, self._order)
        xhat = np.dot(-what[1:], (term[::-1] - self._mu)) + self._mu
        resid = x - xhat
        self._sigma = (1.0 - self._r) * self._sigma + self._r * resid * resid
        sigma = max(self._sigma, _EPS)
        # Negative Gaussian log-likelihood, evaluated in log space so that it
        # never overflows/underflows.
        score = 0.5 * math.log(2.0 * math.pi * sigma) + 0.5 * resid * resid / sigma
        return score, xhat


class _ChangeFinderAbstract(object):
    def _add_one(self, one, ts, size):
        ts.append(one)
        if len(ts) == size + 1:
            ts.pop(0)

    def _smoothing(self, ts):
        return sum(ts) / float(len(ts))


class ChangeFinder(_ChangeFinderAbstract):
    def __init__(self, r=0.5, order=1, smooth=7):
        assert order > 0, "order must be 1 or more."
        assert smooth > 2, "term must be 3 or more."
        self._smooth = smooth
        self._smooth2 = int(round(self._smooth / 2.0))
        self._order = order
        self._r = r
        self._ts = []
        self._first_scores = []
        self._smoothed_scores = []
        self._second_scores = []
        self._sdar_first = _SDAR_1Dim(r, self._order)
        self._sdar_second = _SDAR_1Dim(r, self._order)

    def update(self, x):
        score = 0
        predict = x
        predict2 = 0
        if len(self._ts) == self._order:  # 第一段学習
            score, predict = self._sdar_first.update(x, self._ts)
            self._add_one(score, self._first_scores, self._smooth)
        self._add_one(x, self._ts, self._order)
        second_target = None
        if len(self._first_scores) == self._smooth:  # 平滑化
            second_target = self._smoothing(self._first_scores)
        if second_target is not None and len(self._smoothed_scores) == self._order:  # 第二段学習
            score, predict2 = self._sdar_second.update(second_target, self._smoothed_scores)
            self._add_one(score,
                          self._second_scores, self._smooth2)
        if second_target is not None:
            self._add_one(second_target, self._smoothed_scores, self._order)
        if len(self._second_scores) == self._smooth2:
            return self._smoothing(self._second_scores), predict
        else:
            return 0.0, predict


class ChangeFinderARIMA(_ChangeFinderAbstract):
    def __init__(self, term=30, smooth=7, order=(1, 0, 0)):
        assert smooth > 2, "term must be 3 or more."
        assert term > smooth, "term must be more than smooth"

        self._term = term
        self._smooth = smooth
        self._smooth2 = int(round(self._smooth / 2.0))
        self._order = order
        self._ts = []
        self._first_scores = []
        self._smoothed_scores = []
        self._second_scores = []

    def _calc_outlier_score(self, ts, target):
        def outlier_score(residuals, x):
            m = residuals.mean()
            s = np.std(residuals, ddof=1)
            if not np.isfinite(s) or s < math.sqrt(_EPS):
                s = math.sqrt(_EPS)
            return -stats.norm.logpdf(x, m, s)
        ts = np.asarray(ts, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arima_model = ARIMA(ts, order=self._order)
            result = arima_model.fit()
            pred = float(np.asarray(result.forecast(1)).ravel()[0])
        resid = np.asarray(result.resid, dtype=np.float64)
        return outlier_score(resid, x=target - pred), pred

    def update(self, x):
        score = 0
        predict = x
        predict2 = 0
        if len(self._ts) == self._term:  # 第一段学習
            try:
                score, predict = self._calc_outlier_score(self._ts, x)
                self._add_one(score, self._first_scores, self._smooth)
            except Exception:
                self._add_one(x, self._ts, self._term)
                return 0, predict
        self._add_one(x, self._ts, self._term)
        second_target = None
        if len(self._first_scores) == self._smooth:  # 平滑化
            second_target = self._smoothing(self._first_scores)
        if second_target is not None and len(self._smoothed_scores) == self._term:  # 第二段学習
            try:
                score, predict2 = self._calc_outlier_score(self._smoothed_scores, second_target)
                self._add_one(score,
                              self._second_scores, self._smooth2)
            except Exception:
                self._add_one(second_target, self._smoothed_scores, self._term)
                return 0, predict
        if second_target is not None:
            self._add_one(second_target, self._smoothed_scores, self._term)
        if len(self._second_scores) == self._smooth2:
            return self._smoothing(self._second_scores), predict
        else:
            return 0.0, predict

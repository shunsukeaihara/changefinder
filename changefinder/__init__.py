# -*- coding: utf-8 -*-
import copy
import math
import warnings

import numpy as np
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA

__version__ = "0.4"

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
    """
    Sequentially Discounting AR model estimation (Yamanishi & Takeuchi).

    ``update(x_t, term)`` scores ``x_t`` with the model learned from
    ``x_1 .. x_{t-1}`` (``-log p_{t-1}(x_t | x^{t-1})``) and then updates the
    model with ``x_t``.  ``term`` holds the ``order`` previous observations.
    """

    def __init__(self, r, order):
        assert 0.0 < r < 1.0, "r must be in (0, 1)."
        self._r = r
        self._order = order
        self._mu = 0.0
        self._sigma = 0.0
        self._c = np.zeros(self._order + 1, dtype=np.float64)
        self._a = np.zeros(self._order + 1, dtype=np.float64)
        self._a[0] = 1.0
        self._initialized = False

    def _predict(self, term):
        return np.dot(-self._a[1:], (term[::-1] - self._mu)) + self._mu

    def _score(self, x, xhat):
        resid = x - xhat
        sigma = max(self._sigma, _EPS)
        # Negative Gaussian log-likelihood, evaluated in log space so that it
        # never overflows/underflows.
        return 0.5 * math.log(2.0 * math.pi * sigma) + 0.5 * resid * resid / sigma

    def _learn(self, x, term):
        self._mu = (1.0 - self._r) * self._mu + self._r * x
        self._c[0] = (1.0 - self._r) * self._c[0] + self._r * (x - self._mu) * (x - self._mu)
        for i in range(1, self._order + 1):
            self._c[i] = (1.0 - self._r) * self._c[i] + self._r * (x - self._mu) * (term[-i] - self._mu)
        self._a, _ = LevinsonDurbin(self._c, self._order)
        xhat = self._predict(term)
        resid = x - xhat
        self._sigma = (1.0 - self._r) * self._sigma + self._r * resid * resid
        return xhat

    def update(self, x, term):
        assert len(term) >= self._order, "term must be order or more"
        term = np.asarray(term, dtype=np.float64)[-self._order:]
        x = float(x)
        if not self._initialized:
            # No model has been learned yet: start from the level of the
            # history, learn x_t and bootstrap the first score from that model.
            self._mu = float(np.mean(term))
            self._initialized = True
            xhat = self._learn(x, term)
            return self._score(x, xhat), xhat
        # Score x_t under p_{t-1}, then learn x_t.
        xhat = self._predict(term)
        score = self._score(x, xhat)
        self._learn(x, term)
        return score, xhat


class _SDARScorer(object):
    """First/second-stage scorer: SDAR on a window of ``order`` past values."""

    def __init__(self, r, order):
        assert order > 0, "order must be 1 or more."
        self._order = order
        self._ts = []
        self._sdar = _SDAR_1Dim(r, order)

    def score(self, x):
        result = None
        if len(self._ts) == self._order:
            result = self._sdar.update(x, self._ts)
        self._ts.append(x)
        if len(self._ts) > self._order:
            self._ts.pop(0)
        return result


class _ARIMAScorer(object):
    """Scorer that refits an ARIMA model on the last ``term`` values."""

    def __init__(self, term, order, seasonal_order=(0, 0, 0, 0), trend=None):
        p, d, q = order
        P, D, Q, s = seasonal_order
        if s == 0:
            assert (P, D, Q) == (0, 0, 0), "seasonal_order needs a period s > 0."
        needed = p + d + q + s * (P + D + Q) + 10
        assert term >= needed, "term must be at least %d for order=%s, seasonal_order=%s" % (
            needed, order, seasonal_order)
        self._term = term
        self._order = tuple(order)
        self._seasonal_order = tuple(seasonal_order)
        self._trend = trend
        self._ts = []
        self._warned = False

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
            arima_model = ARIMA(ts, order=self._order, seasonal_order=self._seasonal_order,
                                trend=self._trend)
            result = arima_model.fit()
            pred = float(np.asarray(result.forecast(1)).ravel()[0])
        resid = np.asarray(result.resid, dtype=np.float64)
        return outlier_score(resid, x=target - pred), pred

    def score(self, x):
        result = None
        if len(self._ts) == self._term:
            try:
                result = self._calc_outlier_score(self._ts, x)
            except Exception as exc:  # the model could not be fitted on this window
                if not self._warned:
                    warnings.warn("ARIMA fit failed, returning a zero score: %r" % (exc,),
                                  RuntimeWarning)
                    self._warned = True
                result = None
        self._ts.append(x)
        if len(self._ts) > self._term:
            self._ts.pop(0)
        return result


class _ChangeFinderAbstract(object):
    """
    Two-stage ChangeFinder driver.

    Stage one scores each observation with ``first``; the scores are
    smoothed over ``smooth`` steps; stage two scores the smoothed series with
    ``second``; the result is smoothed again over ``round(smooth / 2)`` steps.
    A scorer exposes ``score(x) -> (score, predict)`` and returns ``None``
    while it is still warming up.
    """

    def _init_stages(self, smooth, first, second):
        assert smooth > 2, "smooth must be 3 or more."
        self._smooth = smooth
        self._smooth2 = int(round(self._smooth / 2.0))
        self._first = first
        self._second = second
        self._first_scores = []
        self._second_scores = []

    def update(self, x, inplace=True):
        """
        Feed one observation and return ``(score, predict)``.

        With ``inplace=False`` the score is computed for ``x`` without
        learning it: the detector's state is left untouched, so repeated
        calls with the same observation return the same result.
        """
        if not inplace:
            return copy.deepcopy(self)._update(x)
        return self._update(x)

    def _update(self, x):
        predict = x
        first = self._first.score(x)
        if first is None:
            return 0.0, predict
        score, predict = first
        self._add_one(score, self._first_scores, self._smooth)
        if len(self._first_scores) < self._smooth:
            return 0.0, predict
        second_target = self._smoothing(self._first_scores)
        second = self._second.score(second_target)
        if second is None:
            return 0.0, predict
        self._add_one(second[0], self._second_scores, self._smooth2)
        if len(self._second_scores) == self._smooth2:
            return self._smoothing(self._second_scores), predict
        return 0.0, predict

    def _add_one(self, one, ts, size):
        ts.append(one)
        if len(ts) == size + 1:
            ts.pop(0)

    def _smoothing(self, ts):
        return sum(ts) / float(len(ts))


class ChangeFinder(_ChangeFinderAbstract):
    """ChangeFinder with SDAR models in both stages (Yamanishi & Takeuchi)."""

    def __init__(self, r=0.5, order=1, smooth=7):
        assert order > 0, "order must be 1 or more."
        self._order = order
        self._r = r
        self._init_stages(smooth, _SDARScorer(r, order), _SDARScorer(r, order))


class ChangeFinderSI(ChangeFinder):
    """
    ChangeFinder (SDAR) for series with trend and seasonality.

    The input is differenced ``d`` times and seasonally differenced ``D``
    times with period ``period`` before it reaches SDAR: the "I" and
    seasonal "I" parts of SARIMA.  Predictions are mapped back to the
    original scale.  The first ``d + D * period`` observations are a
    warm-up and score 0.0.

    Differencing turns a level shift into a single spike (and a seasonal
    difference echoes it ``period`` steps later), so shifts look more like
    outliers than they do for the state-space model.
    """

    def __init__(self, r=0.5, order=1, smooth=7, d=1, period=None, D=None):
        assert d >= 0, "d must be 0 or more."
        if D is None:
            D = 1 if period else 0
        assert D == 0 or (period is not None and period > 1), "D > 0 needs a period of 2 or more."
        super(ChangeFinderSI, self).__init__(r=r, order=order, smooth=smooth)
        self._lags = [int(period)] * int(D) + [1] * int(d)
        self._histories = [[] for _ in self._lags]

    def _update(self, x):
        v = float(x)
        bases = []
        for lag, hist in zip(self._lags, self._histories):
            if len(hist) < lag:
                hist.append(v)
                return 0.0, x
            base = hist[-lag]
            hist.append(v)
            if len(hist) > lag:
                hist.pop(0)
            bases.append(base)
            v = v - base
        score, predict = super(ChangeFinderSI, self)._update(v)
        return score, predict + sum(bases)


class ChangeFinderARIMA(_ChangeFinderAbstract):
    """
    ChangeFinder that refits a (seasonal) ARIMA model on the last ``term``
    observations at every step.

    ``order=(p, d, q)`` and ``seasonal_order=(P, D, Q, s)`` are passed to
    statsmodels' ARIMA; ``trend`` is one of ``'n'``, ``'c'``, ``'t'``,
    ``'ct'`` or None for the statsmodels default.  The second stage models
    the smoothed scores with the non-seasonal ``second_order`` (default:
    ``order``).  With the default arguments this is the original
    ChangeFinderARIMA.
    """

    def __init__(self, term=30, smooth=7, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0),
                 trend=None, second_order=None):
        assert term > smooth, "term must be more than smooth"
        self._term = term
        self._order = tuple(order)
        self._seasonal_order = tuple(seasonal_order)
        self._trend = trend
        second_order = tuple(order if second_order is None else second_order)
        self._init_stages(smooth,
                          _ARIMAScorer(term, self._order, self._seasonal_order, trend),
                          _ARIMAScorer(term, second_order))

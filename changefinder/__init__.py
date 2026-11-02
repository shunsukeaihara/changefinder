# -*- coding: utf-8 -*-
import copy
import math
import warnings

import numpy as np
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA

from .onlinestl import OnlineSTL

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
    The first ``ceil(1 / r)`` observations are learned with plain averages
    (warm start) and return ``None`` instead of a score.
    """

    def __init__(self, r, order, clip=3.0):
        assert 0.0 < r < 1.0, "r must be in (0, 1)."
        assert order >= 0, "order must be 0 or more."
        assert clip is None or clip > 0, "clip must be positive or None."
        self._r = r
        self._clip = clip
        self._order = order
        self._mu = 0.0
        self._sigma = 0.0
        self._c = np.zeros(self._order + 1, dtype=np.float64)
        self._a = np.zeros(self._order + 1, dtype=np.float64)
        self._a[0] = 1.0
        self._initialized = False
        self._t = 0
        # Learn only (no score) until the plain-average warm start is over.
        self.warmup = max(2, int(math.ceil(1.0 / r)))

    def _predict(self, term):
        return np.dot(-self._a[1:], (term[::-1] - self._mu)) + self._mu

    def _score(self, x, xhat):
        resid = x - xhat
        sigma = max(self._sigma, _EPS)
        # Negative Gaussian log-likelihood, evaluated in log space so that it
        # never overflows/underflows.
        return 0.5 * math.log(2.0 * math.pi * sigma) + 0.5 * resid * resid / sigma

    def _learn(self, x, term):
        # Warm start: plain averages until 1 / r observations have been seen,
        # then exponential forgetting with r.
        r = max(self._r, 1.0 / (self._t + 1.0))
        self._t += 1
        self._mu = (1.0 - r) * self._mu + r * x
        self._c[0] = (1.0 - r) * self._c[0] + r * (x - self._mu) * (x - self._mu)
        for i in range(1, self._order + 1):
            self._c[i] = (1.0 - r) * self._c[i] + r * (x - self._mu) * (term[-i] - self._mu)
        if self._order > 0:
            self._a, _ = LevinsonDurbin(self._c, self._order)
        xhat = self._predict(term)
        resid = x - xhat
        self._sigma = (1.0 - r) * self._sigma + r * resid * resid
        return xhat

    def update(self, x, term):
        assert len(term) >= self._order, "term must be order or more"
        term = np.asarray(term, dtype=np.float64)[len(term) - self._order:]
        x = float(x)
        if not self._initialized:
            # No model has been learned yet: start from the level of the
            # history (or of x_t for order 0) and learn x_t.
            self._mu = float(np.mean(term)) if self._order > 0 else x
            self._initialized = True
            self._learn(x, term)
            return None
        if self._t < self.warmup:
            # Warm start: plain averages are still being accumulated, the
            # variance estimate is not established yet, so learn without
            # scoring.
            self._learn(x, term)
            return None
        # Score x_t under p_{t-1}, then learn x_t.  The learning step sees a
        # winsorized observation (residual clipped at ``clip`` predictive
        # standard deviations) so that one outlier does not inflate the
        # variance estimate and mask the changes that follow.
        xhat = self._predict(term)
        score = self._score(x, xhat)
        if self._clip is not None and self._t * self._r >= 1.0:
            # Only once the warm start is over and the scale is established.
            bound = self._clip * math.sqrt(max(self._sigma, _EPS))
            x = xhat + max(-bound, min(bound, x - xhat))
        self._learn(x, term)
        return score, xhat


class _SDARScorer(object):
    """First/second-stage scorer: SDAR on a window of ``order`` past values."""

    def __init__(self, r, order, clip=3.0):
        assert order >= 0, "order must be 0 or more (0: no AR term, residual from the forgetting mean)."
        self._order = order
        self._ts = []
        self._sdar = _SDAR_1Dim(r, order, clip)

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

    def _init_stages(self, smooth, first, second, smoothing="mean"):
        assert smooth > 2, "smooth must be 3 or more."
        assert smoothing in ("mean", "median"), "smoothing must be 'mean' or 'median'."
        self._smooth = smooth
        self._smoothing_method = smoothing
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
        if self._smoothing_method == "median":
            return float(np.median(ts))
        return sum(ts) / float(len(ts))


class ChangeFinder(_ChangeFinderAbstract):
    """ChangeFinder with SDAR models in both stages (Yamanishi & Takeuchi)."""

    def __init__(self, r=0.5, order=1, smooth=7, clip=3.0, smoothing="mean"):
        assert order > 0, "order must be 1 or more."
        self._order = order
        self._r = r
        self._init_stages(smooth, _SDARScorer(r, order, clip), _SDARScorer(r, order, clip), smoothing)


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

    def __init__(self, r=0.5, order=1, smooth=7, d=1, period=None, D=None, clip=3.0, smoothing="mean"):
        assert d >= 0, "d must be 0 or more."
        if D is None:
            D = 1 if period else 0
        assert D == 0 or (period is not None and period > 1), "D > 0 needs a period of 2 or more."
        super(ChangeFinderSI, self).__init__(r=r, order=order, smooth=smooth, clip=clip, smoothing=smoothing)
        self._lags = ([int(period)] * int(D) if D else []) + [1] * int(d)
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
                 trend=None, second_order=None, smoothing="mean"):
        assert term > smooth, "term must be more than smooth"
        self._term = term
        self._order = tuple(order)
        self._seasonal_order = tuple(seasonal_order)
        self._trend = trend
        second_order = tuple(order if second_order is None else second_order)
        self._init_stages(smooth,
                          _ARIMAScorer(term, self._order, self._seasonal_order, trend),
                          _ARIMAScorer(term, second_order), smoothing)


class ChangeFinderSTL(ChangeFinder):
    """
    ChangeFinder (SDAR) on the one-step-ahead residual of an OnlineSTL
    decomposition (Mishra, Sriharsha & Zhong, PVLDB 2022).

    The first ``4 * max(periods)`` observations initialize the
    decomposition and score 0.0.  Afterwards every observation is scored
    against ``trend + seasonal`` predicted from the state before the
    observation is seen, which keeps the ChangeFinder convention of scoring
    ``x_t`` under the model learned up to ``t - 1``; the decomposition is
    then updated with ``x_t``.  ``predict`` is returned on the original
    scale.

    ``gamma`` is the smoothing factor of the seasonal estimates; the paper's
    default 0.7 follows the latest points closely, smaller values give
    steadier seasonal patterns and cleaner residuals.
    """

    def __init__(self, r=0.5, order=1, smooth=7, periods=7, gamma=0.7, clip=3.0, smoothing="mean"):
        super(ChangeFinderSTL, self).__init__(r=r, order=order, smooth=smooth, clip=clip, smoothing=smoothing)
        self._stl = OnlineSTL(periods, gamma)
        self._init_buffer = []

    @property
    def decomposition(self):
        """The underlying OnlineSTL object (trend, seasonal state)."""
        return self._stl

    def _update(self, x):
        x = float(x)
        if not self._stl.initialized:
            self._init_buffer.append(x)
            if len(self._init_buffer) == self._stl.window:
                self._stl.initialize(self._init_buffer)
                self._init_buffer = []
            return 0.0, x
        forecast = self._stl.forecast()
        self._stl.update(x)
        score, resid_hat = super(ChangeFinderSTL, self)._update(x - forecast)
        return score, forecast + resid_hat


class _DiscountKalman1Dim(object):
    """
    Structural time series model (level, optional slope, optional Fourier
    seasonal) estimated with a discount Kalman filter (West & Harrison).

    Instead of process-noise variances, every state block is inflated by
    ``1 / discount`` at each step; the observation variance is learned with
    the forgetting factor ``r``.  ``update(x)`` scores ``x`` with the
    one-step-ahead predictive distribution from the state before ``x`` is
    seen and then assimilates ``x``.  The first ``warmup`` observations fit
    the model by least squares and return ``None``.
    """

    def __init__(self, r, trend="level", period=None, harmonics=None, discount=0.98, clip=3.0):
        assert 0.0 < r < 1.0, "r must be in (0, 1)."
        assert trend in ("level", "linear"), "trend must be 'level' or 'linear'."
        assert clip is None or clip > 0, "clip must be positive or None."
        self._r = r
        self._clip = clip
        self._trend = trend
        self._period = None if period is None else int(period)
        if self._period is not None:
            assert self._period >= 2, "period must be 2 or more."
            if harmonics is None:
                harmonics = self._period // 2
            assert 1 <= harmonics <= self._period // 2, "harmonics must be in [1, period // 2]."
        self._harmonics = 0 if self._period is None else int(harmonics)
        if isinstance(discount, dict):
            d = {"level": 0.98, "slope": 0.98, "seasonal": 0.98}
            d.update(discount)
        else:
            d = {"level": float(discount), "slope": float(discount), "seasonal": float(discount)}
        for v in d.values():
            assert 0.0 < v <= 1.0, "discount must be in (0, 1]."
        self._discount = d
        self._build()
        self._buffer = []
        self.warmup = max(10, 2 * self._n + 2, 0 if self._period is None else 2 * self._period)
        self._m = None
        self._C = None
        self._V = 0.0
        self._t = 0

    def _build(self):
        blocks = ["level"]
        if self._trend == "linear":
            blocks.append("slope")
        self._freqs = []
        for j in range(1, self._harmonics + 1):
            w = 2.0 * math.pi * j / self._period
            nyquist = (self._period % 2 == 0) and (j == self._period // 2)
            self._freqs.append((w, nyquist))
            blocks.append("seasonal")
            if not nyquist:
                blocks.append("seasonal")
        n = len(blocks)
        G = np.zeros((n, n))
        F = np.zeros(n)
        G[0, 0] = 1.0
        F[0] = 1.0
        i = 1
        if self._trend == "linear":
            G[0, 1] = 1.0
            G[1, 1] = 1.0
            i = 2
        for w, nyquist in self._freqs:
            if nyquist:
                G[i, i] = -1.0
                F[i] = 1.0
                i += 1
            else:
                c, s = math.cos(w), math.sin(w)
                G[i, i], G[i, i + 1] = c, s
                G[i + 1, i], G[i + 1, i + 1] = -s, c
                F[i] = 1.0
                i += 2
        self._n = n
        self._G = G
        self._F = F
        self._blocks = blocks
        delta = np.array([self._discount[b] for b in blocks])
        self._inflate = 1.0 / np.sqrt(np.outer(delta, delta))

    def _design(self, t):
        """Observation vector of the model at time ``t`` for least-squares fitting."""
        row = [1.0]
        if self._trend == "linear":
            row.append(float(t))
        for w, nyquist in self._freqs:
            row.append(math.cos(w * t))
            if not nyquist:
                row.append(math.sin(w * t))
        return row

    def _initialize(self):
        y = np.asarray(self._buffer, dtype=np.float64)
        n = len(y)
        t = np.arange(n, dtype=np.float64)
        X = np.array([self._design(k) for k in t])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X.dot(beta)
        dof = max(n - X.shape[1], 1)
        V = float(np.dot(resid, resid) / dof)
        self._V = max(V, _EPS)
        # State at the end of the warm-up (time n - 1), in the rotating
        # Fourier representation used by the transition matrix.
        tend = float(n - 1)
        m = [beta[0] + (beta[1] * tend if self._trend == "linear" else 0.0)]
        i = 1
        if self._trend == "linear":
            m.append(beta[1])
            i = 2
        for w, nyquist in self._freqs:
            if nyquist:
                m.append(beta[i] * math.cos(w * tend))
                i += 1
            else:
                a, b = beta[i], beta[i + 1]
                m.append(a * math.cos(w * tend) + b * math.sin(w * tend))
                m.append(-a * math.sin(w * tend) + b * math.cos(w * tend))
                i += 2
        self._m = np.array(m)
        self._C = np.eye(self._n) * self._V
        self._t = n
        self._buffer = []

    def _predict(self):
        a = self._G.dot(self._m)
        R = self._G.dot(self._C).dot(self._G.T) * self._inflate
        f = float(self._F.dot(a))
        Q = float(self._F.dot(R).dot(self._F)) + max(self._V, _EPS)
        return a, R, f, Q

    def score(self, x):
        """Scorer interface: ``(score, predict)`` or None during warm-up."""
        return self.update(x)

    def update(self, x):
        x = float(x)
        if self._m is None:
            self._buffer.append(x)
            if len(self._buffer) == self.warmup:
                self._initialize()
            return None
        a, R, f, Q = self._predict()
        e = x - f
        score = 0.5 * math.log(2.0 * math.pi * Q) + 0.5 * e * e / Q
        if self._clip is not None and (self._t - self.warmup) * self._r >= 1.0:
            bound = self._clip * math.sqrt(Q)
            e = max(-bound, min(bound, e))
        A = R.dot(self._F) / Q
        self._m = a + A * e
        self._C = R - np.outer(A, A) * Q
        self._C = 0.5 * (self._C + self._C.T)
        V = max(self._V, _EPS)
        r = max(self._r, 1.0 / (self._t - self.warmup + 1.0))
        self._V = (1.0 - r) * V + r * V * e * e / Q
        self._t += 1
        return score, f

    def components(self):
        if self._m is None:
            return None
        out = {"level": float(self._m[0])}
        i = 1
        if self._trend == "linear":
            out["slope"] = float(self._m[1])
            i = 2
        seasonal = 0.0
        for w, nyquist in self._freqs:
            seasonal += self._m[i]
            i += 1 if nyquist else 2
        out["seasonal"] = float(seasonal)
        out["observation_variance"] = float(self._V)
        return out


class ChangeFinderSTS(_ChangeFinderAbstract):
    """
    ChangeFinder with a structural time series model (level, optional
    linear trend, optional Fourier seasonality) in the first stage,
    estimated online with a discount Kalman filter.

    Parameters
    ----------
    r : float
        Forgetting factor for the observation-noise variance (and for the
        second-stage SDAR).
    smooth : int
        Smoothing window ``T``.
    trend : "level" | "linear"
        Local level, or local linear trend (level + slope).
    period : int or None
        Seasonal period; None for no seasonal component.
    harmonics : int or None
        Number of Fourier harmonics; default ``period // 2`` (equivalent to
        a dummy-variable seasonal).  Fewer harmonics give a smoother pattern.
    discount : float or dict
        Discount factor(s) for the state blocks ``level``, ``slope``,
        ``seasonal``: how fast each component may drift (0.95 lets the
        level move noticeably within ~20 steps; 0.98-0.99 is stiffer).
    second_order : int
        AR order of the second-stage SDAR on the smoothed scores (default
        1; 0 uses no AR term, only the forgetting mean).
    clip : float or None
        Residuals beyond ``clip`` predictive standard deviations are
        winsorized in the learning step (not in the score), so an outlier
        does not inflate the variance and mask the changes that follow.
        None disables it.
    smoothing : "median" | "mean"
        How the first-stage scores are smoothed before the second stage.
        The median passes sustained changes but not isolated spikes, which
        keeps outliers from dominating the change-point score; "mean" is
        the original ChangeFinder smoothing.
    """

    def __init__(self, r=0.05, smooth=7, trend="level", period=None, harmonics=None,
                 discount=0.95, second_order=1, clip=3.0, smoothing="median"):
        self._r = r
        first = _DiscountKalman1Dim(r, trend=trend, period=period, harmonics=harmonics,
                                    discount=discount, clip=clip)
        self._init_stages(smooth, first, _SDARScorer(r, second_order, clip), smoothing)

    @property
    def warmup(self):
        return self._first.warmup

    def components(self):
        """Current level / slope / seasonal estimate and observation variance."""
        return self._first.components()

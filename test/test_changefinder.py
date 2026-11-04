# -*- coding: utf-8 -*-
import numpy as np
import pytest

import changefinder
from .synthetic import generate_daily_series, generate_series


def _run(detector, data):
    scores = []
    predicts = []
    for x in data:
        score, predict = detector.update(x)
        scores.append(score)
        predicts.append(predict)
    return np.asarray(scores, dtype=np.float64), np.asarray(predicts, dtype=np.float64)


def _assert_change_points_detected(scores, boundaries, window=40, quiet_offset=80):
    """Each boundary must produce a score peak that is clearly larger than the
    scores observed in the stationary segment that follows it."""
    for b in boundaries:
        peak = scores[b:b + window].max()
        quiet = scores[b + quiet_offset:b + 280]
        # The log-loss of an unseen level shift is orders of magnitude above
        # the scores produced by ordinary noise.
        assert peak > 100.0 and peak > 10.0 * quiet.max(), (
            "change point at %d not detected (peak=%f, quiet max=%f, quiet mean=%f, quiet std=%f)"
            % (b, peak, quiet.max(), quiet.mean(), quiet.std()))


class TestLevinsonDurbin(object):
    def test_recovers_ar1_coefficients(self):
        phi = 0.6
        r = [phi ** k for k in range(4)]
        a, e = changefinder.LevinsonDurbin(r, 1)
        np.testing.assert_allclose(a, [1.0, -phi])
        np.testing.assert_allclose(e, 1.0 - phi ** 2)
        a, e = changefinder.LevinsonDurbin(r, 3)
        np.testing.assert_allclose(a, [1.0, -phi, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(e, 1.0 - phi ** 2)

    def test_zero_autocovariance_returns_trivial_predictor(self):
        a, e = changefinder.LevinsonDurbin([0.0, 0.0, 0.0], 2)
        np.testing.assert_array_equal(a, [1.0, 0.0, 0.0])
        assert e == 0.0

    def test_invalid_autocovariance_stays_finite(self):
        # Not a valid (positive definite) autocovariance: |r[1]| > r[0].
        a, e = changefinder.LevinsonDurbin([1.0, 2.0, 0.5], 2)
        assert np.all(np.isfinite(a))
        assert np.isfinite(e)
        assert len(a) == 3


    def test_matches_direct_yule_walker_solution(self):
        from scipy.linalg import solve, toeplitz
        rng = np.random.RandomState(0)
        for p in (1, 2, 3, 5):
            while True:
                phi = -np.poly(rng.uniform(-0.9, 0.9, size=p))[1:]
                if np.all(np.abs(np.roots(np.r_[1, -phi])) < 1):
                    break
            n = 5000
            x = np.zeros(n)
            e = rng.normal(size=n)
            for t in range(p, n):
                x[t] = np.dot(phi, x[t - p:t][::-1]) + e[t]
            r = np.array([np.dot(x[:n - l], x[l:]) / n for l in range(p + 1)])
            a, ev = changefinder.LevinsonDurbin(r, p)
            w = solve(toeplitz(r[:p]), r[1:p + 1])
            np.testing.assert_allclose(-a[1:], w, atol=1e-8)
            np.testing.assert_allclose(ev, r[0] - np.dot(w, r[1:p + 1]), atol=1e-8)


class TestSDAR(object):
    def test_recovers_ar1_parameters(self):
        rng = np.random.RandomState(0)
        phi, mu, var = 0.7, 3.0, 0.25
        n = 20000
        x = np.zeros(n)
        e = rng.normal(scale=np.sqrt(var), size=n)
        for t in range(1, n):
            x[t] = mu + phi * (x[t - 1] - mu) + e[t]
        sdar = changefinder._SDAR_1Dim(0.01, 1)
        for t in range(1, n):
            sdar.update(x[t], [x[t - 1]])
        a, _ = changefinder.LevinsonDurbin(sdar._c, 1)
        assert abs(sdar._mu - mu) < 0.3
        assert abs(-a[1] - phi) < 0.05
        assert abs(sdar._sigma - var) < 0.05

    def test_prediction_exploits_autocorrelation(self):
        # Guards the sign of the AR coefficients in the prediction: with the
        # sign flipped the forecast would be anti-correlated with the series
        # and its residuals larger than the series' own variance.
        rng = np.random.RandomState(0)
        phi, n = 0.7, 5000
        x = np.zeros(n)
        e = rng.normal(size=n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + e[t]
        sdar = changefinder._SDAR_1Dim(0.02, 1)
        resid = []
        for t in range(1, n):
            result = sdar.update(x[t], [x[t - 1]])
            if t > 500:
                resid.append(x[t] - result[1])
        assert np.var(resid) < 0.6 * np.var(x[500:])

    def test_score_is_negative_gaussian_log_likelihood(self):
        from scipy import stats
        rng = np.random.RandomState(0)
        xs = rng.normal(size=50)
        sdar = changefinder._SDAR_1Dim(0.1, 2)
        for t in range(2, 49):
            sdar.update(xs[t], xs[t - 2:t])
        # The score of x_t uses the variance (and prediction) learned from
        # x_1 .. x_{t-1}, i.e. the state before x_t is learned.
        sigma_prev = sdar._sigma
        score, xhat = sdar.update(xs[49], xs[47:49])
        ref = -stats.norm.logpdf(xs[49], loc=xhat, scale=np.sqrt(sigma_prev))
        np.testing.assert_allclose(score, ref)

    def test_scores_before_learning(self):
        # A step that the model has not seen yet must be scored as surprising,
        # even though learning it immediately afterwards would absorb it.
        sdar = changefinder._SDAR_1Dim(0.5, 1)
        rng = np.random.RandomState(0)
        xs = rng.normal(size=100)
        for t in range(1, 100):
            sdar.update(xs[t], [xs[t - 1]])
        typical, _ = sdar.update(0.5, [xs[-1]])
        surprising, _ = sdar.update(50.0, [0.5])
        assert surprising > typical + 100.0


class TestChangeFinder(object):
    def setup_method(self, method):
        self._smooth = 7
        self._order = 1
        rng = np.random.RandomState(0)
        self._data = np.concatenate([rng.rand(300) + 5,
                                     rng.rand(300) + 10,
                                     rng.rand(300) + 5,
                                     rng.rand(300)])
        self._boundaries = [300, 600, 900]

    def test_detects_change_points(self):
        cf = changefinder.ChangeFinder(r=0.1, order=self._order, smooth=self._smooth)
        scores, predicts = _run(cf, self._data)
        assert scores.shape == self._data.shape
        assert np.all(np.isfinite(scores))
        assert np.all(np.isfinite(predicts))
        _assert_change_points_detected(scores, self._boundaries)
        # In a stationary segment the one-step prediction tracks the level.
        assert abs(predicts[250:300].mean() - self._data[250:300].mean()) < 0.5

    @pytest.mark.parametrize("order", [1, 2, 3, 5])
    def test_higher_order_detects_change_points(self, order):
        cf = changefinder.ChangeFinder(r=0.1, order=order, smooth=self._smooth)
        scores, _ = _run(cf, self._data)
        assert np.all(np.isfinite(scores))
        _assert_change_points_detected(scores, self._boundaries)

    def test_deterministic(self):
        cf1 = changefinder.ChangeFinder(r=0.1, order=self._order, smooth=self._smooth)
        cf2 = changefinder.ChangeFinder(r=0.1, order=self._order, smooth=self._smooth)
        s1, p1 = _run(cf1, self._data)
        s2, p2 = _run(cf2, self._data)
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(p1, p2)

    def test_no_change_gives_low_scores(self):
        rng = np.random.RandomState(1)
        data = rng.rand(600) + 5
        cf = changefinder.ChangeFinder(r=0.1, order=self._order, smooth=self._smooth)
        scores, _ = _run(cf, data)
        assert np.all(np.isfinite(scores))
        # The first steps return 0.0 until the buffers are filled.
        assert np.all(scores[:self._order * 2 + self._smooth] == 0.0)
        # After burn-in nothing should look like a change.
        assert scores[100:].max() < 20.0

    @pytest.mark.parametrize("order", [1, 2, 3, 5])
    @pytest.mark.parametrize("name", ["constant", "zeros", "alternating", "step", "huge", "tiny"])
    def test_degenerate_inputs_stay_finite(self, order, name):
        rng = np.random.RandomState(2)
        data = {
            "constant": np.ones(300),
            "zeros": np.zeros(300),
            "alternating": np.tile([0.0, 1.0], 150),
            "step": np.concatenate([np.zeros(100), np.ones(100) * 5, np.zeros(100)]),
            "huge": rng.rand(300) * 1e8,
            "tiny": rng.rand(300) * 1e-8,
        }[name]
        cf = changefinder.ChangeFinder(r=0.1, order=order, smooth=self._smooth)
        scores, predicts = _run(cf, data)
        assert np.all(np.isfinite(scores))
        assert np.all(np.isfinite(predicts))

    def test_step_change_is_detected_with_perfectly_predictable_history(self):
        data = np.concatenate([np.ones(200), np.ones(200) * 3])
        cf = changefinder.ChangeFinder(r=0.1, order=self._order, smooth=self._smooth)
        scores, _ = _run(cf, data)
        assert np.all(np.isfinite(scores))
        assert np.argmax(scores) >= 200

    def test_update_not_inplace_leaves_state_untouched(self):
        cf = changefinder.ChangeFinder(r=0.1, order=self._order, smooth=self._smooth)
        _run(cf, self._data[:100])
        snapshot = _state(cf)
        peek1 = cf.update(self._data[100], inplace=False)
        peek2 = cf.update(self._data[100], inplace=False)
        assert peek1 == peek2
        assert _state(cf) == snapshot
        committed = cf.update(self._data[100])
        assert committed == peek1

    def test_invalid_parameters(self):
        with pytest.raises(AssertionError):
            changefinder.ChangeFinder(order=0)
        with pytest.raises(AssertionError):
            changefinder.ChangeFinder(smooth=2)
        with pytest.raises(AssertionError):
            changefinder.ChangeFinder(r=0.0)
        with pytest.raises(AssertionError):
            changefinder.ChangeFinder(r=1.0)


def _state(obj):
    """Recursively convert a detector's state into plain Python data."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_state(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _state(v) for k, v in obj.items()}
    if type(obj).__module__.startswith("statsmodels"):
        return "<%s>" % type(obj).__name__   # fitted results carry lazy caches
    if hasattr(obj, "__dict__"):
        return _state(vars(obj))
    return obj


def _quiet_scores(scores, change_points, outliers, start=150, window=60):
    """Scores outside the windows that follow a change point or an outlier."""
    excluded = set()
    for b in list(change_points) + list(outliers):
        excluded.update(range(b, b + window))
    return np.array([scores[i] for i in range(start, len(scores)) if i not in excluded])


class TestChangeFinderNaturalSeries(object):
    """
    Detection on an autocorrelated series with realistic regime changes
    (see test/synthetic.py).  Notes from tuning these tests:

    * order=1 (or 2) is the sweet spot for natural data.  With forgetting
      factor r the effective sample size is ~1/r, so higher orders overfit:
      the residual variance is underestimated, ordinary residuals score too
      high and the change-point peaks shrink.
    * A change of the AR coefficient alone (same mean and innovation
      variance) is not discussed in the ChangeFinder papers.  Empirically
      it is a weak signal for this implementation: with phi changing from
      0.4 to -0.5 (or 0.6 to -0.6) the peak never exceeded the stationary
      scores over 6 seeds; even for 0.8 to -0.8 it did so in only about half
      of the seeds (order 1-2, r 0.03-0.05).  The one-step-ahead residual
      variance only grows by a factor (1 + (phi_new - phi_old)^2 var(x)),
      and the SDAR coefficient adapts within ~1/r samples, so the smoothed
      score ramps up gently instead of jumping.  See the xfail test below.
    * A level shift after a volatile regime is hidden by the inflated
      variance estimate unless it is large in units of the current sigma.
    """

    def setup_method(self, method):
        self._data, self._cps, self._outliers = generate_series(seed=0)

    @pytest.mark.parametrize("r", [0.03, 0.05])
    def test_detects_regime_changes(self, r):
        cf = changefinder.ChangeFinder(r=r, order=1, smooth=7)
        scores, predicts = _run(cf, self._data)
        assert np.all(np.isfinite(scores))
        assert np.all(np.isfinite(predicts))
        quiet = _quiet_scores(scores, self._cps, self._outliers)
        p99 = np.percentile(quiet, 99)
        for b in self._cps:
            window = scores[b:b + 50]
            peak, lag = window.max(), int(np.argmax(window))
            assert peak > quiet.max() and peak > 3.0 * p99, (
                "change point at %d not detected (peak=%f, quiet max=%f, p99=%f)" % (b, peak, quiet.max(), p99))
            assert lag <= 10, "change point at %d detected too late (lag=%d)" % (b, lag)

    def test_outliers_are_transient(self):
        cf = changefinder.ChangeFinder(r=0.03, order=1, smooth=7)
        scores, _ = _run(cf, self._data)
        quiet = _quiet_scores(scores, self._cps, self._outliers)
        p99 = np.percentile(quiet, 99)
        for o in self._outliers:
            # Whatever a spike scores, the detector is back to normal shortly after.
            assert scores[o + 20:o + 60].max() < p99, "score did not recover after outlier at %d" % o

    def test_high_order_underestimates_residual_variance(self):
        x, _, _ = generate_series(seed=0, segments=[(3000, 10.0, 1.0, 0.4)], outliers=[])
        errors = {}
        for order in (1, 8):
            cf = changefinder.ChangeFinder(r=0.03, order=order, smooth=7)
            _run(cf, x)
            errors[order] = abs(cf._first._sdar._sigma - 1.0)
        assert errors[1] < errors[8]

    @pytest.mark.xfail(reason="empirical: a moderate change of the AR coefficient alone is not reliably "
                              "detected by this implementation (not a claim of the original papers)",
                       strict=False)
    def test_dynamics_only_change(self):
        x, cps, outliers = generate_series(seed=0, segments=[(400, 10.0, 1.0, 0.4), (400, 10.0, 1.0, -0.5)],
                                           outliers=[])
        cf = changefinder.ChangeFinder(r=0.03, order=1, smooth=7)
        scores, _ = _run(cf, x)
        quiet = _quiet_scores(scores, cps, outliers)
        peak = scores[cps[0]:cps[0] + 50].max()
        assert peak > quiet.max() and peak > 3.0 * np.percentile(quiet, 99)


class TestChangeFinderARIMA(object):
    def setup_method(self, method):
        self._term = 30
        self._smooth = 7
        self._arima_order = (1, 0, 0)
        rng = np.random.RandomState(0)
        # Shorter segments than the SDAR test: every update fits an ARIMA model.
        self._data = np.concatenate([rng.rand(120) + 5,
                                     rng.rand(120) + 10,
                                     rng.rand(120) + 5])
        self._boundaries = [120, 240]

    def test_detects_change_points(self):
        cf = changefinder.ChangeFinderARIMA(term=self._term, smooth=self._smooth, order=self._arima_order)
        scores, predicts = _run(cf, self._data)
        assert scores.shape == self._data.shape
        assert np.all(np.isfinite(scores))
        assert np.all(np.isfinite(predicts))
        # The ARIMA model must actually be fitted (no silent failures).
        warmup = self._term * 2 + self._smooth + self._smooth
        assert np.any(scores[warmup:] != 0.0)
        for b in self._boundaries:
            peak = scores[b:b + 40].max()
            quiet = scores[b + 60:b + 110].max()
            assert peak > 100.0 and peak > 10.0 * quiet, (
                "change point at %d not detected (peak=%f, quiet max=%f)" % (b, peak, quiet))
        assert abs(predicts[100:120].mean() - self._data[100:120].mean()) < 0.5

    def test_detects_regime_changes_on_natural_series(self):
        segments = [(120, 10.0, 1.0, 0.4), (120, 15.0, 1.0, 0.4), (120, 15.0, 3.0, 0.4), (120, 0.0, 3.0, 0.4)]
        data, cps, outliers = generate_series(seed=0, segments=segments, outliers=[60, 180, 300, 420])
        cf = changefinder.ChangeFinderARIMA(term=self._term, smooth=self._smooth, order=self._arima_order)
        scores, predicts = _run(cf, data)
        assert np.all(np.isfinite(scores))
        assert np.all(np.isfinite(predicts))
        quiet = _quiet_scores(scores, cps, outliers, start=100, window=40)
        p99 = np.percentile(quiet, 99)
        for b in cps:
            window = scores[b:b + 40]
            peak, lag = window.max(), int(np.argmax(window))
            assert peak > quiet.max() and peak > 2.0 * p99, (
                "change point at %d not detected (peak=%f, quiet max=%f, p99=%f)" % (b, peak, quiet.max(), p99))
            assert lag <= 10, "change point at %d detected too late (lag=%d)" % (b, lag)

    def test_update_not_inplace_leaves_state_untouched(self):
        cf = changefinder.ChangeFinderARIMA(term=self._term, smooth=self._smooth, order=self._arima_order)
        _run(cf, self._data[:60])
        snapshot = _state(cf)
        peek = cf.update(self._data[60], inplace=False)
        assert _state(cf) == snapshot
        assert cf.update(self._data[60]) == peek

    def test_failed_fit_reuses_previous_parameters(self):
        import statsmodels.tsa.arima.model as arima_model
        original_fit = arima_model.ARIMA.fit
        calls = {"n": 0}

        def flaky_fit(model, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] > 1:
                raise ValueError("forced failure")
            return original_fit(model, *args, **kwargs)

        arima_model.ARIMA.fit = flaky_fit
        try:
            cf = changefinder.ChangeFinderARIMA(term=self._term, smooth=self._smooth, order=self._arima_order)
            with pytest.warns(RuntimeWarning, match="reusing"):
                scores, predicts = _run(cf, self._data[:150])
        finally:
            arima_model.ARIMA.fit = original_fit
        assert np.all(np.isfinite(scores)) and np.all(np.isfinite(predicts))
        assert cf.fit_failures == 150 - self._term - 1
        assert np.any(scores[self._term + 2 * self._smooth:] != 0.0)

    def test_window_is_winsorized(self):
        rng = np.random.RandomState(0)
        x = rng.normal(size=80) + 5
        x[50] += 50.0
        cf = changefinder.ChangeFinderARIMA(term=self._term, smooth=self._smooth, order=self._arima_order)
        _run(cf, x[:51])
        stored = cf._first._ts[-1]
        assert abs(stored - 5.0) < 10.0 < abs(x[50] - 5.0)
        cf = changefinder.ChangeFinderARIMA(term=self._term, smooth=self._smooth, order=self._arima_order, clip=None)
        _run(cf, x[:51])
        assert cf._first._ts[-1] == x[50]

    def test_outlier_score_is_finite_on_constant_history(self):
        cf = changefinder.ChangeFinderARIMA(term=self._term, smooth=self._smooth, order=self._arima_order)
        scores, predicts = _run(cf, np.ones(100))
        assert np.all(np.isfinite(scores))
        assert np.all(np.isfinite(predicts))

    def test_invalid_parameters(self):
        with pytest.raises(AssertionError):
            changefinder.ChangeFinderARIMA(smooth=2)
        with pytest.raises(AssertionError):
            changefinder.ChangeFinderARIMA(term=5, smooth=7)
        with pytest.raises(AssertionError):
            changefinder.ChangeFinderARIMA(term=15, smooth=7, seasonal_order=(0, 1, 0, 7))
        with pytest.raises(AssertionError):
            changefinder.ChangeFinderARIMA(seasonal_order=(1, 0, 0, 0))


def _trend_seasonal_series(seed=0, n=700, period=7, noise=0.5, shift=(400, 5.0)):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    season = 3.0 * np.sin(2 * np.pi * t / period) + 1.0 * np.cos(4 * np.pi * t / period)
    x = 10.0 + 0.02 * t + season + rng.normal(scale=noise, size=n)
    if shift is not None:
        x[shift[0]:] += shift[1]
    return x, 10.0 + 0.02 * t, season


def _assert_shift_detected(scores, at, quiet_ranges, min_ratio=3.0):
    quiet = np.concatenate([scores[a:b] for a, b in quiet_ranges])
    peak = scores[at:at + 30].max()
    assert peak > quiet.max() and peak > min_ratio * max(np.percentile(quiet, 99), 0.5), (
        "shift at %d not detected (peak=%f, quiet max=%f)" % (at, peak, quiet.max()))


class TestOnlineSTL(object):
    def test_decomposes_trend_and_seasonality(self):
        from changefinder.onlinestl import OnlineSTL
        x, trend, season = _trend_seasonal_series(shift=None)
        stl = OnlineSTL(7, gamma=0.7).initialize(x[:28])
        T, S, R, F = [], [], [], []
        for v in x[28:]:
            F.append(stl.forecast())
            tt, ss, rr = stl.update(v)
            T.append(tt)
            S.append(ss.sum())
            R.append(rr)
        T, S, R, F = (np.array(a)[100:] for a in (T, S, R, F))
        assert np.std(R) < 0.5                                   # below the noise level
        assert np.std(T - trend[28:][100:]) < 0.3
        assert np.std(S - season[28:][100:]) < 0.8
        assert np.std(x[28:][100:] - F) < 0.3 * np.std(x)        # forecast beats the raw spread

    def test_multiple_periods(self):
        from changefinder.onlinestl import OnlineSTL
        rng = np.random.RandomState(1)
        t = np.arange(900)
        x = 5 + 2 * np.sin(2 * np.pi * t / 7) + 3 * np.sin(2 * np.pi * t / 30) + rng.normal(scale=0.3, size=900)
        stl = OnlineSTL([7, 30]).initialize(x[:120])
        R = np.array([stl.update(v)[2] for v in x[120:]])
        assert np.all(np.isfinite(R))
        assert np.std(R[200:]) < 0.4

    def test_needs_four_periods_to_initialize(self):
        from changefinder.onlinestl import OnlineSTL
        with pytest.raises(AssertionError):
            OnlineSTL(10).initialize(np.zeros(39))


class TestChangeFinderSI(object):
    def test_warmup_and_original_scale_predictions(self):
        x, _, _ = _trend_seasonal_series()
        cf = changefinder.ChangeFinderSI(r=0.05, order=1, d=1, period=7)
        scores, predicts = _run(cf, x)
        assert np.all(scores[:8] == 0.0) and np.all(predicts[:8] == x[:8])
        assert np.std(x[150:400] - predicts[150:400]) < 0.2 * np.std(x)

    def test_detects_level_shift_on_seasonal_series(self):
        x, _, _ = _trend_seasonal_series()
        cf = changefinder.ChangeFinderSI(r=0.05, order=1, d=1, period=7)
        scores, _ = _run(cf, x)
        assert np.all(np.isfinite(scores))
        _assert_shift_detected(scores, 400, [(150, 400), (460, 700)])

    def test_default_seasonal_difference_follows_period(self):
        assert changefinder.ChangeFinderSI(period=7)._lags == [7, 1]
        assert changefinder.ChangeFinderSI()._lags == [1]
        assert changefinder.ChangeFinderSI(d=0, period=12, D=1)._lags == [12]
        with pytest.raises(AssertionError):
            changefinder.ChangeFinderSI(D=1)


class TestChangeFinderSTL(object):
    def test_detects_level_shift_on_seasonal_series(self):
        x, _, _ = _trend_seasonal_series()
        cf = changefinder.ChangeFinderSTL(r=0.05, order=1, periods=7, gamma=0.3)
        scores, predicts = _run(cf, x)
        assert np.all(np.isfinite(scores))
        assert np.all(scores[:28] == 0.0)
        assert cf.decomposition.initialized
        assert np.std(x[150:400] - predicts[150:400]) < 0.2 * np.std(x)
        _assert_shift_detected(scores, 400, [(150, 400), (460, 700)])


class TestChangeFinderSTS(object):
    def test_recovers_level_slope_and_noise(self):
        x, trend, _ = _trend_seasonal_series(shift=None)
        cf = changefinder.ChangeFinderSTS(r=0.05, trend="linear", period=7)
        scores, predicts = _run(cf, x)
        assert np.all(scores[:cf.warmup] == 0.0)
        c = cf.components()
        assert abs(c["slope"] - 0.02) < 0.01
        assert abs(c["level"] - trend[-1]) < 1.0
        assert 0.1 < c["observation_variance"] < 0.6
        assert np.std(x[150:] - predicts[150:]) < 0.7

    def test_detects_level_shift_on_seasonal_series(self):
        x, _, _ = _trend_seasonal_series()
        cf = changefinder.ChangeFinderSTS(r=0.05, trend="linear", period=7)
        scores, _ = _run(cf, x)
        assert np.all(np.isfinite(scores))
        _assert_shift_detected(scores, 400, [(150, 400), (460, 700)])

    def test_realistic_daily_series(self):
        x, changes, outliers = generate_daily_series(seed=0)
        cf = changefinder.ChangeFinderSTS(r=0.03, smooth=7, trend="linear", period=7)
        scores, _ = _run(cf, x)
        assert np.all(np.isfinite(scores))
        quiet = _quiet_scores(scores, list(changes.values()), outliers, start=120, window=40)
        p99 = max(np.percentile(quiet, 99), 0.5)
        for kind in ("level", "seasonal", "variance"):
            b = changes[kind]
            peak = scores[b:b + 30].max()
            assert peak > quiet.max() and peak > 3.0 * p99, (
                "%s change at %d not detected (peak=%f, quiet max=%f)" % (kind, b, peak, quiet.max()))
        # Isolated spikes (flash sale, outage) are not reported as changes.
        for o in outliers:
            assert scores[o:o + 30].max() < 3.0 * p99

    @pytest.mark.xfail(reason="a change of growth rate adds only a fraction of the noise sd per step; "
                              "per-step predictive scores cannot see it, see README",
                       strict=False)
    def test_realistic_daily_series_slope_change(self):
        x, changes, outliers = generate_daily_series(seed=0)
        cf = changefinder.ChangeFinderSTS(r=0.03, smooth=7, trend="linear", period=7)
        scores, _ = _run(cf, x)
        quiet = _quiet_scores(scores, list(changes.values()), outliers, start=120, window=40)
        b = changes["slope"]
        assert scores[b:b + 30].max() > quiet.max()

    def test_second_order_zero(self):
        x, _, _ = _trend_seasonal_series()
        cf = changefinder.ChangeFinderSTS(r=0.05, trend="linear", period=7, second_order=0)
        scores, _ = _run(cf, x)
        assert np.all(np.isfinite(scores))
        _assert_shift_detected(scores, 400, [(150, 400), (460, 700)])

    def test_outlier_does_not_mask_following_shift(self):
        # With clip=None a 15-sigma spike inflates the variance estimate and
        # hides a 10-sigma level shift 60 steps later; the default clipping keeps it visible.
        x, _, _ = _trend_seasonal_series()
        x[340] += 7.5
        peaks = {}
        for clip in (None, 3.0):
            cf = changefinder.ChangeFinderSTS(r=0.05, trend="linear", period=7, clip=clip)
            scores, _ = _run(cf, x)
            peaks[clip] = scores[400:430].max()
        assert peaks[3.0] > 3.0 * peaks[None]

    def test_options(self):
        for kw in (dict(trend="level"), dict(period=12, harmonics=3), dict(period=8),
                   dict(discount={"level": 0.9}), dict(smoothing="mean", clip=None)):
            cf = changefinder.ChangeFinderSTS(r=0.1, **kw)
            x, _, _ = _trend_seasonal_series(n=200)
            scores, _ = _run(cf, x)
            assert np.all(np.isfinite(scores))
        with pytest.raises(AssertionError):
            changefinder.ChangeFinderSTS(trend="quadratic")
        with pytest.raises(AssertionError):
            changefinder.ChangeFinderSTS(period=7, harmonics=5)

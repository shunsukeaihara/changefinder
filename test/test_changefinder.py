# -*- coding: utf-8 -*-
import numpy as np
import pytest

import changefinder


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
        assert peak > quiet.max() + 0.5 and peak > quiet.mean() + 4.0 * quiet.std(), (
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

    def test_score_is_negative_gaussian_log_likelihood(self):
        from scipy import stats
        rng = np.random.RandomState(0)
        xs = rng.normal(size=50)
        sdar = changefinder._SDAR_1Dim(0.1, 2)
        for t in range(2, 50):
            score, xhat = sdar.update(xs[t], xs[t - 2:t])
        ref = -stats.norm.logpdf(xs[49], loc=xhat, scale=np.sqrt(sdar._sigma))
        np.testing.assert_allclose(score, ref)


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
        assert scores[100:].max() < 1.0

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

    def test_invalid_parameters(self):
        with pytest.raises(AssertionError):
            changefinder.ChangeFinder(order=0)
        with pytest.raises(AssertionError):
            changefinder.ChangeFinder(smooth=2)
        with pytest.raises(AssertionError):
            changefinder.ChangeFinder(r=0.0)
        with pytest.raises(AssertionError):
            changefinder.ChangeFinder(r=1.0)


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
            assert peak > quiet + 1.0, (
                "change point at %d not detected (peak=%f, quiet max=%f)" % (b, peak, quiet))
        assert abs(predicts[100:120].mean() - self._data[100:120].mean()) < 0.5

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

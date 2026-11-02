changefinder
============

Online Change-Point Detection Library based on ChangeFinder Algorithm.

Requirements
------------

* Python 3.8 or later
* numpy >= 1.20
* scipy >= 1.7
* statsmodels >= 0.13

Install
-------

::

    pip install .

Usage
-----

::

    import changefinder

    cf = changefinder.ChangeFinder(r=0.1, order=1, smooth=7)
    for x in data:
        score, predict = cf.update(x)

All detectors share the ``update(x) -> (score, predict)`` interface and the
two-stage structure of ChangeFinder (score, smooth, score, smooth).
``update(x, inplace=False)`` scores an observation without learning it.

* ``ChangeFinder``: SDAR in both stages (the original algorithm).
* ``ChangeFinderSI``: SDAR on a differenced (``d``) and seasonally
  differenced (``D``, ``period``) input: the I and seasonal I of SARIMA.
* ``ChangeFinderSTL``: SDAR on the one-step-ahead residual of an OnlineSTL
  decomposition (``periods``, ``gamma``); handles long and multiple periods.
* ``ChangeFinderSTS``: structural time series model (level, optional linear
  ``trend``, Fourier seasonality with ``period`` / ``harmonics``) estimated
  with a discount Kalman filter. Recommended for real data with trend and
  seasonality.
* ``ChangeFinderARIMA``: refits a (seasonal) ARIMA model (``order``,
  ``seasonal_order``, ``trend``) on the last ``term`` points at every step.
  Slow, but the most flexible model.

Options shared by the SDAR / STS detectors:

* ``clip`` (default 3.0): residuals beyond ``clip`` predictive standard
  deviations are winsorized in the learning step, so one outlier does not
  inflate the variance estimate and mask the changes that follow.
* ``smoothing``: ``"mean"`` (original) or ``"median"``. The median passes
  sustained changes but not isolated spikes; it is the default for
  ``ChangeFinderSTS``.

Example on a daily metric with a weekly pattern and growth::

    cf = changefinder.ChangeFinderSTS(r=0.03, trend="linear", period=7)
    for x in data:
        score, predict = cf.update(x)

Notes:

* Use ``order=1`` (at most 2) for SDAR on real data; with forgetting factor
  ``r`` only about ``1 / r`` observations are effective, and higher orders
  overfit.
* Scores start after a warm-up (``1 / r`` observations for SDAR, at least
  two periods for ``ChangeFinderSTS``, four periods for ``ChangeFinderSTL``).

Test
----

::

    pip install .[test]
    pytest

Authors & Contributors
----------------------

* Shunsuke Aihara - http://argmax.jp

License
=======

This code is licensed under MIT License.

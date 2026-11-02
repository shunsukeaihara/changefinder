changefinder
===========

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

``ChangeFinderARIMA`` offers the same ``update`` interface using an ARIMA
model instead of SDAR for the first and second stage.

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

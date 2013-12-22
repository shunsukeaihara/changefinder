# -*- coding: utf-8 -*-
import changefinder
import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample

class TestChangeFinder():
    def setup(self):
        self._term=30
        self._smooth = 7

    def test_changefinder(self):
        cf = changefinder.ChangeFinder(self._term,self._smooth)

        data=np.concatenate([np.random.normal(0.7, 0.1, 100),
                             np.random.normal(1.5, 0.1, 100)])
        for i in data:
            cf.update(i)

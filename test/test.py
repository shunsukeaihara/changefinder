# -*- coding: utf-8 -*-
import changefinder
import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample

class TestChangeFinder():
    def setup(self):
        self._term=10
        self._smooth = 5

    def test_changefinder(self):
        cf = changefinder.ChangeFinder(self._term,self._smooth)
        arparams = np.r_[1,-np.array([.75, .25])]
        maparams = np.r_[1,np.array([.65, .35])]
        sample1 = arma_generate_sample(arparams, maparams, 50)
        arparams = np.r_[1,-np.array([.85, .15])]
        maparams = np.r_[1,np.array([.95, .5])]
        sample2 = arma_generate_sample(arparams, maparams, 50)
        data = np.concatenate([sample1,sample2])
        for i in data:
            cf.update(i)

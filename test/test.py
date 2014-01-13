# -*- coding: utf-8 -*-
import changefinder
import numpy as np

class TestChangeFinder():
    def setup(self):
        self._term=30
        self._smooth = 7
        self._order = 1
        self._arima_order = (1,0,0)
        self._data=np.concatenate([np.random.rand(300)+5,
                                   np.random.rand(300)+10,
                                   np.random.rand(300)+5,
                                   np.random.rand(300)])
    def test_changefinder(self):
        cf = changefinder.ChangeFinder(self._order,self._smooth)

        for i in self._data:
            cf.update(i)


    def test_changefinderarima(self):
        cf = changefinder.ChangeFinderARIMA(self._term,self._smooth,self._arima_order)
        for i in self._data:
            cf.update(i)

# -*- coding: utf-8 -*-
import statsmodels.api as sm
import numpy as np
import scipy as sp

"""
def change_finder(ts, term=30, smooth=7, order=(1,0,0)):
    length = ts.size()
    first_step  = []
    for i in range(term+1,length):
        train = ts[i-term-1:i-1]
        target = ts[i]
        first_step.append(calc_outlier_score(train,target,order))
    first_step = np.array(first_step)
    smoothed = smoothing(first_step,smooth)
    length2 = len(smoothed)
    second_step = []
    for i in range(term+1,length2):
        train = ts[i-term-1:i-1]
        target = ts[i]
        second_step.append(calc_outlier_score(train,target,order))
    second_step = np.array(second_step)
    smoothed = smoothing(second_step,int(round(smooth/2.0)))
    return np.concatenate(np.zeros(length-len(smoothed)),smoothed)


def smoothing(ts,smooth):
    convolve = np.ones(smooth)/smooth
    return np.convolve(ts,convolve,'valid')

def calc_outlier_score(ts,target,order):
    arima_model = sm.tsa.ARIMA(ts,order)
    result = arima_model.fit()
    pred = result.forecast(1)[0][0]
    return outlier_score(result.resid(),x=pred-target)


def outlier_score(residuals,x):
    m = residuals.mean()
    s = np.std(residuals,ddof=1)
    return -sp.stats.norm.logpdf(x,m,s)

"""

class ChangeFinder(object):
    def __init__(self,term = 30, smooth = 7, order = (1,0,0)):
        assert smooth > 2, "term must be 3 or more."
        assert term > smooth, "term must be more than smooth"

        self._term = term
        self._smooth = smooth
        self._order = order
        self._ts = []
        self._first_scores = []
        self._smoothed_scores = []
        self._second_scores = []
        self._convolve = np.ones(self._smooth)
        self._smooth2 = int(round(self._smooth/2.0))
        self._convolve2 = np.ones(int(round(self._smooth2)))

    def _add_one(self,one,ts,size):
        ts.append(one)
        if len(ts)==size+1:
            ts.pop(0)

    def _calc_outlier_score(self,ts,target):
        def outlier_score(residuals,x):
            m = residuals.mean()
            s = np.std(residuals,ddof=1)
            return -sp.stats.norm.logpdf(x,m,s)
        ts = np.array(ts)
        arima_model = sm.tsa.ARIMA(ts,self._order)
        result = arima_model.fit(disp=0)
        pred = result.forecast(1)[0][0]
        return outlier_score(result.resid,x=pred-target)

    def _smoothing(self,ts):
        ts = np.array(ts)
        return np.convolve(ts,self._convolve,'valid')[0]

    def _smoothing2(self,ts):
        ts = np.array(ts)
        return np.convolve(ts,self._convolve2,'valid')[0]

    def update(self,one):
        if len(self._ts) == self._term:#第一段学習
            self._add_one(self._calc_outlier_score(self._ts,one),self._first_scores,self._smooth)
        self._add_one(one,self._ts,self._term)
        second_target = None
        if len(self._first_scores) == self._smooth:#平滑化
            second_target = self._smoothing(self._first_scores)

        if second_target and len(self._smoothed_scores) == self._term:#第二段学習
            self._add_one(self._calc_outlier_score(self._smoothed_scores,second_target),
                          self._second_scores,self._smooth2)
        if second_target:
            self._add_one(second_target,self._smoothed_scores, self._term)
        if len(self._second_scores) == self._smooth2:
            return self._smoothing2(self._second_scores)
        else:
            return 0.0

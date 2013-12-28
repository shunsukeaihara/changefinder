# -*- coding: utf-8 -*-
import statsmodels.api as sm
import numpy as np
import scipy as sp
import math
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
"""
import numpy as np
from changefinder import SDAR_1Dim
s=SDAR_1Dim(0.5)
x=np.random.rand(30)
s.update(0.22,x)
"""


def LevinsonDurbin(r, lpcOrder):
    """
    from http://aidiary.hatenablog.com/entry/20120415/1334458954
    """
    a = np.zeros(lpcOrder + 1,dtype=np.float64)
    e = np.zeros(lpcOrder + 1,dtype=np.float64)

    a[0] = 1.0
    a[1] = - r[1] / r[0]
    e[1] = r[0] + r[1] * a[1]
    lam = - r[1] / r[0]

    for k in range(1, lpcOrder):
        lam = 0.0
        for j in range(k + 1):
            lam -= a[j] * r[k + 1 - j]
        lam /= e[k]

        U = [1]
        U.extend([a[i] for i in range(1, k + 1)])
        U.append(0)

        V = [0]
        V.extend([a[i] for i in range(k, 0, -1)])
        V.append(1)

        a = np.array(U) + lam * np.array(V)
        e[k + 1] = e[k] * (1.0 - lam * lam)

    return a, e[-1]


class _SDAR_1Dim(object):
    def __init__(self, r, term,order = 5):
        self._r = r
        self._mu = np.random.random()
        self._sigma = np.random.random()
        self._term = term
        self._order = order
        self._c = np.zeros(self._term+1)

    def update(self,x,term):
        def outlier_score(residuals,x):
            m = residuals.mean()
            s = np.std(residuals,ddof=1)
            return -sp.stats.norm.logpdf(x,m,s)

        term = np.array(term)
        self._mu = (1 - self._r) * self._mu + self._r * x
        for i in range(1,self._term):
            self._c[i] = (1-self._r)*self._c[i]+self._r * (x-self._mu) * (term[-i]-self._mu)
        self._c[0] = (1-self._r)*self._c[0]+self._r * (x-self._mu)*(x-self._mu)

        what,e = LevinsonDurbin(self._c,self._order)
        xhat = np.dot(-what[1:],(term[::-1][:self._order]  - self._mu))+self._mu
        self._sigma = (1-self._r)*self._sigma + self._r * (x-xhat) * (x-xhat)
        return -math.log(math.exp(-0.5 *(x-xhat)**2/self._sigma)/((2 * math.pi)**0.5 * self._sigma**0.5))

class ChangeFinderSDAR(object):
    def __init__(self, r = 0.5, term=10, smooth=7):
        self._smooth = smooth
        self._term = term
        self._r = r
        self._ts = []
        self._first_scores = []
        self._smoothed_scores = []
        self._second_scores = []
        self._convolve = np.ones(self._smooth)
        self._smooth2 = int(round(self._smooth/2.0))
        self._convolve2 = np.ones(int(round(self._smooth2)))
        self._sdar_first = _SDAR_1Dim(r,term)
        self._sdar_second = _SDAR_1Dim(r,term)

    def _add_one(self,one,ts,size):
        ts.append(one)
        if len(ts)==size+1:
            ts.pop(0)

    def _smoothing(self,ts):
        ts = np.array(ts)
        return np.convolve(ts,self._convolve,'valid')[0]

    def _smoothing2(self,ts):
        ts = np.array(ts)
        return np.convolve(ts,self._convolve2,'valid')[0]


    def update(self,x):
        score = 0
        if len(self._ts) == self._term:#第一段学習
            score = self._sdar_first.update(x,self._ts)
            self._add_one(score,self._first_scores,self._smooth)
        self._add_one(x,self._ts,self._term)
        second_target = None
        if len(self._first_scores) == self._smooth:#平滑化
            second_target = self._smoothing(self._first_scores)
        if second_target and len(self._smoothed_scores) == self._term:#第二段学習
            score = self._sdar_second.update(second_target,self._smoothed_scores)
            self._add_one(score,
                          self._second_scores,self._smooth2)
        if second_target:
            self._add_one(second_target,self._smoothed_scores, self._term)
        if len(self._second_scores) == self._smooth2:
            return self._smoothing2(self._second_scores)
        else:
            return 0.0


class _ChangeFinder(object):
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

    def update(self,sample):
        if len(self._ts) == self._term:#第一段学習
            try:
                self._add_one(self._calc_outlier_score(self._ts,sample),self._first_scores,self._smooth)
            except:
                self._add_one(sample,self._ts,self._term)
                return 0
        self._add_one(sample,self._ts,self._term)

        second_target = None
        if len(self._first_scores) == self._smooth:#平滑化
            second_target = self._smoothing(self._first_scores)

        if second_target and len(self._smoothed_scores) == self._term:#第二段学習
            try:
                self._add_one(self._calc_outlier_score(self._smoothed_scores,second_target),
                              self._second_scores,self._smooth2)
            except:
                self._add_one(second_target,self._smoothed_scores, self._term)
                return 0
        if second_target:
            self._add_one(second_target,self._smoothed_scores, self._term)
        if len(self._second_scores) == self._smooth2:
            return self._smoothing2(self._second_scores)
        else:
            return 0.0

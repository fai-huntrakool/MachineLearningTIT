# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 19:13:34 2019

@author: FaiHuntrakool
"""

import numpy as np
import matplotlib.pyplot as plt

n=200
x=3*(np.random.rand(n,4)-0.5)
y=(2*x[:,0]-1*x[:,1]+0.5+0.5*np.random.randn(1,n))>0
y=(2*y-1)[0]


def cal_guassian(alpha,x):
    ans=np.zeros_like(x)
    

#batch gradient descent
lam=0.001
num_iter=500
ww=np.ones(4)*10
alpha_base=1
ll_history=[]
ww_history=[]
lip=0.25*max(x[:,0]**2)

for t in range(1,num_iter+1):
    hypothesis=np.dot(x,ww)
    deno=1/(1 + np.exp(-y * hypothesis))
    grad=np.dot(1/n*(1-deno)*(-y),x)+2*lam*ww
    ll = np.sum(np.log(1 + np.exp(-y * hypothesis))) + np.sum(lam *(ww**2))
    ww_history.append(ww);
    ll_history.append(ll);
    ww = ww - alpha_base * 1.0 / np.sqrt(t) / lip * grad;

plt.plot(ll_history)
plt.show()

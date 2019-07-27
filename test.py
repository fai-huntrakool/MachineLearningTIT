# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:02:35 2019

@author: FaiHuntrakool
"""

import numpy as np
import matplotlib.pyplot as plt


def dataset2(num = 40):
    num=100
    omega = np.random.randn(1)
    noise = 0.8 * np.random.randn(1, num)
    x = np.random.randn(2, num)
    y = (2 * (x[0, :] + x[1, :] + noise > 0) - 1)
    return(x, y)
    
def soft_threshold(q, mu):
    if mu > q:
        return mu - q
    elif mu < -q:
        return mu + q
    else:
        return 0

# soft thresholding func for vector
def soft_threshold_array(q, arr):
    return np.array([soft_threshold(q, mu) for mu in arr])    

def subgradient_hinge(x, y, w, theta=0.5):
    if y * np.dot(w, x) > 1:
        return np.zeros(x.shape[0])
    elif y * np.dot(w, x) < 1:
        return - y * x
    else:
        return - theta * y * x

# proximal subgradient method
def proximal_subgradient(w_init, x, y, _lambda, _eta, repeat_num=50):
    # history of params
    w_hist = [w_init]
    # update params
    for t in range(0, repeat_num):
        # get subgradient
        g = np.sum(np.array([subgradient_hinge(_x, _y, w_hist[-1])
                             for (_x, _y) in zip(x.T, y.T)]), axis=0)
        w = soft_threshold_array(_eta * _lambda, w_hist[-1] - _eta * g)
        w_hist.append(w)
    return w_hist

_eta = 0.05
_lambda = 1
repeat_num = 50
data_num = 100
w_init = np.zeros(4)

# create training data

n=100
x=3*(np.random.rand(4,n)-0.5)
y=(2*x[0,:]-1*x[1,:]+0.5+0.5*np.random.randn(1,n))>0
y=(2*y-1)

# implement proximal subgradient method
w = proximal_subgradient(w_init, x, y, _lambda, _eta, repeat_num)

print(w[-1])

def st_ops(mu, q):
  x_proj = np.zeros(mu.shape)
  for i in range(len(mu)):
    if mu[i] > q:
      x_proj[i] = mu[i] - q
    else:
      if np.abs(mu[i]) < q:
        x_proj[i] = 0
      else:
        x_proj[i] = mu[i] + q; 
  return x_proj

soft_threshold_array(_eta * _lambda, w_hist[-1] - _eta * g)
st_ops(w_hist[-1]-_eta*g,_eta*_lambda)
def cal_grad(y,x,lam,w):
    w=w_hist[-1]
    loss,grad = 0,0
    for (x_,y_) in zip(x.T,y.T):
        v = y_*np.dot(w,x_)
        loss += max(0,1-v)
        grad += 0 if v >= 1 else (-y_*x_)
    return grad

grad=cal_grad(y,x,lam,w)

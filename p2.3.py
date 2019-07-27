# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:39:02 2019

@author: FaiHuntrakool
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cv
import cvxopt
import itertools
from cvxpy import CVXOPT

d=200
n=180

g={}
for i in range(1,6):
    g[i]=np.arange((i-1)*40+1,i*40+1)
    
x=np.random.randn(n,d)
noise=0.5
w=np.array(list(20*np.random.rand(80))+[0]*120+[5*np.random.rand()])
w=np.resize(w,(len(w),1))

x_tilde=np.hstack((x,np.ones((n,1))))
y=np.dot(x_tilde,w)+noise*np.random.randn(n,1)

lamb=1.0
wridge=np.linalg.lstsq((np.dot(x_tilde.T,x_tilde) + lamb*np.identity(d+1)),np.dot(x_tilde.T,y))[0]


#CVXOPT
west = cv.Variable((d+1,1))
#objective=cv.Minimize(0.5/n*np.dot((x_tilde*west-y).T,x_tilde*west-y)+
#                      lamb*( cv.norm(west[g[1]],2)+cv.norm(west[g[2]],2)+cv.norm(west[g[3]],2)+cv.norm(west[g[4]],2)+cv.norm(west[g[5]],2)))

objective=cv.Minimize(0.5/n*cv.sum_squares(x_tilde*west-y)+
                      +lamb*(cv.norm(cv.vstack([west[i] for i in g[1]])))
                      +(cv.norm(cv.vstack([west[i] for i in g[2]])))
                      +(cv.norm(cv.vstack([west[i] for i in g[3]])))
                      +(cv.norm(cv.vstack([west[i] for i in g[4]])))
                      +(cv.norm(cv.vstack([west[i] for i in g[5]]))))
prob=cv.Problem(objective)
#
result=prob.solve()
west_opt=west.value


#Proximal method
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

w_init = np.zeros((d+1,1))
L = 2

w_history = []
wt = w_init
for t in range(200):
  w_history.append(wt.T)
  grad = 1/n*np.dot(x_tilde.T,(np.dot(x_tilde,wt)-y))
  wth = wt - 1/L * grad
  wt = st_ops(wth, lamb * 1 / L)
  
y_opt=np.dot(x_tilde,west_opt)
y_prox=np.dot(x_tilde,wt)

print(np.average((y_opt-y)**2))
print(np.average((y_prox-y)**2))








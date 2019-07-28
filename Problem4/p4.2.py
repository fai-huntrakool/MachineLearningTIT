# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:36:30 2019

@author: FaiHuntrakool

"""

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cv
import cvxopt
import itertools
from cvxpy import CVXOPT

n=200
x=3*(np.random.rand(n,4)-0.5)
y=(2*x[:,0]-1*x[:,1]+0.5+0.5*np.random.randn(1,n))>0
y=(2*y-1)[0]
y=np.resize(y,(200,1))

#CVXOPT1
lam = 1;
w1=cv.Variable((x.shape[1],1))
loss=cv.sum(cv.pos(1-cv.multiply(y,x*w1)))
reg=cv.norm(w1,1)
objective=cv.Minimize(loss+lam*reg)
prob = cv.Problem(objective)
result = prob.solve(solver=CVXOPT) 
w1 = w1.value

#CVXOPT2
e=cv.Variable((x.shape[0],1))
w2=cv.Variable((x.shape[1],1))
objective=cv.Minimize(cv.norm(e,1)+lam*cv.norm(w2,1))
constraints = [e[i]>=1-y[i]*w2.T@x[i] for i in range(len(x))]
constraints+=[e[i]>=0 for i in range(len(x))]
# cvx
prob = cv.Problem(objective,constraints)
result = prob.solve(solver=CVXOPT) 
w2 = w2.value
e = e.value

print('W1')
print(w1)
print('W2')
print(w2)

loss_cvx=np.sum(e)+lam*np.sum(w2)

###proximal gradient
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

w_init = np.zeros_like(w1)
eta=0.01

def cal_grad(y,x,lam,w):
    loss,grad = 0,0
    for (x_,y_) in zip(x,y):
        v = y_*np.dot(w.T,x_)
        loss += max(0,1-v)
        grad += 0 if v >= 1 else (-y_*x_)
    return loss,np.resize(grad,(x.shape[1],1))
        
loss_history=[]
w_history = []
w3 = w_init
for t in range(30):
  w_history.append(w3.T)
  loss,grad = cal_grad(y,x,lam,w3)
  loss_history.append(loss)
  wth = w3 - eta*grad
  w3 = st_ops(wth, lam*eta )
  
print('W3')
print(w3)



plt.plot(30*[loss_cvx],label='Optimal')
plt.plot(loss_history,label='Proximal Gradient')
plt.legend()
plt.ylabel('Function score')
plt.xlabel('iteration')
plt.show()

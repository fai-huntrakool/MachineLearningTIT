# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 13:43:18 2019

@author: FaiHuntrakool
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:49:32 2019

@author: FaiHuntrakool
"""


import numpy as np
import matplotlib.pyplot as plt

n=200
x=3*(np.random.rand(n,4)-0.5)
y=(2*x[:,0]-1*x[:,1]+0.5+0.5*np.random.randn(1,n))>0
y=(2*y-1)[0]

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

ans_ww=ww_history[ll_history.index(min(ll_history))]


min_loss=min(ll_history)
ll_history=np.array(ll_history)-min_loss
x_axis=np.arange(num_iter)
plt.semilogy(x_axis,ll_history,label='Gradient Method')


#newton based method:
lam=0.001
num_iter=500   
ll_n_history = []
ww_n_history = []
ww = np.ones(x.shape[1])*10

for t in range(num_iter):
    hypothesis=np.dot(x,ww)
    deno=1/(1 + np.exp(-y * hypothesis))
    grad=np.dot(1/n*(1-deno)*(-y),x)+2*lam*ww
    #hess=1/n*np.sum(posterior * (1 - posterior) * x[:,0]**2) + 2 * lam
    hess=np.dot(1/n*(1-deno)*deno,x**2)+2*lam
    ll = np.sum(np.log(1 + np.exp(-y * hypothesis))) + np.sum(lam *(ww**2))
    ww_n_history.append(ww)
    ll_n_history.append(ll)
    ww = ww - grad/hess*1.0/np.sqrt(t+10)

ans_ww=ww_n_history[ll_n_history.index(min(ll_n_history))]
    

min_loss=min(ll_n_history)
ll_history=np.array(ll_n_history)-min_loss
x_axis=np.arange(num_iter)
plt.semilogy(x_axis,ll_history,label='Newton Method')
plt.xlabel('iteration')
plt.ylabel('log J(w^t)-J(w^)')
plt.legend()
plt.show()



    

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

#batch gradient descent
def gradient_descent(x,y):
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
    
    #real Y
    plt.plot(np.extract(y>0,x[:,0]),np.extract(y>0,x[:,1]), 'x')
    plt.plot(np.extract(y<0,x[:,0]),np.extract(y<0,x[:,1]), 'o')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    #predict_Y
    predicted_y=2*np.dot(x,ans_ww)-1
    plt.plot(np.extract(predicted_y>0,x[:,0]),np.extract(predicted_y>0,x[:,1]), 'x')
    plt.plot(np.extract(predicted_y<0,x[:,0]),np.extract(predicted_y<0,x[:,1]), 'o')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

    
    min_loss=min(ll_history)
    ll_history=np.array(ll_history)-min_loss
    x_axis=np.arange(num_iter)
    plt.semilogy(x_axis,ll_history)
    plt.xlabel('iteration')
    plt.ylabel('epoch')
    plt.show()
    
#newton based method:
def newton_based_method(x,y):
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
        
    #real Y
    plt.plot(np.extract(y>0,x[:,0]),np.extract(y>0,x[:,1]), 'x')
    plt.plot(np.extract(y<0,x[:,0]),np.extract(y<0,x[:,1]), 'o')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    #predict_Y
    predicted_y=2*np.dot(x,ans_ww)-1
    plt.plot(np.extract(predicted_y>0,x[:,0]),np.extract(predicted_y>0,x[:,1]), 'x')
    plt.plot(np.extract(predicted_y<0,x[:,0]),np.extract(predicted_y<0,x[:,1]), 'o')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

    min_loss=min(ll_n_history)
    ll_history=np.array(ll_n_history)-min_loss
    x_axis=np.arange(num_iter)
    plt.semilogy(x_axis,ll_history)
    plt.xlabel('iteration')
    plt.ylabel('epoch')
    plt.show()
    


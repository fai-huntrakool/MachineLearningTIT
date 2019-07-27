# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:49:00 2019

@author: FaiHuntrakool
"""


import numpy as np
import matplotlib.pyplot as plt
import itertools


n=200
x=3*(np.random.rand(4,n)-0.5)
y=(2*x[0,:]-1*x[1,:]+0.5+0.5*np.random.randn(1,n))>0
y=(2*y-1)
lamb=10
eta = 0.1

def cal_proj(v,s,l):
    proj=np.zeros_like(l)
    i=0
    for _v,_s,_l in zip(v,s,l):
       if _v>_l:
           proj[i]=_l
       elif _v<_s:
           proj[i]=_s
       else:
           proj[i]=_v
       i+=1
    return proj

def cal_K(x,y):
    K=np.zeros((x.shape[1],x.shape[1]))
    for i in range(K.shape[0]):
        for j in range(K.shape[0]):
            K[i,j]=y[0,i]*y[0,j]*np.dot(x[:,i].T,x[:,j])
    return K

def hinge_loss(x,y,w,lamb):
    loss=np.sum([max(0,1-y[0,i]*np.dot(w,x[:,i])) for i in range(x.shape[1])])+lamb*np.sum(w**2)
    return loss

def dual_lagrange(K,alpha,lamb):
    return -np.dot(alpha.T,np.dot(K,alpha))/(4*lamb)+np.sum(alpha)

alpha_init=np.zeros(n)
alpha=alpha_init
alpha_hist=[]
K=cal_K(x,y)
epoch=50
for t in range(epoch):
    alpha_hist.append(alpha)
    u_alpha=alpha-eta*(np.dot(K,alpha)/(2*lamb)-np.ones(n))
    alpha=cal_proj(u_alpha,np.zeros(n),np.ones(n))    
    

w_hist=np.zeros((epoch,x.shape[0]))
for i in range(epoch):
    w_hist[i,:]=1/(2*lamb)*np.sum((alpha_hist[i]*y)*x,axis=1)
    

loss=[]
loss_dual=[]
    
for i in range(epoch):
    loss.append(hinge_loss(x,y,w_hist[i],lamb))
    loss_dual.append(dual_lagrange(K,alpha_hist[i],lamb))
    
    
plt.plot(loss,label='projected gradient')
plt.plot(loss_dual,label='dual lagrange')
plt.legend()
plt.show()
    
    
    
    
    
    
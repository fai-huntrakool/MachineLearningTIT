# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:31:44 2019

@author: FaiHuntrakool
"""

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

def dual_lagrange(K,alpha,lamb):
    return -np.dot(alpha.T,np.dot(K,alpha))/(4*lamb)+np.sum(alpha)

def cal_w(alpha,x,y):
    return np.sum([alpha[i]*y[0,i]*x[:,i] for i in range(y.shape[1])],axis=0)

def update_alpha(G,Qii,alpha,C,ind):
    new=np.zeros_like(alpha)
    for i in range(len(alpha)):
        new[i]=alpha[i]
    new[ind]=min(max(new[ind]-G/Qii,0),C)
    return new
#
#Dual coordinate
alpha_init=np.zeros(n)
alpha=alpha_init
alpha_hist1=[]
K=cal_K(x,y)
C=0.5/lamb
epoch=100
w=cal_w(alpha,x,y)
count=0
while count<epoch:
    alpha_hist1.append(alpha)
    for i in range(alpha.shape[0]):
        Qii=np.dot(x[:,i].T,x[:,i])
        G=y[0,i]*np.dot(w.T,x[:,i])-1
        if alpha[i]==0:
            PG=min(G,0)
        elif alpha[i]==C:
            PG=max(G,0)
        elif (alpha[i]<C) & (alpha[i]>0):
            PG=G
        if PG!=0:
            temp=alpha[i]
            alpha=update_alpha(G,Qii,alpha,C,i)
            w=w+(alpha[i]-temp)*y[0,i]*x[:,i]
            
    count+=1
    
#Projected Gradient
alpha_init=np.zeros(n)
alpha=alpha_init
alpha_hist2=[]
K=cal_K(x,y)
for t in range(epoch):
    alpha_hist2.append(alpha)
    u_alpha=alpha-eta*(np.dot(K,alpha)/(2*lamb)-np.ones(n))
    alpha=cal_proj(u_alpha,np.zeros(n),np.ones(n))    
     

#SMO
alpha_init=np.zeros(n)
alpha=alpha_init
alpha_hist3=[]
K=cal_K(x,y)
C=0.5/lamb
count=0
while count<epoch:
    alpha_co=alpha.copy()
    alpha_hist3.append(alpha_co)
    for i in range(0,x.shape[1]):
        j=i
        while j==i:
            j=np.random.randint(0,x.shape[1]-1)
                
        y1=y[0,i]
        y2=y[0,j]
        alpha1,alpha2=alpha[i].copy(),alpha[j].copy()
        
        if y1==y2:
            L=max(0,alpha2+alpha1-C)
            H=min(C,alpha1+alpha2)
        else:
            L=max(0,alpha2-alpha1)
            H=min(C,C+alpha2-alpha1)
        if L==H:
            continue
        
        E1=np.dot(cal_w(alpha,x,y).T,x[:,i])-y[0,i]
        E2=np.dot(cal_w(alpha,x,y).T,x[:,j])-y[0,j]
        
        k11=np.dot(x[:,i].T,x[:,i])
        k22=np.dot(x[:,j].T,x[:,j])
        k12=np.dot(x[:,i].T,x[:,j])
        
        eta=2*k12-k11-k22
        if eta>=0:
            continue
        
        alpha[j]-=y2*(E1-E2)/eta
        if alpha[j]>H:
            alpha[j]=H
        elif alpha[j]<L:
            alpha[j]=L
        
        alpha[i]+=y1*y2*(alpha2-alpha[j])
    count+=1
    
#Plot
loss_1=[]
loss_2=[]
loss_3=[]
    
for i in range(epoch):
    loss_1.append(dual_lagrange(K,alpha_hist1[i]/C,lamb))
    loss_2.append(dual_lagrange(K,alpha_hist2[i],lamb))
    loss_3.append(dual_lagrange(K,alpha_hist3[i]/C,lamb))
    
plt.plot(loss_1,label='Dual Coordinate')
plt.plot(loss_2,label='Projected Gradient')
plt.plot(loss_3,label='SMO')
plt.ylabel('function score')
plt.xlabel('number of iteration')
plt.legend()
plt.show()

    
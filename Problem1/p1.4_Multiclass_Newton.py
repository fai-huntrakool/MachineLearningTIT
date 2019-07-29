# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 17:50:47 2019

@author: FaiHuntrakool
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 17:07:09 2019

@author: FaiHuntrakool
"""
import numpy as np

def cal_softmax(w,x):
    exp=np.exp(np.dot(w.T,x))
    softmax=exp/np.sum(exp)
    return softmax

n=200
x=3*(np.random.rand(n,4)-0.5)
W=np.array([[2,-1,0.5],[-3,2,1],[1,2,3]])

t=np.dot(np.hstack((x[:,0:2],np.ones((n,1)))),W.transpose())+0.5*np.random.randn(n,3)
maxlogit=np.max(t,axis=1)
y=np.argmax(t,axis=1)


classes=[0,1,2]
x=np.hstack([x,np.ones((n,1))])
w=np.zeros((x.shape[1],len(classes)))

w_history=[]
loss_history=[]
lam=1
t=0
epoch=100
while(t<epoch):
    grad=np.zeros_like(w)
    hess=[np.zeros((5,5)),np.zeros((5,5)),np.zeros((5,5))]
    w_history.append(w)
    jw=0    
    J_w=0
    for i in range(n):
        softmax=cal_softmax(w,x[i].reshape(5,1))
        for j in classes:
            if j==y[i]:
                grad[:,y[i]]+=(softmax[y[i]]-1)*x[i]
            else:
                grad[:,j]+=(softmax[j])*x[i]
            hess[j]+=(1-softmax[j])*softmax[j]*np.dot(x[i],x[i].T)
        jw-=np.log(softmax[y[i]])
    for j in classes:
        hess[j]=hess[j]/n+2*lam*np.eye(5)
        w[:,j]=w[:,j]-np.dot(np.linalg.inv(hess[j]),grad[:,j])*1.0/np.sqrt(t+10)
    grad=grad/n+2*lam*w
    
    jw/=n
    J_w+=jw+lam*np.sum(np.dot(w.T,w))
    loss_history.append(J_w)
    t+=1





test=np.dot(x,w)
ans_y=np.argmax(test,1)
diff=[]

for i in range(len(loss_history)):
    diff.append(loss_history[i]-min(loss_history))


import matplotlib.pyplot as plt

plt.semilogy(diff)
plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2)
#real Y
ax1.plot(np.extract(y==0,x[:,0]),np.extract(y==0,x[:,1]), 'x')
ax1.plot(np.extract(y==1,x[:,0]),np.extract(y==1,x[:,1]), 'o')
ax1.plot(np.extract(y==2,x[:,0]),np.extract(y==2,x[:,1]), 'D')
ax1.set_title('Actual Data')
#predict_Y
ax2.plot(np.extract(ans_y==0,x[:,0]),np.extract(ans_y==0,x[:,1]), 'x')
ax2.plot(np.extract(ans_y==1,x[:,0]),np.extract(ans_y==1,x[:,1]), 'o')
ax2.plot(np.extract(ans_y==2,x[:,0]),np.extract(ans_y==2,x[:,1]), 'D')
ax2.set_title('Newton')
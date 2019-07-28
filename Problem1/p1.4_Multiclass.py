import numpy as np
import matplotlib.pyplot as plt

n=200
x=3*(np.random.rand(n,4)-0.5)
W=np.array([[2,-1,0.5],[-3,2,1],[1,2,3]])

t=np.dot(np.hstack((x[:,0:2],np.ones((n,1)))),W.transpose())+0.5*np.random.randn(n,3)
maxlogit=np.max(t,axis=1)
y=np.argmax(t,axis=1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


print('Gradient Descent')

lam=0.1
num_iter=500
WW_1=np.zeros((x.shape[1],3))
ww=np.ones(4)*10
alpha_base=1
ll_history=[]
ww_history=[]
lip=0.25*max(x[:,0]**2)
ans_y=np.zeros((y.shape[0],3))
for c in [0,1,2]:    
    ww=np.ones(4)*10
#    ll_history=[]
#    ww_history=[]    
    for t in range(1,num_iter+1):
        y_i=np.array([1 if (i==c) else 0 for i in y ])
        hypothesis=np.dot(x,ww)
        deno=1/(1 + np.exp(-y_i * hypothesis))
        grad=np.dot(1/n*(1-deno)*(-y_i),x)+2*lam*ww
        ll = np.sum(np.log(1 + np.exp(-y_i*hypothesis))) + np.sum(lam*(ww**2))
        ww_history.append(ww)
        ll_history.append(ll)
        ww = ww - alpha_base * 1.0 / np.sqrt(t) / lip * grad;
    #ans_ww=ww_history[ll_history.index(min(ll_history))]
    ans_ww=ww
    WW_1[:,c]=ans_ww
    
predicted_y=sigmoid(np.dot(x,WW_1))
ans_y=np.argmax(predicted_y,axis=1)  

fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
#real Y
ax1.plot(np.extract(y==0,x[:,0]),np.extract(y==0,x[:,1]), 'x')
ax1.plot(np.extract(y==1,x[:,0]),np.extract(y==1,x[:,1]), 'o')
ax1.plot(np.extract(y==2,x[:,0]),np.extract(y==2,x[:,1]), 'D')
ax1.set_title('Actual Data')
#predict_Y
ax2.plot(np.extract(ans_y==0,x[:,0]),np.extract(ans_y==0,x[:,1]), 'x')
ax2.plot(np.extract(ans_y==1,x[:,0]),np.extract(ans_y==1,x[:,1]), 'o')
ax2.plot(np.extract(ans_y==2,x[:,0]),np.extract(ans_y==2,x[:,1]), 'D')
ax2.set_title('Gradient')

def accuracy(ans_y,y):
    return np.sum(ans_y==y)/len(ans_y)

print('accuracy : %.2f'%(accuracy(ans_y,y)))







lam=0.001
num_iter=500   
ll_n_history = []
ww_n_history = []
ww = np.ones(x.shape[1])*10
WW_2=np.zeros((x.shape[1],3))
for c in [0,1,2]:
    ww=np.ones(4)*10
    for t in range(num_iter):
        y_i=np.array([1 if (i==c) else 0 for i in y ])
        hypothesis=np.dot(x,ww)
        deno=1/(1 + np.exp(-y_i * hypothesis))
        grad=np.dot(1/n*(1-deno)*(-y_i),x)+2*lam*ww
        #hess=1/n*np.sum(posterior * (1 - posterior) * x[:,0]**2) + 2 * lam
        hess=np.dot(1/n*(1-deno)*deno,x**2)+2*lam
        ll = np.sum(np.log(1 + np.exp(-y_i * hypothesis))) + np.sum(lam *(ww**2))
        ww_n_history.append(ww)
        ll_n_history.append(ll)
        ww = ww - grad/hess*1.0/np.sqrt(t+10)
    ans_ww=ww
    WW_2[:,c]=ans_ww


predicted_y=sigmoid(np.dot(x,WW_1))
ans_y=np.argmax(predicted_y,axis=1)  


print('NEWTON Method')
print('accuracy : %.2f'%(accuracy(ans_y,y)))

#predict_Y
ax3.plot(np.extract(ans_y==0,x[:,0]),np.extract(ans_y==0,x[:,1]), 'x')
ax3.plot(np.extract(ans_y==1,x[:,0]),np.extract(ans_y==1,x[:,1]), 'o')
ax3.plot(np.extract(ans_y==2,x[:,0]),np.extract(ans_y==2,x[:,1]), 'D')
ax3.set_title('Newton')
plt.show()


ll_history=np.array(ll_history)-min(ll_history)
ll_n_history=np.array(ll_n_history)-min(ll_n_history)
x_axis=np.arange(num_iter*3)
plt.semilogy(x_axis,ll_history,label='gradient')
plt.semilogy(x_axis,ll_n_history,label='newton')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('epoch')
plt.show()








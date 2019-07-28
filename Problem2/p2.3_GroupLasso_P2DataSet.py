import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cv
import cvxopt
import itertools
from cvxpy import CVXOPT

# we need to control this parameter to generate multiple figures
lam = 2;
#lam = 4;
#lam = 6;
#lam = 3.89; 

x_1 = np.arange(-1.5, 3, 0.01)
x_2 = np.arange(-1.5, 3, 0.02)

X1, X2 = np.mgrid[-1.5:3:0.01, -1.5:3:0.02]
fValue = np.zeros((len(x_1), len(x_2)))

A = np.array([[  3, 0.5],
              [0.5,   1]])
mu = np.array([[1],
               [2]])

for i in range(len(x_1)):
  for j in range(len(x_2)):
        inr = np.vstack([x_1[i], x_2[j]])
        fValue[i, j] = np.dot(np.dot((inr-mu).T, A), (inr- mu)) + lam * (np.abs(x_1[i]) + np.abs(x_2[j]))

plt.contour(X1, X2, fValue)


# cvx
w_lasso = cv.Variable((2,1))
obj_fn = cv.quad_form(w_lasso - mu, A) +  lam * (cv.norm(w_lasso[0], 2)+cv.norm(w_lasso[1],2))
objective = cv.Minimize(obj_fn)
constraints = []
prob = cv.Problem(objective, constraints)
result = prob.solve(solver=CVXOPT) 
w_lasso = w_lasso.value




#Proximal
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


x_init = np.array([[ 3],
                   [-1]])
L = 1.01 * np.max(np.linalg.eig(2 * A)[0])

x_history = np.zeros((500,2))
xt = x_init
for t in range(500):
  temp=xt
  x_history[t,:]=xt.T
  grad = 2 * np.dot(A, xt-mu)
  xth = xt - 1/L * grad
  xt = st_ops(xth, lam * 1 / L)
  


plt.figure() 
plt.contour(X1, X2, fValue)
plt.plot(x_history[:,0], x_history[:,1], 'ro-', markersize=3, linewidth=0.5, label='Proximal Gradient')
plt.plot(w_lasso[0], w_lasso[1], 'ko', label='cvx')
plt.legend()
plt.show()
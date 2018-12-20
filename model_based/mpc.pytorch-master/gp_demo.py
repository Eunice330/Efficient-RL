import warnings

warnings.filterwarnings('ignore')

import gptools
#import cPickle as pkl
import scipy
import matplotlib.pyplot as plt
import numpy as np
hp = gptools.UniformJointPrior(0, 20) * gptools.GammaJointPriorAlt(1, 0.7)
k_SE = gptools.SquaredExponentialKernel(hyperprior=hp)

k_noise = gptools.DiagonalNoiseKernel(noise_bound=[0, 4, 1 ,2])
gp_noise = gptools.GaussianProcess(k_SE, noise_k=k_noise)
gp = gptools.GaussianProcess(
    gptools.SquaredExponentialKernel(hyperprior=hp*hp, num_dim=3)
)

y=[1, 2, 3, 4, 5]
X=[[0,1, 1], [0, 1, 6], [0, 1, 8], [2, 2, 2], [3, 4, 5]]

gp.add_data(X, y)

gp.optimize_hyperparameters(verbose=True)

X_star = [[0,1, 1], [0, 1, 6], [0, 1, 8], [2, 2, 2], [3, 4, 5], [0, 1, 6], [0, 1, 8], [2, 2, 2], [3, 4, 5]]
d=len(X_star[0])
X_test=[[0, 1, 1] for i in range(d*d)]
# Now, we can make a prediction of the mean and standard deviation of the fit:
#y_star, err_y_star = gp.predict(X_star)
#N = [[0,0, 2],[1, 0,1], [0, 0,1], [2, 0,0],[2, 0, 0]]
flat = np.zeros((d, d))
index=[]
for i in range(d):
    for j in range(d):
       index.append([i, j])

for k in range(d):
    if index[k][0]==index[k][1]: flat[k][index[k][0]]=1 # first order
print('flat',flat)


H_flat = np.zeros((d*d, d))
for k in range(d*d):
    if index[k][0]==index[k][1]: H_flat[k][index[k][0]]=2 # second order
    else:
       H_flat[k][index[k][0]] = 1
       H_flat[k][index[k][1]] = 1

#print(H_flat)
grad_y_star, err_grad_y_star = gp.predict([[0, 1,1],[0, 1, 1],[0, 1,1]], n=flat)
print('first order', grad_y_star.shape, grad_y_star)
#print('first order grad', np.reshape(grad_y_star, [d, d]))
#print('first order err grad', np.reshape(err_grad_y_star, [d, d]))


import warnings

warnings.filterwarnings('ignore')

import gptools
#import cPickle as pkl
import scipy
import matplotlib.pyplot as plt
import numpy as np

class GP_cost(object):
    def __init__(self, train_X=None, train_R=None):
        super().__init__()

        #self.max_torque = 2.0
        #self.dt = 0.05
        #self.n_state = 3
        #self.n_ctrl = 1
        self.X = train_X
        self.R = train_R
        self.num_dim = self.X.shape[1]

        self.hp = gptools.UniformJointPrior(0, 20) * gptools.GammaJointPriorAlt(1, 0.7)
        self.k_SE = gptools.SquaredExponentialKernel(hyperprior=self.hp)
        self.k_noise = gptools.DiagonalNoiseKernel(noise_bound=[0, 4, 1 ,2])
        self.gp_noise = gptools.GaussianProcess(self.k_SE, noise_k=self.k_noise)
        self.gp = gptools.GaussianProcess(
                  gptools.SquaredExponentialKernel(hyperprior=self.hp*self.hp, num_dim=self.num_dim))
        self.gp.add_data(self.X, self.R) #add init data for gp

    def get_cost(self, x, u): #get costs, grads, hessians
        #print('input u of cost', u)
        batch_n = np.array(x).shape[0]
        x = np.array(x)
        u = np.array(u)
        #print('input x of cost', x.shape)
        #print('input u of cost', u.shape)
        batch_Ct, batch_grads, batch_hessians = [], [], []
        
        for b in range(batch_n):
            xu = np.concatenate((x[b],u[b]),0)
            d = xu.shape[0]
            d1 = x.shape[1]
            
            index=[]
            for i in range(d):
                for j in range(d):
                    index.append([i, j])
            
            flat = np.zeros((d, d)) #first order
            for k in range(d):
                if index[k][0]==index[k][1]: flat[k][index[k][0]]=1 # first order index of dynamics

            H_flat = np.zeros((d*d, d)) #second order
            for k in range(d*d):
                if index[k][0]==index[k][1]: H_flat[k][index[k][0]]=2
            else:
                H_flat[k][index[k][0]] = 1
                H_flat[k][index[k][1]] = 1
            #print('H', H_flat)
            
            xu_copy = [xu for i in range(d)]
            xu_copy_copy = [xu for i in range(d*d)]
            ct, ct_std = self.gp.predict(xu, n=0)
            self.gp.add_data(xu, ct)
            #self.gp.optimize_hyperparameters(verbose=True)
            #add sample
            grad_xu_t_of_ct, err_grad_c_t = self.gp.predict(xu_copy, n=flat)
            hess_xu_t_of_ct, err_hess_c_t = self.gp.predict(xu_copy_copy, n=H_flat)
            # grad_xu_t_of_ct = grad_xu_t_of_ct/1000 #scale for debug
            # hess_xu_t_of_ct = hess_xu_t_of_ct/1000 #scale for debug
            np.clip(grad_xu_t_of_ct, -1, 1, grad_xu_t_of_ct)
            np.clip(hess_xu_t_of_ct, -1, 1, hess_xu_t_of_ct)
            hess = np.reshape(hess_xu_t_of_ct, [d, d])
            #print('one hess shape', hess.shape)
            batch_Ct.append(ct)
            batch_grads.append(grad_xu_t_of_ct)
            batch_hessians.append(hess)
            #print('batch Ct', np.array(batch_Ct).shape, batch_Ct)
            
        return batch_Ct, batch_grads, batch_hessians    
     


























        
 
        
        
        

import warnings

warnings.filterwarnings('ignore')

import gptools
#import cPickle as pkl
import scipy
import matplotlib.pyplot as plt
import numpy as np
import time

class gp_dynamics_dx(object):
    def __init__(self, train_X=None, train_Y=None):
        #super().__init__()

        #self.max_torque = 2.0
        #self.dt = 0.05
        #self.n_state = 3
        #self.n_ctrl = 1
        self.X = train_X
        self.Y = train_Y
        #print('train X', train_X)
        #print('train Y', train_Y)
        self.num_dim =  self.X.shape[1]
        self.isPendulum = False
        
        self.hp = gptools.UniformJointPrior(0, 20) * gptools.GammaJointPriorAlt(1, 0.7)
        self.hp1 = gptools.UniformJointPrior(0, 20)
        self.k_SE = gptools.SquaredExponentialKernel(hyperprior=self.hp)
        self.k_noise = gptools.DiagonalNoiseKernel(noise_bound=[0, 4, 1 ,2])
        self.gp_noise = gptools.GaussianProcess(self.k_SE, noise_k=self.k_noise)
        # self.gp = gptools.GaussianProcess(
        #           gptools.SquaredExponentialKernel(hyperprior=self.hp*self.hp*self.hp1, num_dim=self.num_dim)) # 3 is the dimention of x
        self.all_gps = [gptools.GaussianProcess(
                  gptools.SquaredExponentialKernel(hyperprior=self.hp*self.hp*self.hp1, num_dim=self.num_dim)) for i in range(self.Y.shape[1])]
        # for dim in range(self.Y.shape[1]):
        #     self.all_gps[dim].add_data(self.X, self.Y[:,dim])
            #self.all_gps[dim].optimize_hyperparameters(verbose=True)
        print(1)
    def grad_input(self, x, u): # get next states and grads
        #start = time.time()
        x = np.array(x)
        u = np.array(u)
        # print('input x shape', x.shape)
        # print('input u shape', u.shape)

        batch_n = x.shape[0]
        batch_Ft, batch_new_x_t = [], []
        for b in range(batch_n):
            start = time.time()
            xu = np.concatenate((x[b],u[b]), 0)
            # print('xu into gp', xu)
            #print('xu shape', xu.shape[0])
            d = xu.shape[0] 
            d1 = x.shape[1] 

            flat = np.zeros((d, d))
            index=[]
            for i in range(d):
                for j in range(d):
                    index.append([i, j])

            for k in range(d):
                if index[k][0]==index[k][1]: flat[k][index[k][0]]=1 # first order index of dynamics
        
            #all_gps = [self.gp for i in range(d1)]
            

            Ft, new_x_t =[], []
            for dim in range(d1):
                #print('dim ', d1, 'current dim', dim)
                xu_copy = [xu for i in range(xu.shape[0])]
                [new_x_t_1], new_xu_t_1_std = self.all_gps[dim].predict(xu, n=0) #pred
                grad_xu_t, err_grad_y_t = self.all_gps[dim].predict(xu_copy, n=flat) #grad

                # add sample
                # new_x_t_1 = new_x_t_1/1000 # scale for debug
                if new_x_t_1 >1:
                    new_x_t_1 = 1
                elif new_x_t_1 < -1:
                    new_x_t_1 = 1
                # posterior sampling
                [new_x_t_1] = np.random.normal(new_x_t_1,new_xu_t_1_std[0],1)
                new_x_t.append(new_x_t_1)
 
                # if len(new_x_t) == d1:
                #    self.all_gps[dim].add_data(xu, new_x_t[dim]) #add new generated data on each dim

                # Ft.append(grad_xu_t/1000)#scale for debug
                # add clip
                # np.clip(grad_xu_t, -1, 1, grad_xu_t)
                Ft.append(grad_xu_t)
            
            batch_Ft.append(Ft)
            batch_new_x_t.append(new_x_t)
            # print('pred dynamic of gp', new_x_t)
            end = time.time()
            #print('used time in gp_dynamics', end - start)
        
        return batch_Ft, batch_new_x_t
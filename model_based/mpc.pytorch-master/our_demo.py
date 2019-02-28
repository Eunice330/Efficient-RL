import torch
import mpc
from mpc import mpc
from mpc.mpc import QuadCost, LinDx, GradMethods
from mpc.env_dx import pendulum
from mpc.env_dx import gp_dynamic,gp_cost
import gym
import numpy as np
import numpy.random as npr

import matplotlib.pyplot as plt

import os
import io
import base64
import tempfile
from IPython.display import HTML

from tqdm import tqdm
import time
import copy

params = torch.tensor((10., 1., 1.))
#dx = pendulum.PendulumDx(params, simple=True)
n_batch, T, mpc_T = 1, 20, 5
n_state= 3
n_ctrl=1

train_x = np.random.random((n_batch, n_state+n_ctrl))
train_y = np.random.random((n_batch, n_state))
train_r = np.random.random(n_batch)
# these train datas are used to fit gp model in dynamics and gp model in cost
env = gym.make("Pendulum-v0")
x_init = torch.from_numpy(env.reset()).unsqueeze(0).type(torch.FloatTensor)
x = x_init
u_init = None

dx = gp_dynamic.gp_dynamics_dx(train_x,train_y)
#dx = pendulum.PendulumDx(params, simple=True)
cost = gp_cost.GP_cost(train_x,train_r)
# add true samples
for i in range(30):
    u = env.action_space.sample()
    new_x, new_cost, done, info = env.step(u)
    xu = np.concatenate((x.squeeze(),u),0)
    for j in range(new_x.shape[0]):
        dx.all_gps[j].add_data(xu, new_x[j])
    cost.gp.add_data(xu, new_cost)
    x = new_x


# delete original initialization


t_dir = tempfile.mkdtemp()
print('Tmp dir: {}'.format(t_dir))

num_episode = 100
total_cost = []
for episode in range(num_episode):
    # start of a new episode
    x_init = torch.from_numpy(env.reset()).unsqueeze(0).type(torch.FloatTensor)
    x = x_init
    u_init = None
    print("episode", episode)
    episode_cost = []
    cost_for_plan = copy.deepcopy(cost)
    dx_for_plan = copy.deepcopy(dx)
    for t in range(T):
        # print("step", T)
        nominal_states, nominal_actions, nominal_objs = mpc.MPC(
            n_state, n_ctrl, mpc_T,
            n_batch = n_batch,
            u_init=u_init,
            u_lower=-2.0, u_upper=2.0,
            lqr_iter=50,
            verbose=0,
            exit_unconverged=False,
            detach_unconverged=False,
            linesearch_decay=0.2,
            max_linesearch_iter=5,
            grad_method=GradMethods.ANALYTIC,
            eps=1e-2,
            train_x = train_x,
            train_y = train_y,
            train_r = train_r
        )(x, cost_for_plan, dx_for_plan)
        print('times', t, 'x', x)
        # print('norminal states', nominal_states)
        next_action = nominal_actions[0]
        # print('next action', next_action)
        u_init = torch.cat((nominal_actions[1:], torch.zeros(1, n_batch, n_ctrl)), dim=0)
        u_init[-2] = u_init[-3]
        # _, x = dx.grad_input(x, next_action)
        new_x, new_cost, done, info = env.step(next_action)
        xu = np.concatenate((x[0], next_action[0]), 0)
        for i in range(new_x.shape[0]):
            dx.all_gps[i].add_data(xu, new_x[i])
        cost.gp.add_data(xu, new_cost)
        episode_cost.append(new_cost)
        x = new_x
        x = torch.Tensor(x).unsqueeze(0).type(torch.FloatTensor)
        # print('new x after action', x)
        print('new state after action', x)
        print('cost_step',new_cost)
    total_cost.append(np.array(episode_cost).sum())
    # print("done")
print("DONE")
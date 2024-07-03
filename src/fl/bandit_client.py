import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from utils.Buffer import Buffer
# from sklearn.
from utils.utils import create_reward_func
from env.env import env

nns = ["MLP", "CNN"]

class Bandit_client(object):
    
    def __init__(self, reward_function, env_name, gamma, narms) -> None:
        
        self.env = env(env_name)
        self.narms = narms
        self.nvars = self.env.get_var_dim()
        self.gamma = gamma
        self.arm_h = 0
        self.reward_function = create_reward_func(reward_function, self.narms + self.nvars, 1)
        self.is_nn = True if reward_function in nns else False
       
        self.total_steps = 0
        self.epoch_num = 0
        self.arm_probs = np.ones((narms,))/narms
        self.num_pulls = np.zeros((narms,))
        self.local_dataset = Buffer(self.nvars,self.narms)
        if self.is_nn:
            self.reward_optimizer = SGD(self.reward_function.parameters(), lr = 0.2)
        else:
            self.init_reward_function()
        
        
    def init_reward_function(self):
        
        
        init_batch_size = 256
        X, y = self.env.sample_first_batch(init_batch_size)
        # np.random.seed(1)s
        action_chosen = np.random.randint(self.narms, size=init_batch_size)
        for act in action_chosen:
            self.num_pulls[act] += 1
        rewards_received = y[np.arange(init_batch_size), action_chosen]
        onehot_actions =  np.zeros((action_chosen.size,self.narms))
        onehot_actions[np.arange(action_chosen.size), action_chosen] = 1
        # breakpoint()
        self.reward_function.fit(np.concatenate((X, onehot_actions),axis = 1), rewards_received)
        
        
    def run(self, num_steps):
        
        self.epoch_num += 1
        arm_scores = []
        self.best_arm = 0
        rewards = list()
        losses = list()
        # x = self.env.observe_multi(num_steps)
        # arm_scores = [self.reward_function.predict(
        #         np.concatenate((x, np.array([j/self.narms] * num_steps).reshape(-1,1)), axis= 1)) for j in range(self.narms)]
        # self.best_arm = np.argmax(arm_scores, axis = 1)
        for i in range(num_steps):        
            x = self.env.observe()
            # breakpoint()
            x_tile = np.tile(x, (self.narms, 1))
            arms = np.arange(self.narms)
            onehot_arms = np.zeros((self.narms,self.narms))
            onehot_arms[np.arange(self.narms), arms] = 1
            arm_values = np.concatenate((x_tile, onehot_arms), axis = 1)
            if self.is_nn:
                arm_scores = self.reward_function.forward(torch.FloatTensor(arm_values)).detach().numpy()
            else:
                arm_scores = self.reward_function.predict(arm_values)
            self.best_arm = np.argmax(arm_scores)

            
            self.update_probs(arm_scores)
            act = self.choose_action()
            reward = self.env.act(act)
            rewards.append(reward)
            self.num_pulls[act] += 1
            self.local_dataset.add(x, act, reward)
            
            if self.is_nn and self.local_dataset.size >= 128 and (i % 2) == 0:
                X, a, y = self.local_dataset.sample_minibatch(batch_size=128)
                a = a.astype(int)
                onehot_actions =  np.zeros((a.size,self.narms))
                onehot_actions[np.arange(a.size), a] = 1
                features = np.concatenate((X,onehot_actions),axis = 1)
                pred = self.reward_function.forward(torch.FloatTensor(features))
                # breakpoint()
                loss = F.mse_loss(pred, torch.FloatTensor(y))
                self.reward_optimizer.zero_grad()
                loss.backward()
                self.reward_optimizer.step()
                losses.append(loss.detach().numpy())
                
        if losses:
            print(f"current loss at epoch {self.epoch_num} is {np.mean(losses):.3f}")
        
        
        
        # t_end = time.time()
        # print(f"Round {self.epoch_num} total {t_end - t_start} S ")
            
        return np.mean(rewards)
        
    
    def choose_action(self, mode = "default")-> int:
        
        if mode == "default":
            return np.random.choice(self.narms, p=self.arm_probs)
        
    
    def update_probs(self, arm_scores):
        
        for j in range(self.narms):
                if j != self.best_arm:
                    # NEED to check 
                    self.arm_probs[j] = 1/(self.narms + 
                                           self.gamma * 
                                           (arm_scores[self.best_arm] - arm_scores[j]))
               
        self.arm_probs[self.best_arm] = 1 - np.sum(self.arm_probs) + self.arm_probs[self.best_arm]
        # breakpoint()
        
    def update_to_server(self):
        
        # breakpoint()
        
        # t_start = time.time()
        # breakpoint()
        if self.is_nn:
            X, a, y = self.local_dataset.sample_minibatch(batch_size=128)
            a = a.astype(int)
            onehot_actions =  np.zeros((a.size,self.narms))
            onehot_actions[np.arange(a.size), a] = 1
            features = np.concatenate((X,onehot_actions),axis = 1)
            pred = self.reward_function.forward(torch.FloatTensor(features))
            loss = F.mse_loss(pred, torch.FloatTensor(y))
            self.reward_optimizer.zero_grad()
            loss.backward()
            self.reward_optimizer.step()
            return self.reward_function.state_dict()
        else:
            X, a, y = self.local_dataset.sample_all()
            a = a.astype(int)
            onehot_actions =  np.zeros((a.size,self.narms))
            onehot_actions[np.arange(a.size), a] = 1
            features = np.concatenate((X,onehot_actions),axis = 1)
            self.reward_function.fit(np.concatenate((X,onehot_actions),axis = 1), y.ravel())
        # t_end = time.time()
        # print(f"Fitting at round {self.epoch_num} takes {t_end - t_start} s...")
        # self.local_dataset.reset()
        
            return self.reward_function.coef_
    
    def receive_from_server(self, params):
        
        if self.is_nn:
            self.reward_function.load_state_dict(params)
        else:
            self.reward_function.coef_ = params
        # self.gamma = gamma
        
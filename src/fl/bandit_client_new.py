import numpy as np
import pandas as pd
import time
import copy
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.optim import SGD, Adam
import torch.optim.lr_scheduler as lr_scheduler
from utils.Buffer import Buffer
from utils.utils import create_reward_func
from env.env import env
from collections import OrderedDict

nns = ["MLP", "CNN"]

class Bandit_client(object):
    
    def __init__(self, reward_function, env_name, gamma, narms, mu:float = 0.0,sample_method = "default", fl_structure:str = "Fedavg") -> None:
        
        self.env = env(env_name)
        self.narms = narms
        self.nvars = self.env.get_var_dim()
        self.gamma = gamma
        self.arm_h = 0
        self.reward_function = create_reward_func(reward_function, self.nvars, self.narms)
        
        self.sample_method = sample_method
        
        self.is_nn = True if reward_function in nns else False
        self.mu = mu # Only used for FedProx
       
        self.total_steps = 0
        self.epoch_num = 0
        self.arm_probs = np.ones((narms,))/narms
        self.num_pulls = np.zeros((narms,))
        self.num_updates = 1
        self.fl_structure = fl_structure
        self.local_dataset = Buffer(self.nvars,self.narms)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.is_nn:
            self.reward_function.to(self.device)
            self.reward_optimizer = SGD(self.reward_function.parameters(), lr = 0.1)
            self.scheduler = lr_scheduler.StepLR(self.reward_optimizer, step_size=10, gamma=0.1)
            self.init_model(self.reward_function)
        else:
            self.init_reward_function()
            
            
      
        self.global_params = copy.deepcopy(self.reward_function.state_dict())
        
    
    def init_model(self, model):
        for name, param in model.named_parameters():
            if 'weight' in name:
                init.kaiming_uniform_(param)
            elif 'bias' in name:
                init.constant_(param, 0.1)
                
        if self.fl_structure == "Scaffold":
            self.control_variate = OrderedDict()
            
            for name, param in model.named_parameters():
                self.control_variate[name] = torch.zeros_like(param)
            # breakpoint()
                
    def init_reward_function(self):
        
        init_batch_size = 32
        X, y = self.env.sample_first_batch(init_batch_size)
        # np.random.seed(1)s
    
        action_chosen = np.random.randint(self.narms, size=init_batch_size)
        # for act in action_chosen:
        #   self.num_pulls[act] += 1
        rewards_received = y[np.arange(init_batch_size), action_chosen]
            # if np.sum(rewards_received) <= 2:
            #     continue
            # breakpoint()
            # X_true = X[rewards_received == 1, :]
            # a_true = action_chosen[rewards_received == 1]
        # breakpoint()
        self.reward_function.fit(X, action_chosen, rewards_received)
        
        
    def run(self, num_steps):
        
        self.epoch_num += 1
        arm_scores = []
        self.best_arm = 0
        rewards = list()
        losses = list()
        arm_pull_start = time.time()
        x, y = self.env.observe_multi(num_steps)
        # breakpoint()
        if self.is_nn:
            arm_scores = self.reward_function.forward(torch.FloatTensor(x).to(self.device)).cpu().detach().numpy()
        else:
            # if x.shape[0] > 32:
            #     breakpoint() 
            # arms = np.eye(self.narms)
            # x_rep = np.repeat(x, self.narms, axis=0)
            # arms_rep = np.tile(arms, (x.shape[0], 1))
            # new_features = np.concatenate((x_rep,arms_rep),axis=1)
            arm_scores = self.reward_function.decision_function(x).reshape(-1, self.narms)
        
        breakpoint()
        best_arms = np.argmax(arm_scores, axis = 1)
        if self.sample_method == "default":
            sample_probs = self.compute_probs(arm_scores, best_arms)
        elif self.sample_method == "softmax":
            # breakpoint()
            # Improve the softmax with temperature, make it less balanced
            scaled_score = arm_scores / 0.01
            max_score = np.max(scaled_score, axis = 1).reshape(-1,1)
            scaled_score -= max_score
            sample_probs = np.exp(scaled_score) / np.sum(np.exp(scaled_score), axis = 1).reshape(-1,1)
            # sample_probs = 
            
        elif self.sample_method == "greedy":
            sample_probs = np.zeros_like(arm_scores)
            sample_probs[np.arange(arm_scores.shape[0]), best_arms] = 1
        if torch.isnan(torch.FloatTensor(sample_probs)).any():
            breakpoint()
        acts = self.choose_action(sample_probs)
        rewards = y[np.arange(y.shape[0]),acts]  
        self.local_dataset.add(x, acts, rewards)
        arm_pull_end = time.time()
        # print(f"Pulling arms at round {self.epoch_num} takes {(arm_pull_end - arm_pull_start) * 1000} ms...")
            
            
        
        if self.is_nn and self.local_dataset.size >= 128:
            # breakpoint()
            training_start = time.time()
            for _ in range(self.local_dataset.size // 128):
                X, act, y = self.local_dataset.sample_minibatch(batch_size=128)
                pred = self.reward_function.forward(X.to(self.device))
                pred_a = pred[np.arange(pred.shape[0]),act[:,0]].reshape(-1,1)
                loss = F.mse_loss(pred_a.to(self.device), y.to(self.device))
                # breakpoint()
                if self.fl_structure  == "FedProx":
                    proximal_term = 0.0
                    temp = self.global_params
                    for param_name, param_value in self.reward_function.named_parameters():
                        proximal_term += ((param_value - self.global_params[param_name]) ** 2).sum()
                    loss += self.mu * proximal_term
                    
                    
                    

                self.reward_optimizer.zero_grad()
                loss.backward()
                
                if self.fl_structure == "Scaffold":
                    for param_name, param_value in self.reward_function.named_parameters():
                        corrected_grad = param_value.grad + self.control_variate[param_name]
                        self.control_variate[param_name] += param_value.grad - corrected_grad
                        param_value.grad = corrected_grad
                self.reward_optimizer.step()
                
                # if self.fl_structure == "FedProx":
                #     for param_name, param_value in self.reward_function.named_parameters():
                #         self.reward_function.state_dict()[param_name] -= self.mu *  (param_value - self.global_params[param_name])
            self.num_updates += 1
            training_end = time.time()
            # print(f"Training at round {self.epoch_num} takes {(training_end - training_start) * 1000} ms...")
            
        return np.mean(rewards)
        
    
    def choose_action(self, probs, mode = "default")-> int:
        
        if mode == "default":
            try:
                return np.array([np.random.choice(self.narms, p=prob) for prob in probs])
            except:
                breakpoint()
        
    
    def compute_probs(self, arm_scores, best_arms):
        
        arange = np.arange(arm_scores.shape[0])
        best_scores = arm_scores[arange,best_arms].reshape(-1,1)
        best_scores = np.tile(best_scores, (1, arm_scores.shape[1]))
        arm_probs =  1/(self.narms + self.gamma * (best_scores - arm_scores))
               
        arm_probs[arange, best_arms] = 1 - np.sum(arm_probs,axis = 1) + arm_probs[arange,best_arms]
        # breakpoint()
        return arm_probs
        
    def update_to_server(self):
        
        # breakpoint()
        
        # t_start = time.time()
        # breakpoint()
        if self.is_nn:
            self.local_dataset.reset()
            if self.fl_structure == "Scaffold":
                return self.reward_function.state_dict(), self.control_variate, self.num_updates
            elif self.fl_structure == "FedProx":
                return self.reward_function.state_dict(), self.num_updates
            else:
                return self.reward_function.state_dict()
        else:
            # breakpoint()
            X, a, y = self.local_dataset.sample_all()
            a = a.astype(int)
            # rewards_received = y[np.arange(X.shape[0]), a]
            try:
                self.reward_function.fit(X, a, y.ravel())
            except:
                breakpoint()
        # t_end = time.time()
        # print(f"Fitting at round {self.epoch_num} takes {t_end - t_start} s...")
        # self.local_dataset.reset()
            self.local_dataset.reset()
            return self.reward_function.base_algorithm.coef_
    
    def receive_from_server(self, params):
        # breakpoint()
        if self.is_nn:
            if self.fl_structure == "Scaffold":
                params, control_variate = params
                self.global_params = params
                if control_variate is not None:
                    self.control_variate = control_variate
            else:
                self.reward_function.load_state_dict(params)
                self.global_params = params
            
        else:
            self.reward_function.base_algorithm.coef_ = params
        
        
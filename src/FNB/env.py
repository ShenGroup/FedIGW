import pandas as pd, numpy as np, re
from utils import parse_data

dataset_path = "/home/lican/workarea/Kun/Federated_Contextual_Bandits/src/datasets/"

class env(object):
    
    def __init__(self, env_name) -> None:
        
        if env_name == "Bibtex":
            self.X, self.y = parse_data(f"{dataset_path}{env_name}.txt")
            self.cur_idx = 0
        else:
            self.X, self.y = parse_data(f"{dataset_path}{env_name}.txt")
            self.cur_idx = 0
        
    
    def observe(self):
        
        self.cur_idx = np.random.randint(0, self.X.shape[0]-1)
        return self.X[self.cur_idx,:]
    
    def get_var_dim(self):
        
        return self.X.shape[1]
    
    def get_act_dim(self):
        
        return self.y.shape[1]
    
    def act(self, action):
        
        return self.y[self.cur_idx, action]
    
    def observe_multi(self, num):
        
        self.cur_idx = np.random.random_integers(0, self.X.shape[0]-1, size = (num, ))
        return self.X[self.cur_idx,:], self.y[self.cur_idx,:]
    
    def step(self):
        self.cur_idx = np.random.randint(0, self.X.shape[0]-1)
        contex = self.X[self.cur_idx, :].reshape(-1,1)
        arm_num = self.get_act_dim()
        contexs = np.tile(contex, arm_num)
        rewards = self.y[self.cur_idx, :]
        # breakpoint()
        return contexs.transpose(), rewards
    
    def sample_first_batch(self,batch_size):
        
        return self.X[:batch_size,:], self.y[:batch_size, :]
    
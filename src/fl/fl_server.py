import numpy as np
import time
import pandas as pd
import time
import copy
import hashlib
import torch
import torch.nn.init as init
# from .bandit_client import Bandit_client
from .bandit_client_new import Bandit_client
from utils.utils import create_reward_func
from env.env import env
from collections import OrderedDict

nns = ["MLP", "CNN"]

class FL_Server():
    
    def __init__(
        self,
        env_name,
        reward_function,
        client_num,
        gamma,
        sample_method,
        fl_method
        )->None: 
        current_time = str(time.time())
        hash_object = hashlib.sha256(current_time.encode())
        self.hash_code = hash_object.hexdigest()[:16]
        self.eval_env = env(env_name)
        self.env_name = env_name
        action_dim = self.eval_env.get_act_dim()
        feature_dim = self.eval_env.get_var_dim()
        self.global_function = create_reward_func(reward_function, feature_dim, action_dim)
        self.env_name = env_name
        self.reward_func = reward_function
        self.gamma = gamma
        self.sample_method = sample_method 
        
        
        
        # state_dim = self.eval_env.get_var_dim()
        if fl_method == "FedProx" or fl_method == "Scaffold":
            self.local_update_nums = list()
        if fl_method == "Scaffold":
            self.control_variates = list()
        self.fl_method = fl_method
        self.clients = self.create_clients(client_num, env_name, reward_function,gamma, action_dim, sample_method)
        self.is_nn = True if reward_function in nns else False
        self.client_params = list()
        self.rewards = list()
        
        
        self.total_it = 0
        if self.is_nn:
            self.init_model(self.global_function)
        
        
    def create_clients(self, client_num, env_name, reward_function,gamma, narms, sample_method):
        
        clients = list()
        for _ in range(client_num):
            
            clients.append(Bandit_client(reward_function,env_name,gamma, narms, mu = 0.01, fl_structure = self.fl_method, sample_method= sample_method))
            
            
        return clients
    
    def init_model(self, model):
        for name, param in model.named_parameters():
            if 'weight' in name:
                init.kaiming_uniform_(param)
            elif 'bias' in name:
                init.constant_(param, 0.1)
    
    def train(self, comm_round, local_steps, total_rounds, client_num):
        
        rounds = list()
        cur_round = 0
        for i in range(total_rounds):
            
            self.rewards.append([])
            if i == 0:
                for client in self.clients:
                    if self.is_nn:
                        if self.fl_method == "Scaffold":
                            client.receive_from_server((self.global_function.state_dict(), None))
                        else:
                            client.receive_from_server(self.global_function.state_dict())
                    else:
                        break
            for client in self.clients:
                # if i >= 10:
                #     breakpoint()
                local_start_time = time.time()
                if local_steps == 0:
                    # breakpoint()
                    avg_reward = client.run(min(2 ** i, 4096))
                else:
                    avg_reward = client.run(local_steps)
                local_end_time = time.time()
                self.rewards[-1].append(avg_reward)
            if (i + 1) % comm_round == 0:
                sampled_clients = np.random.choice(self.clients, client_num, replace = False)
                for client in sampled_clients:
                    if self.fl_method == "FedAvg":
                        self.client_params.append(client.update_to_server())
                    elif self.fl_method == "FedProx":
                        params, update_num = client.update_to_server()
                        self.client_params.append(params)
                        self.local_update_nums.append(update_num)
                    elif self.fl_method == "Scaffold":
                        params, control_variate, update_num = client.update_to_server()
                        self.client_params.append(params)
                        self.local_update_nums.append(update_num)
                        self.control_variates.append(control_variate)  
            # print(f"Finished round {i}, average_reward {avg_reward:.4f}, time {local_end_time - local_start_time:.4f}")
                    # self.flrountine(self.fl_method)
            if local_steps == 0:
                cur_round += min(2**i, 256)
            else:
                cur_round += local_steps
            rounds.append(cur_round)
                    
            if (i + 1) % comm_round == 0:
                
                fl_start_time = time.time()
                global_param = self.flrountine(self.fl_method)
                for client in self.clients:
                    client.receive_from_server(global_param)
                self.client_params = list()
                self.control_variates = list()
                self.local_update_nums = list()
                fl_end_time = time.time()
                # print(f"FL finished round {i}, time {fl_end_time - fl_start_time:.4f}")
                # breakpoint()
                df_log = pd.DataFrame({
                    "rounds": rounds,
                    "mean_rewards" : np.mean(self.rewards, axis = 1)
                                })
                df_log.to_csv(f"../results/Diff_feds/{self.env_name}_method_{self.reward_func}_fl_{self.fl_method}_bandit_step_{local_steps}_client_num_{len(self.clients)}_gamma_{self.gamma}_sample_method_{self.sample_method}_time_{self.hash_code}.csv", index = False)
                # breakpoint()
                # np.save("../results/nn_results", np.array(self.rewards))
                print(f"Current round {i}, average_reward {np.mean(self.rewards[i]):.4f}")
                
                
                   
            
                    
                    
                    
                    
    def flrountine(self, fl_method):
        
        if self.is_nn:
            global_params = OrderedDict()
            # breakpoint()
            
            for name in self.client_params[0]:
                global_params[name] = torch.zeros_like(self.client_params[0][name])
            if fl_method == "Scaffold":
                # breakpoint()
                global_control_variate = OrderedDict()
                for name in self.control_variates[0]:
                    global_control_variate[name] = torch.zeros_like(self.control_variates[0][name])
        else:
            global_params = np.zeros_like(self.client_params[0])
        if fl_method == "FedAvg":
            weight = 1/len(self.client_params)
            # breakpoint()
            for i, param in enumerate(self.client_params):
                if self.is_nn:
                    for name in param:
                        global_params[name] += weight * param[name]
                else:
                    global_params += weight * param
            
            return global_params
        elif fl_method == "FedProx":
            total_updates = np.sum(self.local_update_nums)
            for i, param in enumerate(self.client_params):
                
                weight = self.local_update_nums[i] / total_updates
                for name in param:
                    global_params[name] += weight * param[name]
                    
                    
            return global_params
        elif fl_method == "Scaffold":
            total_updates = np.sum(self.local_update_nums)
            for i, param in enumerate(self.client_params):
                weight = self.local_update_nums[i] / total_updates
                for name in param:
                    global_params[name] += weight * (param[name] - self.control_variates[i][name])
                    global_control_variate[name] += weight * self.control_variates[i][name]
                
                
        
            return (global_params, global_control_variate)

            
            
            
                
    def get_it(self):
        return self.total_it
    

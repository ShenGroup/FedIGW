import numpy as np
import torch


class Buffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, 1))
		self.reward = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, reward):
		sample_size = state.shape[0]
		if self.ptr + sample_size >= self.max_size:
			self.ptr = (self.ptr + sample_size) % self.max_size
		self.state[self.ptr:self.ptr+sample_size] = state
		self.action[self.ptr:self.ptr+sample_size] = action.reshape(-1,1)
		self.reward[self.ptr:self.ptr+sample_size] = reward.reshape(-1,1)

		self.ptr = (self.ptr + sample_size) % self.max_size
		self.size = min(self.size + sample_size, self.max_size)


	def sample_all(self):

		return (
			np.array(self.state[:self.size,:]),
			np.array(self.action[:self.size,:]),
			np.array(self.reward[:self.size,:])
		)
  
  
	def sample_minibatch(self, batch_size):
     
		batch_idx = np.random.randint(0,self.size, batch_size)

		return (
			torch.FloatTensor(self.state[batch_idx,:]),
			torch.LongTensor(self.action[batch_idx,:]),
			torch.FloatTensor(self.reward[batch_idx,:])
		)
  
  
  
  
	def reset(self):
     
		self.ptr = 0
		self.size = 0
		max_size = self.state.shape[0]
		state_dim = self.state.shape[1]
		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, 1))
		self.reward = np.zeros((max_size, 1))

        


	
import torch
import numpy as np
import pandas as pd
import collections
import torch.nn as nn

def weight_init(size, val = None):
	val = val or size[0]
	wgt = 1/np.sqrt(val)
	return torch.Tensor(size).uniform_(-wgt, wgt)

class ActorNetwork(nn.Module):
	def __init__(self, stdim, acdim, aclim):
		super(ActorNetwork, self).__init__()
		self.stdim = stdim
		self.acdim = acdim
		self.aclim = aclim

		self.firstlay = nn.Linear(self.stdim, 256)
		torch.nn.init.xavier_uniform(self.firstlay.weight)
		#self.firstlay.weight.data = weight_init(self.firstlay.weight.data.size())

		self.secondlay = nn.Linear(256, 128)
		torch.nn.init.xavier_uniform(self.secondlay.weight)
		#self.secondlay.weight.data = weight_init(self.secondlay.weight.data.size())

		self.thirdlay = nn.Linear(128, 64)
		torch.nn.init.xavier_uniform(self.thirdlay.weight)
		#self.thirdlay.weight.data = weight_init(self.thirdlay.weight.data.size())

		self.finallay = nn.Linear(64, self.acdim)
		torch.nn.init.xavier_uniform(self.finallay.weight)
		self.finallay.weight.data.uniform_(-0.003, 0.003)

	def forward(self, state):
		out_first = nn.functional.relu(self.firstlay(state))
		out_second = nn.functional.relu(self.secondlay(out_first))
		out_third = nn.functional.relu(self.thirdlay(out_second))
		out_final = nn.functional.tanh(self.finallay(out_third))

		return out_final*self.aclim
		
class CriticNetwork(nn.Module):
	def __init__(self, stdim, acdim):
		super(CriticNetwork, self).__init__()
		self.stdim = stdim
		self.acdim = acdim

		self.firstlay_state = nn.Linear(self.stdim, 256)
		torch.nn.init.xavier_uniform(self.firstlay_state.weight)
		#self.firstlay_state.weight.data = weight_init(self.firstlay_state.weight.data.size())

		self.secondlay_state = nn.Linear(256, 128)
		torch.nn.init.xavier_uniform(self.secondlay_state.weight)
		#self.secondlay_state.weight.data = weight_init(self.secondlay_state.weight.data.size())

		self.firstlay_action = nn.Linear(self.acdim, 128)
		torch.nn.init.xavier_uniform(self.firstlay_action.weight)
		#self.firstlay_action.weight.data = weight_init(self.firstlay_action.weight.data.size())

		self.firstlay_state_action = nn.Linear(256, 128)
		torch.nn.init.xavier_uniform(self.firstlay_state_action.weight)
		#self.firstlay_state_action.weight.data = weight_init(self.firstlay_state_action.weight.data.size())

		self.finallay_state_action = nn.Linear(128, 1)
		torch.nn.init.xavier_uniform(self.finallay_state_action.weight)
		self.finallay_state_action.weight.data.uniform_(-0.003, 0.003)

	def forward(self, state, action):
		out_state_first = nn.functional.relu(self.firstlay_state(state))
		out_state_second = nn.functional.relu(self.secondlay_state(out_state_first))
		out_action = nn.functional.relu(self.firstlay_action(action))
		state_action = torch.cat((out_state_second, out_action), dim=1)

		out_state_action_first = nn.functional.relu(self.firstlay_state_action(state_action))
		q_state_action = self.finallay_state_action(out_state_action_first)

		return q_state_action

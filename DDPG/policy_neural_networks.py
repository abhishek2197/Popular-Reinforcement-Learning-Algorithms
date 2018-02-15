import torch
import numpy as np
import pandas as pd
import collections
import torch.nn as nn

def weight_init(size):
	wgt = 1/np.sqrt(size[0])
	return torch.Tensor(size).uniform_(-wgt, wgt)

class ActorNetwork(nn.Module):
	def __init__(self, stdim, acdim, aclim):
		super(ActorNetwork, self).__init__()
		self.stdim = stdim
		self.acdim = acdim
		self.aclim = aclim

		self.firstlay = nn.Linear(self.stdim, 256)
		self.firstlay.weight.data = weight_init(self.firstlay.weight.data.size())

		self.secondlay = nn.Linear(256, 128)
		self.secondlay.weight.data = weight_init(self.secondlay.weight.data.size())

		self.thirdlay = nn.Linear(128, 64)
		self.thirdlay.weight.data = weight_init(self.thirdlay.weight.data.size())

		self.finallay = nn.Linear(64, acdim)
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
		self.firstlay_state.weight.data = weight_init(self.firstlay_state.weight.data.size())

		self.secondlay_state = nn.Linear(256, 128)
		self.secondlay_state.weight.data = weight_init(self.secondlay_state.weight.data.size())

		self.firstlay_action = nn.Linear(acdim, 128)
		self.firstlay_action.weight.data = weight_init(self.firstlay_action.weight.data.size())

		self.firstlay_state_action = nn.Linear(256, 128)
		self.firstlay_state_action.weight.data = weight_init(self.firstlay_state_action.weight.data.size())

		self.finallay_state_action = nn.Linear(128, 1)
		self.finallay_state_action.weight.data.uniform_(-0.003, 0.003)

	def forward(self, state, action):
		out_state_first = nn.functional.relu(self.firstlay_state(state))
		out_state_second = nn.functional.relu(self.secondlay_state(out_state_first))
		out_action = nn.functional.relu(self.firstlay_action(action))
		state_action = torch.cat((out_state_second, out_action), dim=1)

		out_state_action_first = nn.functional.relu(self.firstlay_state_action(state_action))
		q_state_action = nn.functional.tanh(self.finallay_state_action(out_state_action_first))

		return q_state_action

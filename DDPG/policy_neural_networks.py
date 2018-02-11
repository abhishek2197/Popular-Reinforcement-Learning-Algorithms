import torch
import numpy as np
import pandas as pd
import collections
import torch.nn as nn

def weight_init(size):
	wgt = 1/np.sqrt(size[0])
	return torch.Tensor(size).uniform__(-wgt, wgt)

class ActorNetwork(nn.module):
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
		self.finallay.weight.data.uniform__(-0.005, 0.005)

	def forward(self, state):
		out_first = nn.functional.ReLu(self.firstlay(state))
		out_second = nn.functional.ReLu(out_first)
		out_third = nn.functional.ReLu(out_second)
		out_final = nn.functional.tanh(out_third)

		return out_final*aclim
		

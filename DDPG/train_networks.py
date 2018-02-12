import torch.nn as nn
import os
from torch.autograd import Variable
import math
import policy_neural_networks
import replay_memory

class training:

	def __init__(self, stdim, acdim, aclim, mem):
		self.stdim = stdim
		self.acdim = acdim
		self.aclim = aclim
		self.mem = mem

		self.actor = policy_neural_networks.ActorNetwork(self.stdim, self.acdim, self.aclim)
		self.target_actor = policy_neural_networks.ActorNetwork(self.stdim, self.acdim, self.aclim)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), 0.004)

		self.critic = policy_neural_networks.CriticNetwork(self.stdim, self.acdim)
		self.target_critic = policy_neural_networks.CriticNetwork(self.stdim, self.acdim)
		self.actor_optimizer = torch.optim.Adam(self.critic.parameters(), 0.004)

		for taract_par, act_par in zip(self.target_actor.parameters(), self.actor.parameters()):
			taract_par.data.copy(act_par)

		for tarcrit_par, crit_par in zip(self.target_critic.parameters(), self.critic.parameters()):
			tarcrit_par.data.copy(act_par)	

		


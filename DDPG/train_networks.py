import torch.nn as nn
import os
from torch.autograd import Variable
import math
import policy_neural_networks
import replay_memory

gamma = 0.999
Tau = 0.95
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
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), 0.004)

		for taract_par, act_par in zip(self.target_actor.parameters(), self.actor.parameters()):
			taract_par.data.copy(act_par)

		for tarcrit_par, crit_par in zip(self.target_critic.parameters(), self.critic.parameters()):
			tarcrit_par.data.copy(crit_par)	

			
	def learn(self):
		s, a, r, s1  = self.mem.sample(10)
		s = Variable(torch.from_numpy(s))
		a = Variable(torch.from_numpy(a))
		r = Variable(torch.from_numpy(r))
		s1 = Variable(torch.from_numpy(s1))

		a1 = self.target_actor.forward(s1).detach()
		q_val = self.target_critic.forward(s1, a1).detach()
		expected_rew = r + gamma*(q_val)

		pred_rew = self.critic.forward(s, a).detach()
		loss_crit = nn.functional.smooth_l1_loss(pred_rew, expected_rew)
		self.critic.optimizer.zero_grad()
		self.critic_optimizer.backward()
		self.critic_optimizer.step()

		pred_action = self.actor.forward(s)
		loss_actor = -1*torch.sum(self.critic.forward(s,pred_action))
		self.actor_optimizer.zero_grad()
		self.actor_optimizer.backward()
		self.actor_optimizer.step()

		for target_par, par in zip(self.target_actor.parameters(), self.actor.parameters()):
		target_para.data.copy_(
			para.data*Tau + target_para.data*(1.0-Tau) 
		)

		for target_par, par in zip(self.target_critic.parameters(), self.critic.parameters()):
		target_para.data.copy_(
			para.data*Tau + target_para.data*(1.0-Tau) 
		)

	def next_action(self, state):
		s = Variable(torch.from_numpy(state))
		action = self.target_actor.forward(s).detach()
		return action.data.numpy()
from __future__ import division
import torch.nn as nn
import os
from torch.autograd import Variable
import math
import torch
import noise
import policy_neural_networks
import replay_memory

gamma = 0.99
Tau = 0.001
class Training:

	def __init__(self, stdim, acdim, aclim, mem):
		self.stdim = stdim
		self.acdim = acdim
		self.aclim = aclim
		self.mem = mem
		self.actnoise = noise.OrnsteinUhlenbeckActionNoise(self.acdim)

		self.actor = policy_neural_networks.ActorNetwork(self.stdim, self.acdim, self.aclim)
		self.target_actor = policy_neural_networks.ActorNetwork(self.stdim, self.acdim, self.aclim)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), 0.001)

		self.critic = policy_neural_networks.CriticNetwork(self.stdim, self.acdim)
		self.target_critic = policy_neural_networks.CriticNetwork(self.stdim, self.acdim)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), 0.001)

		for taract_par, act_par in zip(self.target_actor.parameters(), self.actor.parameters()):
			taract_par.data.copy_(act_par.data)

		for tarcrit_par, crit_par in zip(self.target_critic.parameters(), self.critic.parameters()):
			tarcrit_par.data.copy_(crit_par.data)	

			
	def learn(self):
		s, a, r, s1  = self.mem.sample(128)
		s = Variable(torch.from_numpy(s))
		a = Variable(torch.from_numpy(a))
		r = Variable(torch.from_numpy(r))
		s1 = Variable(torch.from_numpy(s1))

		a1 = self.target_actor.forward(s1).detach()
		q_val = torch.squeeze(self.target_critic.forward(s1, a1).detach())
		expected_rew = r + gamma*(q_val)

		pred_rew = torch.squeeze(self.critic.forward(s, a).detach())
		loss_crit = nn.functional.smooth_l1_loss(pred_rew, expected_rew)
		loss_crit.requires_grad=True
		self.critic_optimizer.zero_grad()
		loss_crit.backward()
		self.critic_optimizer.step()

		pred_action = self.actor.forward(s)
		loss_actor = -1*torch.sum(self.critic.forward(s,pred_action))

		self.actor_optimizer.zero_grad()
		loss_actor.backward()
		self.actor_optimizer.step()

		for target_para, para in zip(self.target_actor.parameters(), self.actor.parameters()):
			target_para.data.copy_(para.data*Tau + target_para.data*(1.0-Tau))

		for target_para, para in zip(self.target_critic.parameters(), self.critic.parameters()):
			target_para.data.copy_(para.data*Tau + target_para.data*(1.0-Tau))

	def next_action(self, state):
		s = Variable(torch.from_numpy(state))
		action = self.actor.forward(s).detach()
		new_action = action.data.numpy() + (self.actnoise.sample() * self.aclim)
		return new_action

	def save_models(self, episode_no):
		torch.save(self.target_actor.state_dict(), './Models/' + str(episode_no) + '_actor.pt')
		torch.save(self.target_critic.state_dict(), './Models/' + str(episode_no) + '_critic.pt')
		print 'Models saved successfully'

	def load_models(self, episode):
		self.actor.load_state_dict(torch.load('./Models/' + str(episode) + '_actor.pt'))
		self.critic.load_state_dict(torch.load('./Models/' + str(episode) + '_critic.pt'))
		for taract_par, act_par in zip(self.target_actor.parameters(), self.actor.parameters()):
			taract_par.data.copy_(act_par.data)

		for tarcrit_par, crit_par in zip(self.target_critic.parameters(), self.critic.parameters()):
			tarcrit_par.data.copy_(crit_par.data)
		
		print 'Models loaded succesfully'
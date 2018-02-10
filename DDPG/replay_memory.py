import numpy as np
import os
import random
import pandas as pd
import collections

class ReplayMemory:

	def __init__(self, size):
		self.memory = deque(maxlen=size)
		self.maxisize = size
		self.len = 0

	def add_transition(self, st, act, r, st1):
		self.len+=1
		if self.len>maxisize:
			self.len = maxisize
		self.memory.append(st, act, r, st1)
		
	def len(self):
		return self.len			

	def sample(self, size):
		buff = random.sample(self.memory, min(size, self.len))
		states = []
		actions = []
		reward = []
		next_states = []
		for a in buf:
			states.append(a[0])
			actions.append(a[1])
			reward.append(a[2])
			next_states.append(a[3])
		states_np = np.float32(states)
		actions_np = np.float32(actions)
		reward_np = np.float32(reward)
		next_states_np = np.float32(next_states)
		return states_np, actions_np, reward_np, next_states_np



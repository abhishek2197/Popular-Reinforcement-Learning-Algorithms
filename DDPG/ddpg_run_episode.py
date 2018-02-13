from __future__ import division
import gym
import numpy as np 
import pandas as pd 
import collections
import torch
import psutil
import replay_memory
import policy_neural_networks
import train_networks
from itertools import count



no_of_episodes = 5000
no_of_steps = 5000

if __name__ == '__main__':
	env = gym.make('Walker2d-v1')

	mem = replay_memory.ReplayMemory(10000000)
	trainer = train_networks.Training(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0], mem)

	
	
	for i_episode in count(1):	
		num_episodes = 0
		num_steps = 0
		reward_batch = 0
		while num_steps < 5000:
			obs = env.reset()
			 
			reward_sum = 0
			for k in range(10000):
					#if i%100 == 0:
				env.render()	
			
				state = np.float32(obs)
				action = trainer.next_action(state)

				next_obs, reward, done, ext = env.step(action)
				reward_sum += reward
			
				if done==False:
					next_state = np.float32(next_obs)
				mem.add_transition(state, action, reward, np.float32(next_state))
			
				trainer.learn()
				obs = next_obs
				if done:
					break
			num_episodes += 1
			num_steps += (k-1)
        	reward_batch += reward_sum		
		reward_batch /= num_episodes		
		print "Episode No - "+str(i_episode)+" "+str(reward_batch)+""		

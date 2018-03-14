from __future__ import division
# import sys
# sys.path.append('/usr/local/lib/python3.6/site-packages')
import gym
import argparse
import sys
import numpy as np 
import pandas as pd 
import collections
import torch
import psutil
import replay_memory
import policy_neural_networks
import train_networks
import noise
from itertools import count


no_of_episodes = 1000
no_of_steps = 250000

parser = argparse.ArgumentParser(description='PyTorch DDPG')

parser.add_argument('--env-name', default="Walker2d-v2", metavar='G',
                    help='name of the environment to run')

parser.add_argument('--render', action='store_true',
                    help='render the environment')
args = parser.parse_args()


if __name__ == '__main__':

	env =  gym.make(args.env_name)
	mem = replay_memory.ReplayMemory(1000000)
	trainer = train_networks.Training(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0], mem)

	
	
	# for i_episode in count(1):	
	# 	num_episodes = 0
	# 	num_steps = 0
	# 	reward_batch = 0
	# 	while num_steps < 1000:
	num_episodes = 0
	reward_batch = 0
	
	for i in range(no_of_episodes):
		obs = env.reset()
			 
		reward_sum = 0
		for k in range(no_of_steps):
			if args.render:
				env.render()	
		
			state = np.float32(obs)
			action = trainer.next_action(state)

			next_obs, reward, done, ext = env.step(action)
			reward_sum += reward
			
			if done:
				next_state = None
			else:	
				next_state = np.float32(next_obs)
				mem.add_transition(state, action, reward, np.float32(next_state))
			
			trainer.learn()
			obs = next_obs
			if done:
				break

		num_episodes += 1
		reward_batch += reward_sum		
		average_reward = reward_batch/num_episodes		
		print ("Episode No - "+str(i)+" "+str(average_reward)+"")
		if i%100==0:
			trainer.save_models(i)		

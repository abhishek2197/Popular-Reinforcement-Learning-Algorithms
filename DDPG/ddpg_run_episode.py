from __future__ import division
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

import plotly
import plotly.graph_objs as go
from plotly.graph_objs import Layout,Scatter

def main(env_name="Walker2d-v2", render=False, number_of_batches=20, maximum_steps=500):

	''' no_of_episodes = 10000
	no_of_steps = 1000000'''

	''' parser = argparse.ArgumentParser(description='PyTorch DDPG')

	parser.add_argument('--env-name', default="Walker2d-v2", metavar='G',
				help='name of the environment to run')

	parser.add_argument('--render', action='store_true',
				help='render the environment')
	args = parser.parse_args()'''

	env =  gym.make(env_name)
	mem = replay_memory.ReplayMemory(1000000)
	trainer = train_networks.Training(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0], mem)
	batch_size = 500
	plot_rew = []
	for i_episode in range(number_of_batches):
		num_steps = 0
		reward_batch = 0
		num_episodes = 0
		while num_steps < batch_size:
			obs = env.reset()
			reward_sum = 0
			for k in range(maximum_steps): # Don't infinite loop while learning
				if render:
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
			num_steps += (k-1)
		reward_batch /= num_episodes
		plot_rew.append(reward_batch)
		print ("Episode No - "+str(i_episode)+" "+str(reward_batch)+"")
		if i_episode%100==0:
			trainer.save_models(i_episode)

	plot_epi = []
	for i in range (number_of_batches):
		plot_epi.append(i)
	trace = go.Scatter( x = plot_epi, y = plot_rew )
	layout = go.Layout(title='DDPG',xaxis=dict(title='Episodes', titlefont=dict(family='Courier New, monospace',size=18,color='#7f7f7f')),
	yaxis=dict(title='Average Reward', titlefont=dict(family='Courier New, monospace',size=18,color='#7f7f7f')))

	plotly.offline.plot({"data": [trace], "layout": layout},filename='DDPG.html',image='jpeg')


if __name__ == '__main__':
	no_of_episodes = 10000
	no_of_steps = 1000000

	parser = argparse.ArgumentParser(description='PyTorch DDPG')

	parser.add_argument('--env-name', default="Walker2d-v2", metavar='G',
				help='name of the environment to run')

	parser.add_argument('--render', action='store_true',
				help='render the environment')
	args = parser.parse_args()
	main(env_name=args.env_name, render=args.render)

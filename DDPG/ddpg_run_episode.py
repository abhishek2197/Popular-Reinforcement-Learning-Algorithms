import gym
import numpy as np 
import pandas as pd 
import collections
import torch
import replay_memory
import policy_neural_networks
import train_networks


no_of_episodes = 6000
no_of_steps = 2000

if __name__ == '__main__':
	env = gym.make("Walker 2d-v1")

	mem = replay_memory.ReplayMemory(10000000)
	trainer = train_networks.Training(env.observation_space.shape[0], env.action_space.space[0], env.action_space.high[0], mem)

	
	avg_reward = 0.0
	for i in range(no_of_episodes):
		obs = env.reset()
		sum_reward = 0
		print "Episode No - "+i 
		for j in range(no_of_steps):
			env.render()
			state = np.float32(obs)
			action = train_networks.next_action(state)

			next_obs, reward, done, ext = env.step(action)
			sum_reward += reward
			
			if !done:
				next_state = np.float32(next_state)
				mem.add_transition(state, action, reward, np.float32(next_state))
			
			trainer.learn()
			obs = next_obs
			if done:
				break

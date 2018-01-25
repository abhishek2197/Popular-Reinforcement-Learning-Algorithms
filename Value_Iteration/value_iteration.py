import gym
import numpy as np
import pandas as pd
from gym import wrappers
from gym import spaces, utils
from gym.envs.toy_text import discrete


def value_iterate(env):
	tot_iterations = 10000
	gamma = 1
	no_states = env.observation_space.n
	no_actions = env.action_space.n
	v = np.zeros(no_states)
	for i in range(tot_iterations):
		pre_v = np.copy(v)
		error_epsilon = 1e-30
		for s in range(env.nS):
			qval = [sum([reward + prob*gamma*pre_v[st] for prob, st, reward, _ in env.P[s][a]]) for a in range(env.nA)] 		
		v[s] = max(qval)		
		error = np.sum(np.fabs(v - pre_v))
		if(error<= error_epsilon):
			print "Value Function has converged"
			break
	return v



if __name__ == '__main__':
    env = gym.make('Taxi-v1')
    print env.nS
    optimal_value_function = value_iterate(env)
    for i in range(int(env.nS):
         print optimal_value_function[i]

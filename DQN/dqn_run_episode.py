import sys
import gym
import torch
#import pylab
import random
import numpy as np
from collections import deque
from datetime import datetime
from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import time
from agent import *
from model import *
from hyperparameters import *
from skimage.transform import resize
from skimage.color import rgb2gray
from copy import deepcopy

def get_frame(input):
    x = np.uint8(resize(rgb2gray(input), (84, 84), mode='reflect') * 255)
    return x

def get_init_state(history, s):
    for i in range(frames_history):
        history[i, :, :] = get_frame(s)

def convert_to_variable(x):
    state = Variable(torch.Tensor(x), requires_grad=True)
    return state

def train():
    rewards, episodes = [], [] 

    env = gym.make('BreakoutDeterministic-v4')
    action_size = env.action_space.n
    agent = Agent(action_size)
    evaluation_reward = deque(maxlen=evaluation_reward_length)
    frame = 0
    memory_size = 0
  
    for e in range(no_of_episodes):
        done = False
        score = 0
        history = np.zeros([5, 84, 84], dtype=np.uint8)
        step = 0
        state = env.reset()
        get_init_state(history, state)

        while not done:
            step += 1
            frame += 1
            # Select and perform an action
            env.render()
            time.sleep(0.1)

            action = agent.get_action(np.float32(history[:4, :, :]) / 255.)
            next_state, reward, done, info = env.step(action)

            frame_next_state = get_frame(next_state)
            history[4, :, :] = frame_next_state

            r = np.clip(reward, -1, 1)

            # Store the transition in memory 
            agent.replay_memory.push(deepcopy(frame_next_state), action, r, done)
            
            # Start training after random sample generation
            if(frame >= train_frame):
                agent.train_action_net(frame)
                # Update the target network
                if(frame % update_iteration)== 0:
                    agent.update_target_action_net()
            score += reward
            history[:4, :, :] = history[1:, :, :]

            if frame % 50000 == 0:
                print('now time : ', datetime.now())
                rewards.append(np.mean(evaluation_reward))
                episodes.append(e)
                #pylab.plot(episodes, rewards, 'b')
                #pylab.savefig("./breakout_dqn.png")

            if done:
                evaluation_reward.append(score)
                # every episode, plot the play time
                print("episode:", e, "  score:", score, "  memory length:",
                    len(agent.replay_memory), "  epsilon:", agent.epsilon, "   steps:", step,
                    "    evaluation reward:", np.mean(evaluation_reward))

                # if the mean of scores of last 10 episode is bigger than 400
                # stop training
                if np.mean(evaluation_reward) > 10:
                    torch.save(agent.model, "./breakout_dqn")
                    sys.exit()

train()
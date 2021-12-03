import gym
import numpy as np
#import matplotlib.pyplot as plt

env = gym.make('LunarLanderContinuous-v2')

print(f'\n\t\t\t\tINITIAL STATE\nSTATE : \n\tSHAPE : {env.observation_space.shape}\n\tRANGE : [{env.observation_space.low[0]},{env.observation_space.high[0]}]\n')
print(f'ACTION : \n\tSHAPE : {env.action_space.shape}\n\tRANGE : [{env.action_space.low[0]},{env.action_space.high[0]}]\n')
print(f'REWARD : {list(env.reward_range)}\n')

initial_state = env.reset()

print(f'INITIAL STATE : \n\t[x, y]        : [{initial_state[0]}, {initial_state[1]}]')
print(f'\t[Vx, Vy]      : [{initial_state[2]}, {initial_state[3]}]')
print(f'\t[Theta, Vw]   : [{initial_state[4]}, {initial_state[5]}]')
print(f'\t[Left, Right] : [{initial_state[6]}, {initial_state[7]}]')


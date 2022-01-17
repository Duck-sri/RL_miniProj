import os
from sys import path_hooks
import time
import numpy as np
import pickle as pkl

from typing import Union

from .stability import check_stability_local

# TODO have to make a save all the data ink
# TODO remove all random seed
# TODO make seeds as a global constant

TO_SAVE = {
  'ddpg' : {
    'setpoint' : [],
    'disturbance' : []
  }, 
  'td3' : {
    'setpoint' : [],
    'disturbance' : []
  }
}
FILE_NAMES = [f"{x}_{y}_logs.pkl" for x in TO_SAVE.keys() for y in TO_SAVE[x].keys() ]

def init_save(save_dir:str,force_clear:bool=False):
  save_dir = os.path.abspath(save_dir)
  if not os.path.exists(save_dir) :
    raise FileNotFoundError

  start = {
    'episode_wise_stability': [],
    'all_states' : [],
    'all_scores' : [],
  }

  for name in FILE_NAMES:
    path = os.path.join(save_dir,name)
    if os.path.exists(path): 
      if not force_clear:
        continue

    with open(path,'wb') as handle:
      pkl.dump(start,handle,pkl.HIGHEST_PROTOCOL)


def save_logs(file_name:str,data:dict):
  path = os.path.abspath(file_name)
  if not os.path.exists(path):
    raise FileNotFoundError(f"You suck, File: {path} does not exists")
    
  with open(path,'rb') as handle:
    prev_data = pkl.load(handle)

  prev_data['all_states'] += data['all_states']
  prev_data['all_scores'] += data['all_scores']
  prev_data['episode_wise_stability'] += data['episode_wise_stability']

  with open(path,'wb') as handle:
    pkl.dump(prev_data,handle,pkl.HIGHEST_PROTOCOL)

  print(f"Done saving in file... {path}")

def train_model(agent,env,epochs:int,save_dir:str,render:bool=True,load:bool=True,eval_:bool=False,saveCycle:int=25,change_mid:bool=False,verbose:bool=True,time_constraint:bool=True):

  print(f'''Running {"Eval" if eval_ else "Training"}:
  No.of Epochs : {epochs}
  Change Midpoints : {str(change_mid)}\n\n''')

  if load: agent.LoadModel()
  array_of_states,array_of_scores,stability_list = [],[],[]

  for i in range(epochs):
      score = 0
      mid = np.random.randint(1,19)
      helipad_pos = [mid-1,mid+1]
      env.startEpisode()
      state = env.reset(helipad_pos) if change_mid else env.reset()
      done = False
      states = []

      while not done:
          action = agent.ChooseAction(state)
          state_,rew, done, _ = env.step(action,time_constraint)
          agent.ReplayBuffer(state, action, np.array([rew]), state_, np.array([done]))
          agent.learn()
          score += rew
          state = state_
          if render: env.render()
          states.append(state_)

      states = np.concatenate(states,axis=0).reshape(-1,8)
      array_of_states.append(states)
      array_of_scores.append(score)
      tmp = check_stability_local(states[-1])

      stability_list.append(tmp == 'STABLE')

      if (not eval_) and ((i == epochs-1) or (i%saveCycle == 0)): 
        agent.SaveModel()
        res = {
          'episode_wise_stability': stability_list,
          'all_states' : array_of_states,
          'all_scores' : array_of_scores,
        }
        save_logs(save_dir,res)
        array_of_states,array_of_scores,stability_list = [],[],[]

      print(f'...Iteration {i+1} over !!!!! Avg. Reward -> {np.array(score)[-50:].mean() if verbose else "" } stability : {tmp}')

  print(f"Done training {epochs} epochs")
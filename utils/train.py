import os
import numpy as np
import pickle as pkl

np.random.seed(0)

def save_histories(data,details:dict):
  
  agent_name = details['agent']
  env_name = details['env']
  history_type = details['type']

  path = os.path.abspath(details['path'])
  this_file = f"{agent_name}_{env_name}_{history_type}.pkl"
  path = os.path.join(path, this_file)

  if os.path.exists(path):
    with open(path,'rb') as saveFile:
      prev_data = pkl.load(saveFile)
      prev_data = np.concatenate((prev_data,np.array(data)),axis=0)

    with open(path,'wb') as saveFile:
      pkl.dump(prev_data,saveFile)

  else:
    with open(path,'wb') as saveFile:
      pkl.dump(np.array(data),saveFile)

  print(f"Done saving {this_file}....")


def train_model(agent,env,epochs:int,render:bool=True,load:bool=True,eval_:bool=False,saveCycle:int=25,change_mid:bool=False,verbose:bool=True):

    print(f'''Running {"Eval" if eval_ else "Training"}:
    No.of Epochs : {epochs}
    Change Midpoints : {str(change_mid)}\n\n''')

    if load: agent.LoadModel()
    score_history,state_history = [],[]
    # TODO add way to log and measure data for results while in eval mode

    for i in range(epochs):
        score = 0
        mid = np.random.randint(1,19)
        helipad_pos = [mid-1,mid+1]
        state = env.reset(helipad_pos) if change_mid else env.reset()
        done = False
        while not done:
            action = agent.ChooseAction(state)
            state_,rew, done, _ = env.step(action)
            agent.ReplayBuffer(state, action, np.array([rew]), state_, np.array([done]))
            agent.learn()
            score += rew
            state = state_
            if render: env.render()
        score_history.append(score)
        state_history.append(state)
        if (not eval_) and i%saveCycle == 0: agent.SaveModel()
        print(f'...Iteration {i+1} over !!!!! Reward -> {score if verbose else "" }')

    return score_history,state_history

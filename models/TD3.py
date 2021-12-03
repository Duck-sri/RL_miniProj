"""
Hyper-Parameter list
Critic Learning Rate : 1e-3
Actor Learning Rate : 1e-3
Optimizer : Adam
Target update rate (tau) : 5e-3
Batch size : 100
Discount Factor (gamma) : 0.99
Exploration Policy (Noise) : N(0, 0.1) - Normal Distributon
Noise Clip : 0.5
Policy frequency (For Delayed Update) : 2
"""
import copy
import os
import random
from collections import deque

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, lr, max_action, chkpt_dir):
        super().__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        # max_action defined to fit to any environment 
        # i.e if output is servo with rotation from 
        # -180 to 180 max_action = 180 so that the 
        # output from tanh [-1, 1] is scaled to the action range.
        self.max_action = max_action

        self.save_path = os.path.join(chkpt_dir, 'Actor_TD3')

        self.optimizer = T.optim.Adam(self.parameters(), lr = lr)
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'

        self.to(self.device)
    
    def forward(self, state):
        val = F.relu(self.l1(state))
        val = F.relu(self.l2(val))

        return self.max_action * T.tanh(self.l3(val))

    def SaveCheckpoint(self):
        print('...SAVING CHECKPOINT...')
        T.save(self.state_dict(), self.save_path)
    
    def LoadCheckpoint(self):
        print('...LAODING CHECKPOINT...')
        self.load_state_dict(T.load(self.save_path, map_location= T.device(self.device)))

class Critic(nn.Module):
    '''
    There are two critics q1, q2 involved in TD3
    '''
    def __init__(self, state_dim, action_dim, lr, chkpt_dir):
        super().__init__()                                      

        self.c1l1 = nn.Linear(state_dim + action_dim, 256)
        self.c1l2 = nn.Linear(256, 256)
        self.c1l3 = nn.Linear(256, 1)

        self.c2l1 = nn.Linear(state_dim + action_dim, 256)
        self.c2l2 = nn.Linear(256, 256)
        self.c2l3 = nn.Linear(256, 1)

        self.save_path = os.path.join(chkpt_dir, 'Critic_TD3')

        self.optimizer = T.optim.Adam(self.parameters(), lr = lr)
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'

        self.to(self.device)

    def forward(self, state, action):
        sa = T.cat([state, action], 1)

        q1 = F.relu(self.c1l1(sa))
        q1 = F.relu(self.c1l2(q1))
        q1 = self.c1l3(q1)

        q2 = F.relu(self.c2l1(sa))
        q2 = F.relu(self.c2l2(q2))
        q2 = self.c2l3(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = T.cat([state, action], 1)

        Q1 = F.relu(self.c1l1(sa))
        Q1 = F.relu(self.c1l2(Q1))
        Q1 = self.c1l3(Q1)

        return Q1
    
    def SaveCheckpoint(self):
        print('...Saving Checkpoint...')
        T.save(self.state_dict(), self.save_path)
    
    def LoadCheckpoint(self):
        print('...Loading Checkpoint')
        self.load_state_dict(T.load(self.save_path, map_location= T.device(self.device)))

class TD3_Agent():
    def __init__(self, state_dim, action_dim, max_action,chkpt_dir, gamma = 0.99, tau = 5e-3, noise_clip = 0.5, policy_delay = 2, batch_size = 100):
        '''
        chkpt_dir - Enter the directory where the model files are present.
        '''
        self.name = 'TD3'
        self.chkpt_dir = chkpt_dir
        self.actor = Actor(state_dim, action_dim, 1e-3, max_action, self.chkpt_dir)
        self.target_actor = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim, 1e-3, self.chkpt_dir)
        self.target_critic = copy.deepcopy(self.critic)

        self.max_action = max_action
        self.gamma = gamma
        self.noise = np.random.normal(0, 0.1, action_dim)
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.tau = tau
        
        self.batch_size = batch_size
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'
        self.memory = deque(maxlen=10000)

        self.mem_counter = 0

    def ChooseAction(self, state):
        self.actor.zero_grad()
        state = T.FloatTensor(state).to(self.device)
        a_t = self.actor(state).to(self.device) + T.Tensor(self.noise).to(self.device)
        self.actor.train()
        self.mem_counter += 1
        return a_t.cpu().detach().numpy()

    def ReplayBuffer(self, state, action, rew, state_, done):
        state = T.FloatTensor(state).to(self.device)
        action = T.as_tensor(action).to(self.device)
        state_ = T.FloatTensor(state_).to(self.device)
        rew = T.Tensor(rew).to(self.device)
        done = T.Tensor(1 - done).to(self.device)

        self.memory.append((state, action, rew, state_, done))
    
    def RetrieveBatch(self):
        batch = random.sample(self.memory, self.batch_size)
        try:
            state, action, rew, state_, done = map(T.stack, zip(*batch))
        except RuntimeError as e:
            print(e)
            for k, i in enumerate(batch):
                for j in i:
                    print(k, ':', j.shape)
        return state, action, rew, state_, done
    
    def learn(self):
        if self.mem_counter > self.batch_size:
            state, action, rew, state_, done = self.RetrieveBatch()
            with T.no_grad():
                noise_clipped = T.Tensor(self.noise).clamp(-self.noise_clip,self.noise_clip).to(self.device)
                action_ = self.target_actor(state_) + noise_clipped
                target_q1, target_q2 = self.critic(state_, action_)
                target_Q = T.min(target_q1, target_q2)
                y = rew + self.gamma * done * target_Q

            Q1, Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(Q1, y) + F.mse_loss(Q2, y)
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed Policy update

            if not self.mem_counter % self.policy_delay:
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean().to(self.device)
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                for param, target_param in zip(self.critic.parameters(),self.target_critic.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.actor.parameters(),self.target_actor.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def SaveModel(self):
        self.actor.SaveCheckpoint()
        self.target_actor.SaveCheckpoint()
        self.critic.SaveCheckpoint()
        self.target_critic.SaveCheckpoint()
        
    def LoadModel(self):
        self.actor.LoadCheckpoint()
        self.target_actor.LoadCheckpoint()
        self.critic.LoadCheckpoint()
        self.target_critic.LoadCheckpoint()

"""
1) REQUIRES A REPLAY BUFFER CLASS.
2) CLASS FOR TARGET Q - NETWORK CLASS.
3) BATCH NORM USED.
4) POLICY IS DETERMINISTIC, HOW TO HANDLE E-GREEDY? 
5) WE USE A STOCHASTIC OFF POLICY TO LEARN A DETERMINISTIC POLICY.
6) DETERMINISTIC POLICY MEANS OP OF NN IS ACTION RATHER THAN PROBABILITIES.
7) WE NEED TO RESTRICT THE ACTIONS TO THE BOUND SPECIFIED BY THE ENVIRONMENT.
8) WE HAVE 2 ACTOR AND 2 CRITIC NETWORKS, SIMILAR TO DOUBLE Q-LEARNING
9) UPDATES ARE SOFT I.E THETA_PRIME = TAU * THETA + (1 - THETA) * THETA_PRIME , TAU << 1
    TARGET NETWORK IS NOT DIRECTLY COPIED WITH THE WEIGHTS OF THE ONLINE NETWORK
    BUT MADE TO CONVERGE SLOWLY WITH THE ABOVE UPDATE EQUATION. 
10) TARGET_ACTOR = ONLINE_ACTOR + N
    N = OH_NOISE.
    HERE NOISE IS ADDED FOR EXPLORATION IN CONTINUOUS SPACES.     
"""

import os
from collections import deque
import torch as T
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

class OUActionNoise:
    '''
    OH_NOISE-PROCESS : https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    '''
    def __init__(self, mu, sigma = 0.15, theta = 0.2, dt = 1e-2, x0 = None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()
    
    def __call__(self):
        '''
        __call__() allows the object to be called as a Function.
        '''
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size = self.mu.shape)
        self.x_prev = x
        return x
    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class CriticNetwork(nn.Module):
    '''
    Critic Network class whose input is state and action taken while the output is a single value for 
    how good the action taken was.
    '''

    def __init__(self, beta, input_dim, fc1_dim, fc2_dim, n_actions, name, chkpt_dir):
        super().__init__()
        self.input_dim = input_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.n_actions = n_actions
        self.save_path = os.path.join(chkpt_dir, name + '_ddpg')

        self.fc1 = nn.Linear(*self.input_dim, self.fc1_dim)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.norm1 = nn.LayerNorm(self.fc1_dim)

        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.norm2 = nn.LayerNorm(self.fc2_dim)

        self.a_v = nn.Linear(self.n_actions, self.fc2_dim)

        self.q = nn.Linear(self.fc2_dim, 1)
        fq = 0.003
        nn.init.uniform_(self.q.weight.data, -fq, fq)
        nn.init.uniform_(self.q.bias.data, -fq, fq)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'

        self.to(self.device)

    def forward(self, state, action):
        state_value = F.relu(self.norm1(self.fc1(state)))

        state_value = self.norm2(self.fc2(state_value))

        action_value = F.relu(self.a_v(action))

        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def SaveCheckpoint(self):
        print('... SAVING CHECKPOINT ...')
        T.save(self.state_dict(), self.save_path)

    def LoadCheckpoint(self):
        print('... LOADING CHECKPOINT ...')
        self.load_state_dict(T.load(self.save_path,map_location=T.device(self.device)))


class ActorNetwork(nn.Module):
    '''
    Actor Network class whose input is state while the output is vector of action probabilities.
    '''

    def __init__(self, alpha, input_dim, fc1_dim, fc2_dim, n_actions, name, chkpt_dir):
        super().__init__()
        self.input_dim = input_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.n_actions = n_actions
        self.save_path = os.path.join(chkpt_dir, name + '_ddpg')

        self.fc1 = nn.Linear(*self.input_dim, self.fc1_dim)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.norm1 = nn.LayerNorm(self.fc1_dim)

        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.norm2 = nn.LayerNorm(self.fc2_dim)

        self.mu = nn.Linear(self.fc2_dim, self.n_actions)
        fmu = 0.003
        nn.init.uniform_(self.mu.weight.data, -fmu, fmu)
        nn.init.uniform_(self.mu.bias.data, -fmu, fmu)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'

        self.to(self.device)

    def forward(self, state):
        state_action_probs = F.relu(self.norm1(self.fc1(state)))
        state_action_probs = F.relu(self.norm2(self.fc2(state_action_probs)))
        state_action_probs = T.tanh(self.mu(state_action_probs))
        return state_action_probs

    def SaveCheckpoint(self):
        print('... SAVING CHECKPOINT ...')
        T.save(self.state_dict(), self.save_path)

    def LoadCheckpoint(self):
        print('... LOADING CHECKPOINT ...')
        self.load_state_dict(T.load(self.save_path, map_location = T.device(self.device)))

class Agent():
    def __init__(self, alpha, beta, tau, input_dim, chkpt_dir,gamma=0.99, n_actions = 2, fc1_dim=400, fc2_dim=300, batch_size=64,):
        self.name = 'DDPG'
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'

        self.memory = deque(maxlen=10000)

        self.Q = CriticNetwork(self.beta, input_dim, fc1_dim, fc2_dim, n_actions, 'Critic',chkpt_dir=chkpt_dir)
        self.Q_prime = CriticNetwork(self.beta, input_dim, fc1_dim, fc2_dim, n_actions, 'TargetCritic',chkpt_dir=chkpt_dir)
        self.mu = ActorNetwork(self.alpha, input_dim, fc1_dim, fc2_dim, n_actions, 'Actor',chkpt_dir=chkpt_dir)
        self.mu_prime = ActorNetwork(self.alpha, input_dim, fc1_dim, fc2_dim, n_actions, 'TargetActor',chkpt_dir=chkpt_dir)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        # The initial value of target is same as the online network
        self.UpdateNetworkParameters(tau = 1)

        self.mem_cntr = 0
    
    def ChooseAction(self, state):
        self.mu.zero_grad()
        state = T.FloatTensor(state).to(self.device)
        a_t = self.mu(state).to(self.device) + T.Tensor(self.noise()).to(self.device)
        self.mu.train()
        self.mem_cntr += 1
        return a_t.cpu().detach().numpy()
        
    def ReplayBuffer(self, state, action, rew, state_, done):
        state = T.FloatTensor(state).to(self.device)
        action = T.as_tensor(action).to(self.device)
        state_ = T.FloatTensor(state_).to(self.device)
        rew = T.Tensor(rew).to(self.device)
        done = T.Tensor(1 - done).to(self.device)
        #print(state.shape, action.shape, rew.shape, done.shape)
        self.memory.append((state, action, rew, state_, done))
    
    def RetrieveBatch(self):
        batch = random.sample(self.memory, self.batch_size)
        #state, action, rew, state_, done = map(T.stack, zip(*batch))
        #"""
        try:
          state, action, rew, state_, done = map(T.stack, zip(*batch))
        except RuntimeError as e:
          print(e)
          for k, i in enumerate(batch):
            for j in i:
              print(k, ':', j.shape)
        #"""
        return state, action, rew, state_, done

    def learn(self):
        if self.mem_cntr > self.batch_size:
          state, action, rew, state_, done = self.RetrieveBatch()
          
          self.Q.eval()
          self.Q_prime.eval()
          self.mu_prime.eval()

          target_action = self.mu_prime(state_)
          critic_value_prime = self.Q_prime(state_, target_action).to(self.device)
          
          y = []
          for i in range(self.batch_size):
              y.append(rew[i] + self.gamma * critic_value_prime[i] * done[i])
          
          y = T.Tensor(y).view(self.batch_size, 1).to(self.device)
          
          self.Q.train()
          self.Q.optimizer.zero_grad()
          critic_value = self.Q(state, action).to(self.device)
          critic_loss = F.mse_loss(y, critic_value)
          critic_loss.backward()
          self.Q.optimizer.step()
          self.Q.eval()

          self.mu.optimizer.zero_grad()
          a = self.mu(state).to(self.device)
          actor_loss = -self.Q(state, a)
          actor_loss = T.mean(actor_loss)
          actor_loss.backward()
          self.mu.optimizer.step()

          self.UpdateNetworkParameters()
    
    def UpdateNetworkParameters(self, tau = None):
        if tau is None:
            tau = self.tau

        actor_parameters = dict(self.mu.named_parameters())
        target_actor_parameters = dict(self.mu_prime.named_parameters())
        critic_parameters = dict(self.Q.named_parameters())
        target_critic_parameters = dict(self.Q_prime.named_parameters())

        for name in critic_parameters:
            target_critic_parameters[name] = tau * critic_parameters[name].clone() +\
                 (1 - tau) * target_critic_parameters[name].clone()

        self.Q_prime.load_state_dict(target_critic_parameters)

        for name in actor_parameters:
            target_actor_parameters[name] = tau * actor_parameters[name].clone() +\
                (1 - tau) * target_actor_parameters[name].clone()
        
        #self.VerifyUpload()

        self.mu_prime.load_state_dict(target_actor_parameters)
    
    def VerifyUpload(self):
        target_actor_params = self.mu_prime.named_parameters()
        target_critic_params = self.Q_prime.named_parameters()
        critic_state_dict = dict(target_critic_params)
        actor_state_dict = dict(target_actor_params)
        print('\nActor Networks\n')
        for name, param in self.mu.named_parameters():
            print(name, T.equal(param, actor_state_dict[name]))
        print('\nCritic Networks \n')
        for name, param in self.Q.named_parameters():
            print(name, T.equal(param, critic_state_dict[name]))
    
    def SaveModel(self):
        self.Q.SaveCheckpoint()
        self.Q_prime.SaveCheckpoint()
        self.mu.SaveCheckpoint()
        self.mu_prime.SaveCheckpoint()
        
    def LoadModel(self):
        self.Q.LoadCheckpoint()
        self.Q_prime.LoadCheckpoint()
        self.mu.LoadCheckpoint()
        self.mu_prime.LoadCheckpoint()

if __name__ == '__main__':
    agent = Agent()

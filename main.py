import os
import sys
from argparse import ArgumentParser

import models
from enviroments import make_env
from utils import get_state_details, loadConfig, plot_state_graph, train_model,init_save

config = loadConfig('./config.yml')
if config is None:
  print("No configurations")
  sys.exit(1)

if __name__ == '__main__':
  # while using colab,
  # remove this if case
  # put values in else case constants
  parser = ArgumentParser(description="Train TD3 and DDPG models to solve LunarLander environment from OpenAI gym")
  parser.add_argument('-e','--environment',metavar=f"[{' | '.join(config['environments'])}]",choices=config['environments'],required=True,help="Enviroment to work on")
  parser.add_argument('-a','--agent',metavar=f"[{' | '.join(config['agents'])}]",choices=config['agents'],required=True,help="Which Agent to use")
  parser.add_argument('--eval',action='store_true',help="Use when not training, jus expecting results")
  parser.add_argument('--state_graph',action='store_true',help="Plot State graph")
  parser.add_argument('--state_details',action='store_true',help="Get State details about state and enviroments")
  parser.add_argument('--epochs',metavar="EPOCHS",default=10,type=int,help='No.of Episodes to run')

  args = parser.parse_args()

  AGENT = args.agent
  ENV = args.environment
  EVAL = args.eval
  PLOT = args.state_graph
  PRINT_STATE = args.state_details
  EPOCHS = args.epochs
  RENDER = True

else :
  # put ur thing here 
  # or set while training in colab
  AGENT = 'td3'
  ENV = 'setpoint'
  EVAL = False
  PLOT = False
  PRINT_STATE = False
  EPOCHS = 10
  RENDER = False

checkpoint_dir = os.path.abspath(config['checkpoints_dir'])
checkpoint_dir = os.path.join(checkpoint_dir,AGENT,ENV)

agent_args = config['agent_config'][AGENT]

env = make_env(name=ENV)
if AGENT == 'td3':
  agent_args['state_dim'] = env.observation_space.shape[0]
  agent_args['action_dim'] = env.action_space.shape[0]
  agent_args['max_action'] = (env.action_space.high)[0]

agent_args['chkpt_dir'] = checkpoint_dir

agent = models.TD3(**agent_args) if AGENT=='td3' else models.DDPG(**agent_args)

save_dir = '_'.join([AGENT,ENV,'logs']) + '.pkl'
save_dir = os.path.join(config['results_dir'],save_dir)

init_save(config['results_dir'],force_clear=True)

if PLOT or PRINT_STATE:
  _ = plot_state_graph(agent,env,EPOCHS) if (PLOT) else get_state_details(env)

else:
  train_model(
    agent,
    env,
    EPOCHS,
    save_dir,
    render=RENDER,
    eval_=EVAL,
    saveCycle=5,
    change_mid=(ENV=='setpoint'),
    time_constraint=False
  )
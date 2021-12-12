import os
import sys
from argparse import ArgumentParser

import models
from enviroments import make_env
from utils import get_state_details, loadConfig, plot_state_graph, train_model

config = loadConfig('./config.yml')
if config is None:
  print("No configurations")
  sys.exit(1)

parser = ArgumentParser(description="Train TD3 and DDPG models to solve LunarLander environment from OpenAI gym")
parser.add_argument('-e','--environment',metavar=f"[{' | '.join(config['environments'])}]",choices=config['environments'],required=True,help="Enviroment to work on")
parser.add_argument('-a','--agent',metavar=f"[{' | '.join(config['agents'])}]",choices=config['agents'],required=True,help="Which Agent to use")
parser.add_argument('--eval',action='store_true',help="Use when not training, jus expecting results")
parser.add_argument('--state_graph',action='store_true',help="Plot State graph")
parser.add_argument('--state_details',action='store_true',help="Get State details about state and enviroments")
parser.add_argument('--epochs',metavar="EPOCHS",default=10,type=int,help='No.of Episodes to run')

args = parser.parse_args()


## for getting the directories straight
checkpoint_dir = os.path.abspath(config['checkpoints_dir'])
checkpoint_dir = os.path.join(checkpoint_dir,args.agent,args.environment)

agent_args = config['agent_config'][args.agent]

env = make_env(name=args.environment)
if args.agent == 'td3':
  agent_args['state_dim'] = env.observation_space.shape[0]
  agent_args['action_dim'] = env.action_space.shape[0]
  agent_args['max_action'] = (env.action_space.high)[0]

agent_args['chkpt_dir'] = checkpoint_dir

agent = models.TD3(**agent_args) if args.agent=='td3' else models.DDPG(**agent_args)
epochs = args.epochs

if args.state_graph or args.state_details:
  _ = plot_state_graph(agent,env,epochs) if (args.state_graph) else get_state_details(env)

else:
  score_history,state_history = train_model(agent,env,epochs,render=True,eval_=args.eval,change_mid=(args.environment=='setpoint'))

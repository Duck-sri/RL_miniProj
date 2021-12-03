import os

from enviroments import available_envs,make_env
import models
from utils import train_model,loadConfig

from argparse import ArgumentParser

config = loadConfig('./config.yml')

parser = ArgumentParser(description="Train TD3 and DDPG models to solve LunarLander environment from OpenAI gym")
parser.add_argument('-e','--environment',metavar=f"[{' | '.join(config['environments'])}]",choices=config['environments'],required=True,help="Enviroment to work on")
parser.add_argument('-a','--agent',metavar=f"[{' | '.join(config['agents'])}]",choices=config['agents'],required=True,help="Which Agent to use")
parser.add_argument('--eval',action='store_true',help="Use when not training, jus expecting results")
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
  agent_args['max_action'] = env.action_space.high[0]

agent_args['chkpt_dir'] = checkpoint_dir

agent = models.TD3(**agent_args) if args.agent=='td3' else models.DDPG(**agent_args)
epochs = args.epochs

score_history,state_history = train_model(agent,env,epochs,render=True,eval_=args.eval,change_mid=(args.environment=='setpoint'))
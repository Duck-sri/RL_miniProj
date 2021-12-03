# RL_miniProj
TD3 and DDPG for lunar lander

## Usage

Training for TD3 agent with Setpoint Tracking mode

```sh
$ pip install -r requirements.txt

$ python ./main.py -h # shows all the possible options

usage: main.py [-h] -e [disturbance | setpoint] -a [ddpg | td3] [--eval] [--epochs EPOCHS]

Train TD3 and DDPG models to solve LunarLander environment from OpenAI gym

optional arguments:
  -h, --help            show this help message and exit
  -e [disturbance | setpoint], --environment [disturbance | setpoint]
                        Enviroment to work on
  -a [ddpg | td3], --agent [ddpg | td3]
                        Which Agent to use
  --eval                Use when not training, jus expecting results
  --epochs EPOCHS       No.of Episodes to run, default 10
```

## Config

1. Edit the **[Config file](config.yml)** file to change checkpoints directory
```
This repo
├── ...
├── checkpoints <-- "CheckPoints dir in the config"
│   ├── ddpg
│   │   ├── disturbance
│   │   │   ├── Actor_ddpg
│   │   │   ├── Critic_ddpg
│   │   │   ├── TargetActor_ddpg
│   │   │   └── TargetCritic_ddpg
│   │   └── setpoint
│   │       ├── Actor_ddpg
│   │       ├── Critic_ddpg
│   │       ├── TargetActor_ddpg
│   │       ├── TargetCritic_ddpg
│   └── td3
│       ├── disturbance
│       │   ├── Actor_TD3
│       │   └── Critic_TD3
│       └── setpoint
│           ├── Actor_TD3
│           └── Critic_TD3
├── ...
```
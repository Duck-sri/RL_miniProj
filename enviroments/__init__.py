import gym
from .dynamic_setpoints import RamLander,GymLander

available_envs = ['disturbance','setpoint']

def make_env(name:str,*args,**kwargs):
  if name not in available_envs:
    raise NotImplemented(f"Env : {name} not implemented")

  return GymLander(*args,**kwargs) if (name == 'disturbance') else RamLander(*args,**kwargs)

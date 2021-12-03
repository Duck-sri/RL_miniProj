import yaml

def loadConfig(path:str):
  try:
    with open(path,'r') as configFile:
      config = yaml.load(configFile,Loader=yaml.Loader)
    
    return config
  except FileNotFoundError:
    print(f"No file found in {path}")
    return None
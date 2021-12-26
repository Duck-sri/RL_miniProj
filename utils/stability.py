import numpy as np
import matplotlib.pyplot as plt

# @TODO add way to measure the stability of the agent
# 1. Show that the values are converging  for a series of episodes after training to agrue for stability
# 2. 


def check_stability_local(final_state : np.ndarray, tolerance=0.2) -> str:
    return 'STABLE' if np.allclose(final_state[:-2], np.zeros_like(final_state[:-2]), atol = tolerance) else 'UNSTABLE'

def check_stability_global(states : np.ndarray):
    state_error = []
    for state in states:
        per_state_error =0
        for j in state:
            per_state_error += (j - 0)**2
        state_error.append(per_state_error)

    plt.plot(np.array(state_error))
    plt.title('error over the episode')
    plt.show()

if __name__ == '__main__': 
  stablers = []
  unstablers = []
  for x in np.linspace(0,0.5,10):
    for y in np.linspace(0,0.5,10):
      a = np.array([x, y, 0, 0, 0, 0])
      tmp = check_stability_local(a)
      if tmp == 'STABLE':
        stablers.append((x,y))
      else:
        unstablers.append((x,y))

      print(f"x: {x} y: {y} - res : {tmp}")

  print("Stables values...")
  print(stablers)
  print("\nUnstable values...")
  print(unstablers)

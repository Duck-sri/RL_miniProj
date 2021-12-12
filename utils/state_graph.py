import matplotlib.pyplot as plt
import numpy as np


def plot_state_graph(agent,env,epochs):
  agent.LoadModel()

  np.random.seed(0)

  for _ in range(epochs):
    done = False    
    score = 0
    state = env.reset()
    val = []
    while not done:
        action = agent.ChooseAction(state)
        state_, rew, done, _ = env.step(action)
        #print(state[0], state[1])
        val.append(state)
        score += rew
        state = state_
        env.render()
    val = np.array(val)
    fig1, axs = plt.subplots(4, 2)
  #print(dir(axs[0, 0]))

    fig2, axs2 = plt.subplots(1, 1)
    axs2.set_xlabel('X-CO-ORD')
    axs2.set_ylabel('Y-CO-ORD')
    axs2.plot(val[:,0],val[:,1])
    axs2.axis([-1, 1, 0, 1])
    axs2.grid()
    fig2.show()

    axs[0, 0].set_xlabel('TIME')
    axs[0, 0].set_ylabel('X-CO-ORD')
    axs[0, 0].plot(range(len(val)), val[:,0])
    axs[0, 0].grid()

    axs[0, 1].set_xlabel('TIME')
    axs[0, 1].set_ylabel('Y-CO-ORD')
    axs[0, 1].plot(range(len(val)), val[:,1])
    axs[0, 1].grid()

    axs[1, 0].set_xlabel('TIME')
    axs[1, 0].set_ylabel('X-VELOCITY')
    axs[1, 0].plot(range(len(val)), val[:,2])
    axs[1, 0].grid()

    axs[1, 1].set_xlabel('TIME')
    axs[1, 1].set_ylabel('Y-VELOCITY')
    axs[1, 1].plot(range(len(val)), val[:,3])
    axs[1, 1].grid()

    axs[2, 0].set_xlabel('TIME')
    axs[2, 0].set_ylabel('THETA')
    axs[2, 0].plot(range(len(val)), val[:,4])
    axs[2, 0].grid()

    axs[2, 1].set_xlabel('TIME')
    axs[2, 1].set_ylabel('THETA_DOT')
    axs[2, 1].plot(range(len(val)), val[:,5])
    axs[2, 1].grid()

    axs[3, 0].set_xlabel('TIME')
    axs[3, 0].set_ylabel('LEFT LEG')
    axs[3, 0].plot(range(len(val)), val[:,6])
    axs[3, 0].grid()

    axs[3, 1].set_xlabel('TIME')
    axs[3, 1].set_ylabel('RIGHT LEG')
    axs[3, 1].plot(range(len(val)), val[:,7])
    axs[3, 1].grid()

    fig1.show()
    plt.show()

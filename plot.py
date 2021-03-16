import numpy as np
import matplotlib.pyplot as plt
from utils import Saver, Plotter

saver = Saver()

Plotter.plot_svd(["models/pendulum_Q_svd.pck",
                  "models/cartpole_Q_svd.pck",
                  "models/mountaincar_Q_svd.pck"])

path_base = "models/pendulum_ql_action_bucket_{}_exp_{}.pck"

plt.figure()
parameters = np.zeros((10, 6))
rewards = np.zeros((10, 6))
for i in range(6):
    for j in range(10):
        model = saver.load_from_pickle(path_base.format(i, j))
        parameters[j, i] = np.prod(model.Q.shape)
        rewards[j, i] = np.mean(model.greedy_cumulative_reward[-100:])
plt.plot(np.mean(parameters, axis=0), np.max(rewards, axis=0))

path_base = "models/pendulum_ql_state_bucket_{}_exp_{}.pck"

parameters = np.zeros((10, 5))
rewards = np.zeros((10, 5))
for i in range(5):
    for j in range(10):
        model = saver.load_from_pickle(path_base.format(i, j))
        parameters[j, i] = np.prod(model.Q.shape)
        rewards[j, i] = np.mean(model.greedy_cumulative_reward[-100:])
plt.plot(np.mean(parameters, axis=0), np.max(rewards, axis=0))

path_base = "models/pendulum_lr_k_bucket_{}_exp_{}.pck"

parameters = np.zeros((10, 6))
rewards = np.zeros((10, 6))
for i in range(6):
    for j in range(10):
        model = saver.load_from_pickle(path_base.format(i, j))
        parameters[j, i] = np.prod(model.L.shape) + np.prod(model.R.shape)
        rewards[j, i] = np.mean(model.greedy_cumulative_reward[-100:])
plt.plot(np.mean(parameters, axis=0), np.max(rewards, axis=0))
plt.legend(["q_afix", "q_sfix", "lr"])
plt.show()
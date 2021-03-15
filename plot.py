import numpy as np
import matplotlib.pyplot as plt
from utils import Saver

saver = Saver()

Q_svd_pendulum = saver.load_from_pickle("models/pendulum_Q_svd.pck")
Q_svd_cartpole = saver.load_from_pickle("models/cartpole_Q_svd.pck")
Q_svd_mountaincar = saver.load_from_pickle("models/mountaincar_Q_svd.pck")

_, sigma_pendulum, _ = np.linalg.svd(Q_svd_pendulum.reshape(-1, Q_svd_pendulum.shape[-1]))
_, sigma_cartpole, _ = np.linalg.svd(Q_svd_cartpole.reshape(-1, Q_svd_cartpole.shape[-1]))
_, sigma_mountaincar, _ = np.linalg.svd(Q_svd_mountaincar.reshape(-1, Q_svd_mountaincar.shape[-1]))

fig, axes = plt.subplots(nrows=1, ncols=3)
axes[0].bar(np.arange(len(sigma_pendulum)), sigma_pendulum)
axes[1].bar(np.arange(len(sigma_cartpole)), sigma_cartpole)
axes[2].bar(np.arange(len(sigma_mountaincar)), sigma_mountaincar)
plt.show()

path_base = "models/pendulum_ql_action_bucket_{}_exp_{}.pck"

plt.figure()
parameters = np.zeros((10, 5))
rewards = np.zeros((10, 5))
for i in range(6):
    for j in range(10):
        model = saver.load_from_pickle(path_base.format(i, j))
        parameters[j, i] = np.prod(model.Q.shape)
        rewards[j, i] = np.mean(model.greedy_cumulative_reward[-100:])
plt.plot(np.mean(parameters, axis=0), np.mean(rewards, axis=0))

path_base = "models/pendulum_ql_state_bucket_{}_exp_{}.pck"

parameters = np.zeros((10, 5))
rewards = np.zeros((10, 5))
for i in range(5):
    for j in range(10):
        model = saver.load_from_pickle(path_base.format(i, j))
        parameters[j, i] = np.prod(model.Q.shape)
        rewards[j, i] = np.mean(model.greedy_cumulative_reward[-100:])
plt.plot(np.mean(parameters, axis=0), np.mean(rewards, axis=0))

path_base = "models/pendulum_lr_k_bucket_{}_exp_{}.pck"

parameters = np.zeros((10, 5))
rewards = np.zeros((10, 5))
for i in range(6):
    for j in range(10):
        model = saver.load_from_pickle(path_base.format(i, j))
        parameters[j, i] = np.prod(model.L.shape) + np.prod(model.R.shape)
        rewards[j, i] = np.mean(model.greedy_cumulative_reward[-100:])
plt.plot(np.mean(parameters, axis=0), np.mean(rewards, axis=0))
plt.legend(["q_afix", "q_sfix", "lr"])
plt.show()
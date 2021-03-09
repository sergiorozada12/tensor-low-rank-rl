import numpy as np
import matplotlib.pyplot as plt
from utils import Saver

saver = Saver()

Q_svd_pendulum = saver.load_from_pickle("models/pendulum_Q_svd.pck")
Q_svd_cartpole = saver.load_from_pickle("models/cartpole_Q_svd.pck")

_, sigma_pendulum, _ = np.linalg.svd(Q_svd_pendulum.reshape(-1, Q_svd_pendulum.shape[-1]))
_, sigma_cartpole, _ = np.linalg.svd(Q_svd_cartpole.reshape(-1, Q_svd_cartpole.shape[-1]))

fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].bar(np.arange(len(sigma_pendulum)), sigma_pendulum)
axes[1].bar(np.arange(len(sigma_cartpole)), sigma_cartpole)
plt.show()

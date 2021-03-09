import numpy as np
import matplotlib.pyplot as plt
from utils import Saver

saver = Saver()

Q = saver.load_from_pickle("models/pendulum_Q_svd.pck")

_, sigma, _ = np.linalg.svd(Q.reshape(-1, Q.shape[-1]))

plt.figure()
plt.bar(np.arange(len(sigma)), sigma)
plt.show()

#Q = saver.load_from_pickle("models/pendulum_Q_svd.pck")
Q = saver.load_from_pickle("models/cartpole_Q_svd.pck")

_, sigma, _ = np.linalg.svd(Q.reshape(-1, Q.shape[-1]))

plt.figure()
plt.bar(np.arange(len(sigma)), sigma)
plt.show()
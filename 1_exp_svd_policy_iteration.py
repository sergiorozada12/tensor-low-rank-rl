import gym
import gym_classics
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from src.utils.utils import OOMFormatter
from src.algorithms.policy_iteration import PolicyIteration, PolicyIterationClassic


env = gym.make('FrozenLake8x8-v1')
q = PolicyIteration(env=env, max_iter=10000, k=1000, gamma=0.95).run()
q -= np.mean(q)
_, sigma_frozenlake, _ = np.linalg.svd(q)

env = gym.make('Taxi-v3')
q = PolicyIteration(env=env, max_iter=10000, k=1000, gamma=0.9).run()
q -= np.mean(q)
_, sigma_taxi, _ = np.linalg.svd(q)

env = gym.make('Racetrack1-v0')
q = PolicyIterationClassic(env=env, max_iter=10000, k=1000, gamma=0.99).run()
q -= np.mean(q)
_, sigma_racetrack, _ = np.linalg.svd(q)

env = gym.make('JacksCarRental-v0')
q = PolicyIterationClassic(env=env, max_iter=10000, k=1000, gamma=0.99).run()
q -= np.mean(q)
_, sigma_rental, _ = np.linalg.svd(q)

with plt.style.context(['science'], ['ieee']):
    matplotlib.rcParams.update({'font.size': 18})

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[8, 7])
    axes = axes.flatten()

    axes[0].bar(np.arange(1, len(sigma_frozenlake) + 1), sigma_frozenlake)
    axes[0].set_xlabel("(a) SV index")
    axes[0].set_ylabel("$\sigma$")
    axes[0].set_ylim(0, 2.1)
    axes[0].set_yticks([0, 1, 2])
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    axes[1].bar(np.arange(1, len(sigma_racetrack) + 1), sigma_racetrack)
    axes[1].set_xlabel("(b) SV index")
    axes[1].set_ylabel("$\sigma$")
    axes[1].set_yticks([0, 200, 400])
    axes[1].set_xticks([2, 4, 6, 8])
    axes[1].yaxis.set_major_formatter(OOMFormatter(3, "%1.1f"))
    axes[1].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))

    axes[2].bar(np.arange(1, len(sigma_rental) + 1), sigma_rental)
    axes[2].set_xlabel("(c) SV index")
    axes[2].set_ylabel("$\sigma$")
    axes[2].set_yticks([0, 2500, 5000])
    axes[2].yaxis.set_major_formatter(OOMFormatter(4, "%1.1f"))
    axes[2].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))

    axes[3].bar(np.arange(1, len(sigma_taxi) + 1), sigma_taxi)
    axes[3].set_xlabel("(d) SV index")
    axes[3].set_ylabel("$\sigma$")
    axes[3].set_yticks([0, 500, 1000])
    axes[3].yaxis.set_major_formatter(OOMFormatter(3, "%1.1f"))
    axes[3].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
    axes[3].set_xticks([2, 4, 6])

    plt.tight_layout()

    fig.savefig('figures/fig_1.jpg', dpi=300)
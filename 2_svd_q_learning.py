import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from multiprocess import Pool

from src.environments.pendulum import CustomPendulumEnv
from src.environments.cartpole import CustomContinuousCartPoleEnv
from src.environments.mountaincar import CustomContinuous_MountainCarEnv
from src.environments.acrobot import CustomAcrobotEnv

from src.algorithms.q_learning import QLearning
from src.utils.utils import Discretizer


# Pendulum
env = CustomPendulumEnv()

discretizer = Discretizer(
    min_points_states=[-1, -5],
    max_points_states=[1, 5],
    bucket_states=[20, 20],
    min_points_actions=[-2],
    max_points_actions=[2],
    bucket_actions=[20]
)

learner = QLearning(
        env=env,
        discretizer=discretizer,
        episodes=15000,
        max_steps=100,
        epsilon=1.0,
        alpha=0.1,
        gamma=0.9,
        decay=0.99995,
        min_epsilon=0.0)

learner.train(run_greedy_frequency=10)

nS = np.prod(discretizer.n_states)
nA = np.prod(discretizer.n_actions)

q_matrix = learner.Q.reshape(nS, nA).copy()
q_matrix -= np.mean(q_matrix)
_, sigma_pendulum, _ = np.linalg.svd(q_matrix)

# Cartpole
env = CustomContinuousCartPoleEnv()

discretizer = Discretizer(
    min_points_states=[-4.8, -0.5, -0.42, -0.9],
    max_points_states=[4.8, 0.5, 0.42, 0.9],
    bucket_states=[10, 10, 10, 10],
    min_points_actions=[-1],
    max_points_actions=[1],
    bucket_actions=[20]
)

learner = QLearning(
        env=env,
        discretizer=discretizer,
        episodes=40000,
        max_steps=100,
        epsilon=0.4,
        alpha=0.4,
        gamma=0.9,
        decay=0.999999,
        min_epsilon=0.0
    )

learner.train(run_greedy_frequency=10)

nS = np.prod(discretizer.n_states)
nA = np.prod(discretizer.n_actions)

q_matrix = learner.Q.reshape(nS, nA).copy()
q_matrix -= np.mean(q_matrix)
_, sigma_cartpole, _ = np.linalg.svd(q_matrix)

# Mountaincar
env = CustomContinuous_MountainCarEnv()

discretizer = Discretizer(
    min_points_states=[-1.2, -0.07],
    max_points_states=[0.6, 0.07],
    bucket_states=[10, 20],
    min_points_actions=[-1],
    max_points_actions=[1],
    bucket_actions=[20]
)

learner = QLearning(
        env=env,
        discretizer=discretizer,
        episodes=5000,
        max_steps=2000,
        epsilon=1.0,
        alpha=0.1,
        gamma=0.99,
        decay=0.9999999,
        min_epsilon=0.0
    )

learner.train(run_greedy_frequency=10)

nS = np.prod(discretizer.n_states)
nA = np.prod(discretizer.n_actions)

q_matrix = learner.Q.reshape(nS, nA).copy()
q_matrix -= np.mean(q_matrix)
_, sigma_mountaincar, _ = np.linalg.svd(q_matrix)

# Acrobot
env = CustomAcrobotEnv()

discretizer = Discretizer(
    min_points_states=[-3.1416, -3.1416, -14, -30],
    max_points_states=[3.1416, 3.1416, 14, 30],
    bucket_states=[10, 10, 10, 10],
    min_points_actions=[-2],
    max_points_actions=[2],
    bucket_actions=[20]
)

learner = QLearning(
        env=env,
        discretizer=discretizer,
        episodes=10000,
        max_steps=1000,
        epsilon=1.0,
        alpha=0.1,
        gamma=0.9,
        decay=0.9999999,
        min_epsilon=0.0
    )

learner.train(run_greedy_frequency=10)

nS = np.prod(discretizer.n_states)
nA = np.prod(discretizer.n_actions)

q_matrix = learner.Q.reshape(nS, nA).copy()
q_matrix -= np.mean(q_matrix)
_, sigma_acrobot, _ = np.linalg.svd(q_matrix)

with plt.style.context(['science'], ['ieee']):
    matplotlib.rcParams.update({'font.size': 18})

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[8, 7])
    axes = axes.flatten()

    axes[0].bar(np.arange(1, len(sigma_pendulum) + 1), sigma_pendulum)
    axes[0].set_xlabel("(a)", labelpad=6)
    axes[0].set_ylabel("$\sigma$")
    axes[0].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))

    axes[1].bar(np.arange(1, len(sigma_cartpole) + 1), sigma_cartpole)
    axes[1].set_xlabel("(b)", labelpad=6)
    axes[1].set_ylabel("$\sigma$")
    axes[1].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))

    axes[2].bar(np.arange(1, len(sigma_mountaincar) + 1), sigma_mountaincar)
    axes[2].set_xlabel("(c)", labelpad=6)
    axes[2].set_ylabel("$\sigma$")
    axes[2].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))

    axes[3].bar(np.arange(1, len(sigma_acrobot) + 1), sigma_acrobot)
    axes[3].set_xlabel("(d)", labelpad=6)
    axes[3].set_ylabel("$\sigma$")
    axes[3].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))

    plt.tight_layout()

    fig.savefig('figures/fig_2.jpg', dpi=300)
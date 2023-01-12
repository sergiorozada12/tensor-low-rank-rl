import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorly
from tensorly.decomposition import parafac

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

q_matrix = learner.Q
q_matrix -= np.mean(q_matrix)

ranks_pendulum = list()
errors_pendulum = list()
for k in range(1, 100):
    if k == 20:
        continue
    weights, factors = parafac(q_matrix, rank=k)
    q_hat = tensorly.cp_to_tensor((weights, factors))
    error = np.linalg.norm(q_matrix.flatten() - q_hat.flatten(), 2)
    normalizer = np.linalg.norm(q_matrix.flatten(), 2)
    ranks_pendulum.append(k)
    errors_pendulum.append(100*error/normalizer)

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

q_matrix = learner.Q
q_matrix -= np.mean(q_matrix)

ranks_cartpole = list()
errors_cartpole = list()
for k in range(1, 1001, 50):
    print(k)
    if k == 10:
        continue
    weights, factors = parafac(q_matrix, rank=k)
    q_hat = tensorly.cp_to_tensor((weights, factors))
    error = np.linalg.norm(q_matrix.flatten() - q_hat.flatten(), 2)
    normalizer = np.linalg.norm(q_matrix.flatten(), 2)
    ranks_cartpole.append(k)
    errors_cartpole.append(100*error/normalizer)

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

q_matrix = learner.Q
q_matrix -= np.mean(q_matrix)

ranks_mountaincar = list()
errors_mountaincar = list()
for k in range(1, 100):
    print(k)
    if k == 20:
        continue
    weights, factors = parafac(q_matrix, rank=k)
    q_hat = tensorly.cp_to_tensor((weights, factors))
    error = np.linalg.norm(q_matrix.flatten() - q_hat.flatten(), 2)
    normalizer = np.linalg.norm(q_matrix.flatten(), 2)
    ranks_mountaincar.append(k)
    errors_mountaincar.append(100*error/normalizer)

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

q_matrix = learner.Q
q_matrix -= np.mean(q_matrix)

ranks_acrobot = list()
errors_acrobot = list()
for k in range(1, 1001, 50):
    print(k)
    if k == 10:
        continue
    weights, factors = parafac(q_matrix, rank=k)
    q_hat = tensorly.cp_to_tensor((weights, factors))
    error = np.linalg.norm(q_matrix.flatten() - q_hat.flatten(), 2)
    normalizer = np.linalg.norm(q_matrix.flatten(), 2)
    ranks_acrobot.append(k)
    errors_acrobot.append(100*error/normalizer)

with plt.style.context(['science'], ['ieee']):
    matplotlib.rcParams.update({'font.size': 18})

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[8, 7])
    axes = axes.flatten()

    axes[0].plot(ranks_pendulum, errors_pendulum)
    axes[0].set_xlabel("Rank", labelpad=6)
    axes[0].set_ylabel("(a) $\mathrm{NFE}$ (\%)")
    axes[0].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 100)
    axes[0].grid()

    axes[1].plot(ranks_cartpole, errors_cartpole)
    axes[1].set_xlabel("Rank", labelpad=6)
    axes[1].set_ylabel("(b) $\mathrm{NFE}$ (\%)")
    axes[1].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
    axes[1].set_xlim(0, 1000)
    axes[1].set_ylim(0, 100)
    axes[1].grid()

    axes[2].plot(ranks_mountaincar, errors_mountaincar)
    axes[2].set_xlabel("Rank", labelpad=6)
    axes[2].set_ylabel("(c) $\mathrm{NFE}$ (\%)")
    axes[2].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
    axes[2].set_xlim(0, 100)
    axes[2].set_ylim(0, 100)
    axes[2].grid()

    axes[3].plot(ranks_acrobot, errors_acrobot)
    axes[3].set_xlabel("Rank", labelpad=6)
    axes[3].set_ylabel("(d) $\mathrm{NFE}$ (\%)")
    axes[3].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
    axes[3].set_xlim(0, 1000)
    axes[3].set_ylim(0, 100)
    axes[3].grid()

    plt.tight_layout()

    fig.savefig('figures/fig_4.jpg', ddpi=300)

import os
import json

import matplotlib.pyplot as plt
import matplotlib

from src.environments.pendulum import CustomPendulumEnv
from src.environments.cartpole import CustomContinuousCartPoleEnv
from src.environments.mountaincar import CustomContinuous_MountainCarEnv
from src.environments.goddard import CustomGoddardEnv

from src.experiments.experiments import Experiment

env_pendulum = CustomPendulumEnv()
env_cartpole = CustomContinuousCartPoleEnv()
env_mountaincar = CustomContinuous_MountainCarEnv()
env_rocket = CustomGoddardEnv()


N_NODES = 100


if __name__ == "__main__":
    # Pendulum
    experiments = [f for f in os.listdir('parameters') if 'pendulum' in f]
    experiments_done = [f for f in os.listdir('results') if 'pendulum' in f]
    for name in experiments:
        if name in experiments_done:
            continue
        experiment = Experiment(name, env_pendulum, N_NODES)
        experiment.run_experiments(window=70)

    # Cartpole
    experiments = [f for f in os.listdir('parameters') if 'cartpole' in f]
    experiments_done = [f for f in os.listdir('results') if 'cartpole' in f]
    for name in experiments:
        if name in experiments_done:
            continue
        experiment = Experiment(name, env_cartpole, N_NODES)
        experiment.run_experiments(window=500)

    # Mountaincar
    experiments = [f for f in os.listdir('parameters') if 'mountaincar' in f]
    experiments_done = [f for f in os.listdir('results') if 'mountaincar' in f]
    for name in experiments:
        if name in experiments_done:
            continue
        experiment = Experiment(name, env_mountaincar, N_NODES)
        experiment.run_experiments(window=100)

    # Rocket
    experiments = [f for f in os.listdir('parameters') if 'rocket' in f]
    experiments_done = [f for f in os.listdir('results') if 'rocket' in f]
    for name in experiments:
        if name in experiments_done:
            continue
        experiment = Experiment(name, env_rocket, N_NODES)
        experiment.run_experiments(window=50)

    paths_pendulum = [f for f in os.listdir('results') if 'pendulum' in f]
    paths_cartpole = [f for f in os.listdir('results') if 'cartpole' in f]
    paths_mountaincar = [f for f in os.listdir('results') if 'mountaincar' in f]
    paths_rocket = [f for f in os.listdir('results') if 'rocket' in f]

    labels = [
        "Q-lear. low",
        "Q-lear. high",
        "MLR-lear.",
        "TLR-lear.",
        "DQN-lear. sm.",
        "DQN-lear. la."
    ]

    #with plt.style.context(['science'], ['ieee']):
    with plt.style.context(['ieee']):
        matplotlib.rcParams.update({'font.size': 18})

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 7))
        axes = axes.flatten()

        prefix = 'results/pendulum_'
        axes[0].plot(json.load(open(prefix + 'q_learning_low.json', 'r'))['steps'], color='b')
        axes[0].plot(json.load(open(prefix + 'q_learning_high.json', 'r'))['steps'], color='r')
        axes[0].plot(json.load(open(prefix + 'mlr_learning.json', 'r'))['steps'], color='g')
        axes[0].plot(json.load(open(prefix + 'tlr_learning.json', 'r'))['steps'], color='y')
        axes[0].plot(json.load(open(prefix + 'dqn_learning_small_sample.json', 'r'))['steps'], color='orange')
        axes[0].plot(json.load(open(prefix + 'dqn_learning_large_sample.json', 'r'))['steps'], color='k')
        axes[0].set_xlabel("Episodes", labelpad=4)
        axes[0].set_ylabel("(a) $\# Steps$")
        axes[0].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        axes[0].set_xlim(0, 5000)
        axes[0].legend(labels, fontsize=12)
        axes[0].grid()

        prefix = 'results/cartpole_'
        axes[1].plot(json.load(open(prefix + 'q_learning_low.json', 'r'))['steps'], color='b')
        axes[1].plot(json.load(open(prefix + 'q_learning_high.json', 'r'))['steps'], color='r')
        axes[1].plot(json.load(open(prefix + 'mlr_learning.json', 'r'))['steps'], color='g')
        axes[1].plot(json.load(open(prefix + 'tlr_learning.json', 'r'))['steps'], color='y')
        axes[1].plot(json.load(open(prefix + 'dqn_learning_large_sample.json', 'r'))['steps'], color='k')
        axes[1].set_xlabel("Episodes", labelpad=4)
        axes[1].set_ylabel("(a) $\# Steps$")
        axes[1].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        axes[1].set_xlim(0, 40000)
        axes[1].legend([l for l in labels if 'sm.' not in l], fontsize=12)
        axes[1].grid()

        prefix = 'results/mountaincar_'
        axes[2].plot(json.load(open(prefix + 'q_learning_low.json', 'r'))['steps'], color='b')
        axes[2].plot(json.load(open(prefix + 'q_learning_high.json', 'r'))['steps'], color='r')
        axes[2].plot(json.load(open(prefix + 'mlr_learning.json', 'r'))['steps'], color='g')
        axes[2].plot(json.load(open(prefix + 'tlr_learning.json', 'r'))['steps'], color='y')
        axes[2].plot(json.load(open(prefix + 'dqn_learning_small_sample.json', 'r'))['steps'], color='orange')
        axes[2].plot(json.load(open(prefix + 'dqn_learning_large_sample.json', 'r'))['steps'], color='k')
        axes[2].set_xlabel("Episodes", labelpad=4)
        axes[2].set_ylabel("(a) $\# Steps$")
        axes[2].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        axes[2].set_xlim(0, 5000)
        axes[2].legend(labels, fontsize=12)
        axes[2].grid()

        prefix = 'results/rocket_'
        axes[3].plot(json.load(open(prefix + 'q_learning_low.json', 'r'))['steps'], color='b')
        axes[3].plot(json.load(open(prefix + 'q_learning_high.json', 'r'))['steps'], color='r')
        axes[3].plot(json.load(open(prefix + 'mlr_learning.json', 'r'))['steps'], color='g')
        axes[3].plot(json.load(open(prefix + 'tlr_learning.json', 'r'))['steps'], color='y')
        axes[3].plot(json.load(open(prefix + 'dqn_learning_small_sample.json', 'r'))['steps'], color='orange')
        #axes[3].plot(json.load(open(prefix + 'dqn_learning_large_sample.json', 'r'))['steps'], color='k')
        axes[3].set_xlabel("Episodes", labelpad=4)
        axes[3].set_ylabel("(a) $\# Steps$")
        axes[3].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        axes[3].set_ylim(180, 500)
        axes[3].set_xlim(0, 500000)
        axes[3].legend(labels, fontsize=12)
        axes[3].grid()

        plt.tight_layout()
        fig.savefig('figures/fig_5.jpg', ddpi=300)


    with plt.style.context(['ieee']):
        matplotlib.rcParams.update({'font.size': 18})

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[8, 7])
        axes = axes.flatten()

        prefix = 'results/pendulum_'
        rewards = [
            json.load(open(prefix + 'q_learning_low.json', 'r'))['rewards'],
            json.load(open(prefix + 'q_learning_high.json', 'r'))['rewards'],
            json.load(open(prefix + 'mlr_learning.json', 'r'))['rewards'],
            json.load(open(prefix + 'tlr_learning.json', 'r'))['rewards'],
            json.load(open(prefix + 'dqn_learning_small_sample.json', 'r'))['rewards'],
            json.load(open(prefix + 'dqn_learning_large_sample.json', 'r'))['rewards'],
        ]
        axes[0].bar(labels, rewards, color='b')
        axes[0].set_ylabel("Cumm. Reward")
        axes[0].set_xticklabels(labels, rotation = 90, size=10)
        axes[0].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))

        prefix = 'results/cartpole_'
        rewards = [
            json.load(open(prefix + 'q_learning_low.json', 'r'))['rewards'],
            json.load(open(prefix + 'q_learning_high.json', 'r'))['rewards'],
            json.load(open(prefix + 'mlr_learning.json', 'r'))['rewards'],
            json.load(open(prefix + 'tlr_learning.json', 'r'))['rewards'],
            json.load(open(prefix + 'dqn_learning_large_sample.json', 'r'))['rewards'],
        ]
        axes[1].bar([l for l in labels if 'sm.' not in l], rewards, color='b')
        axes[1].set_ylabel("Cumm. Reward")
        axes[1].set_xticklabels([l for l in labels if 'sm.' not in l], rotation = 90, size=10)
        axes[1].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))

        prefix = 'results/mountaincar_'
        rewards = [
            json.load(open(prefix + 'q_learning_low.json', 'r'))['rewards'],
            json.load(open(prefix + 'q_learning_high.json', 'r'))['rewards'],
            json.load(open(prefix + 'mlr_learning.json', 'r'))['rewards'],
            json.load(open(prefix + 'tlr_learning.json', 'r'))['rewards'],
            json.load(open(prefix + 'dqn_learning_small_sample.json', 'r'))['rewards'],
            json.load(open(prefix + 'dqn_learning_large_sample.json', 'r'))['rewards'],
        ]
        axes[2].bar(labels, rewards, color='b')
        axes[2].set_ylabel("Cumm. Reward")
        axes[2].set_xticklabels(labels, rotation = 90, size=10)
        axes[2].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))

        prefix = 'results/rocket_'
        rewards = [
            json.load(open(prefix + 'q_learning_low.json', 'r'))['rewards'],
            json.load(open(prefix + 'q_learning_high.json', 'r'))['rewards'],
            json.load(open(prefix + 'mlr_learning.json', 'r'))['rewards'],
            json.load(open(prefix + 'tlr_learning.json', 'r'))['rewards'],
            json.load(open(prefix + 'dqn_learning_small_sample.json', 'r'))['rewards'],
            #json.load(open(prefix + 'dqn_learning_large_sample.json', 'r'))['rewards'],
        ]
        axes[3].bar([l for l in labels if 'la.' not in l], rewards, color='b')
        axes[3].set_ylabel("Cumm. Reward")
        axes[3].set_xticklabels([l for l in labels if 'la.' not in l], rotation = 90, size=10)
        axes[3].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))

        plt.tight_layout()

        fig.savefig('figures/fig_6.jpg', dpi=300)


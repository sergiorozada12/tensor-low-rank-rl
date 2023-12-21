import json

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from src.environments.highway import CustomHighwayEnv
from src.experiments.experiments import ExperimentHighway
from src.utils.utils import OOMFormatter


env_highway = CustomHighwayEnv()

N_NODES = 20
PATH_TLR_SMALL = 'highway_tlr_small.json'
PATH_TLR_LARGE = 'highway_tlr_large.json'
PATH_DQN_LARGE = 'highway_dqn_large.json'
PATH_DQN_SMALL = 'highway_dqn_small.json'


if __name__ == "__main__":
    experiment_tlr = ExperimentHighway(PATH_TLR_SMALL, env_highway, N_NODES)
    experiment_tlr.run_experiments(window=100)

    experiment_tlr = ExperimentHighway(PATH_TLR_LARGE, env_highway, N_NODES)
    experiment_tlr.run_experiments(window=100)

    experiment_dqn = ExperimentHighway(PATH_DQN_SMALL, env_highway, N_NODES)
    experiment_dqn.run_experiments(window=100)

    experiment_dqn = ExperimentHighway(PATH_DQN_LARGE, env_highway, N_NODES)
    experiment_dqn.run_experiments(window=100)

    rewards_dqn_large = json.load(open('results/highway_dqn_large.json', 'r'))
    rewards_dqn_small = json.load(open('results/highway_dqn_small.json', 'r'))
    rewards_tlr_large = json.load(open('results/highway_tlr_large.json', 'r'))
    rewards_tlr_medium = json.load(open('results/highway_tlr_medium.json', 'r'))
    rewards_tlr_small = json.load(open('results/highway_tlr_small.json', 'r'))

    rewards_dqn_large = pd.Series(rewards_dqn_large['rewards']).fillna(0)
    rewards_dqn_small = pd.Series(rewards_dqn_small['rewards']).fillna(0)
    rewards_tlr_large = pd.Series(rewards_tlr_large['rewards']).fillna(0)
    rewards_tlr_medium = pd.Series(rewards_tlr_medium['rewards']).fillna(0)
    rewards_tlr_small = pd.Series(rewards_tlr_small['rewards']).fillna(0)

    labels = [
        "DQN $23,210$ params. la.",
        "DQN $23,210$ params. sm.",
        "TLR $22,750$ params.",
        "TLR $3,700$ params.",
        "TLR $1,850$ params.",
    ]

    with plt.style.context(['science'], ['ieee']):
        matplotlib.rcParams.update({'font.size': 34})

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))
        axes = axes.flatten()

        steps = range(0, 100_000, 10)
        axes[0].plot(steps, rewards_dqn_large, color='b')
        axes[0].plot(steps, rewards_dqn_small, color='r')
        axes[0].plot(steps, rewards_tlr_large, color='orange')
        axes[0].plot(steps, rewards_tlr_medium, color='g')
        axes[0].plot(steps, rewards_tlr_small, color='k')
        axes[0].set_xlabel("Episodes", labelpad=4)
        axes[0].set_ylabel("(a) Return")
        axes[0].set_xlim(0, 100_000)
        axes[0].set_ylim(0, 45)
        axes[0].set_yticks([0, 15, 30, 45])
        axes[0].set_xticks([1000, 50000, 100000], [0, 50000, 100000])
        axes[0].yaxis.set_major_formatter(OOMFormatter(1, "%1.1f"))
        axes[0].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        axes[0].legend(labels, fontsize=20, loc='lower right')
        axes[0].grid()

        axes[1].plot(steps, rewards_dqn_large, color='b')
        axes[1].plot(steps, rewards_dqn_small, color='r')
        axes[1].plot(steps, rewards_tlr_large, color='orange')
        axes[1].plot(steps, rewards_tlr_medium, color='g')
        axes[1].plot(steps, rewards_tlr_small, color='k')
        axes[1].set_xlabel("Episodes", labelpad=4)
        axes[1].set_ylabel("(b) Return")
        axes[1].set_xlim(0, 100_000)
        axes[1].set_ylim(42, 45)
        axes[1].set_yticks([42, 43, 44, 45])
        axes[1].set_xticks([2000, 50000, 100000], [0, 50000, 100000])
        axes[1].yaxis.set_major_formatter(OOMFormatter(1, "%1.1f"))
        axes[1].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        axes[1].legend(labels, fontsize=20, loc='upper left')
        axes[1].grid()

        plt.tight_layout()
        fig.savefig('figures/fig_7.jpg', dpi=300)

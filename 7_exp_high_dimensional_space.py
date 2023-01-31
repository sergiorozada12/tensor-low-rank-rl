import json

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from src.environments.highway import CustomHighwayEnv
from src.experiments.experiments import ExperimentHighway


env_highway = CustomHighwayEnv()

N_NODES = 20
PATH_TLR_SMALL = 'highway_tlr_small.json'
PATH_TLR_LARGE = 'highway_tlr_large.json'
PATH_DQN_LARGE = 'highway_dqn_large.json'
PATH_DQN_SMALL = 'highway_dqn_small.json'


if __name__ == "__main__":
    """experiment_tlr = ExperimentHighway(PATH_TLR_SMALL, env_highway, N_NODES)
    experiment_tlr.run_experiments(window=100)

    experiment_tlr = ExperimentHighway(PATH_TLR_LARGE, env_highway, N_NODES)
    experiment_tlr.run_experiments(window=100)

    experiment_dqn = ExperimentHighway(PATH_DQN_SMALL, env_highway, N_NODES)
    experiment_dqn.run_experiments(window=100)

    experiment_dqn = ExperimentHighway(PATH_DQN_LARGE, env_highway, N_NODES)
    experiment_dqn.run_experiments(window=100)"""

    rewards_dqn_large = json.load(open('results/highway_dqn_large.json', 'r'))
    rewards_dqn_small = json.load(open('results/highway_dqn_small.json', 'r'))
    rewards_tlr_large = json.load(open('results/highway_tlr_large.json', 'r'))
    rewards_tlr_small = json.load(open('results/highway_tlr_small.json', 'r'))

    rewards_dqn_large = pd.Series(rewards_dqn_large['rewards']).fillna(0)
    rewards_dqn_small = pd.Series(rewards_dqn_small['rewards']).fillna(0)
    rewards_tlr_large = pd.Series(rewards_tlr_large['rewards']).fillna(0)
    rewards_tlr_small = pd.Series(rewards_tlr_small['rewards']).fillna(0)

    labels = [
        "DQN la.",
        "DQN sm.",
        "TLR $22,750$ params.",
        "TLR $4,550$ params.",
    ]

    with plt.style.context(['science'], ['ieee']):
        matplotlib.rcParams.update({'font.size': 24})

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 7))

        steps = range(0, 100_000, 10)
        axes.plot(steps, rewards_dqn_large, color='b')
        axes.plot(steps, rewards_dqn_small, color='r')
        axes.plot(steps, rewards_tlr_large, color='orange')
        axes.plot(steps, rewards_tlr_small, color='k')
        axes.set_xlabel("Episodes", labelpad=4)
        axes.set_ylabel("Return")
        axes.set_xlim(0, 100_000)
        axes.set_ylim(10, 45)
        axes.set_yticks([10, 20, 30, 40])
        axes.set_xticks([0, 25000, 50000, 75000, 100000])
        axes.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        axes.legend(labels, fontsize=20, loc='lower right')
        axes.grid()

        plt.tight_layout()
        fig.savefig('figures/fig_7.jpg', dpi=300)

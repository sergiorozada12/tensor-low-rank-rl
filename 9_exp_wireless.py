import random
import numpy as np
import torch
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt

from src.environments.wireless import WirelessCommunicationsEnv
from src.utils.utils import DiscretizerTorch, OOMFormatter
from src.algorithms.dqn_learning import DqnLearningTorch
from src.algorithms.tlr_learning import TensorLowRankLearningTorch


NUM_EXPS = 100
NUM_PROCS = 50

ENV = WirelessCommunicationsEnv(
    T=1_000,
    K=3,
    snr_max=10,
    snr_min=2,
    snr_autocorr=0.7,
    P_occ=np.array(
        [  
            [0.4, 0.6],
            [0.6, 0.4],
        ]
    ),
    occ_initial=[1, 1, 1],
    batt_harvest=1.0, 
    P_harvest=0.2, 
    batt_initial=5,
    batt_max_capacity=10,  
    batt_weight=1.0, 
    queue_initial=10,
    queue_arrival=5,
    queue_max_capacity=20,
    t_queue_arrival=10,
    queue_weight=0.2,
    loss_busy=0.8,  
)

DISCRETIZER = DiscretizerTorch(
    min_points_states=[0, 0, 0, 0, 0, 0, 0, 0],
    max_points_states=[20, 20, 10, 1, 1, 1, 20, 10],
    bucket_states=[10, 10, 10, 2, 2, 2, 10, 10],
    min_points_actions=[0, 0, 0],
    max_points_actions=[2, 2, 2],
    bucket_actions=[10, 10, 10],
)


def run_train_episode(env, agent, eps, eps_decay, H):
    s, _ = env.reset()
    for h in range(H):
        a = agent.select_action(s, h, eps)
        sp, r, d, _, _ = env.step(a)
        agent.buffer.append(h, s, a, sp, r, d)
        agent.update()

        if d:
            break

        s = sp
        eps *= eps_decay
    return eps

def run_test_episode(env, agent, H):
    G = 0
    s, _ = env.reset()
    for h in range(H):
        a = agent.select_greedy_action(s, h)
        s, r, d, _, _ = env.step(a)
        G += r

        if d:
            break
    return G

def run_experiment(
    n: int, E: int, H: int, eps: float, eps_decay: float, env, agent
):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)

    Gs = []
    for epoch in range(E):
        eps = run_train_episode(env, agent, eps, eps_decay, H)
        G = run_test_episode(env, agent, H)
        Gs.append(G)
        print(f'{epoch}/{E}: {np.around(G, 2)} - Epsilon: {eps} \r', end='')
    return Gs

def run_experiment_dqn_1(n):
    agent = DqnLearningTorch(
        DISCRETIZER,
        alpha=0.001,
        gamma=0.99,
        buffer_size=1_000,
        batch_size=1
    )

    G = run_experiment(
        n=n,
        E=1_000,
        H=1_000,
        eps=1.0,
        eps_decay=0.99999,
        env=ENV,
        agent=agent
    )

    return G

def run_experiment_dqn_16(n):
    agent = DqnLearningTorch(
        DISCRETIZER,
        alpha=0.001,
        gamma=0.99,
        buffer_size=1_000,
        batch_size=16
    )

    G = run_experiment(
        n=n,
        E=1_000,
        H=1_000,
        eps=1.0,
        eps_decay=0.99999,
        env=ENV,
        agent=agent
    )

    return G

def run_experiment_tlr_20(n):
    agent = TensorLowRankLearningTorch(
        DISCRETIZER,
        alpha=0.001,
        gamma=0.99, 
        k=20,
        scale=0.1,
        bias=-1.0,
        method=0
    )

    G = run_experiment(
        n=n,
        E=1_000,
        H=1_000,
        eps=1.0,
        eps_decay=0.99999,
        env=ENV,
        agent=agent
    )

    return G

def run_experiment_tlr_40(n):
    agent = TensorLowRankLearningTorch(
        DISCRETIZER,
        alpha=0.001,
        gamma=0.99, 
        k=40,
        scale=0.1,
        bias=-1.0,
        method=0
    )

    G = run_experiment(
        n=n,
        E=1_000,
        H=1_000,
        eps=1.0,
        eps_decay=0.99999,
        env=ENV,
        agent=agent
    )

    return G


if __name__ == "__main__":

    with multiprocessing.Pool(processes=NUM_PROCS) as pool:
        res = pool.map(run_experiment_tlr_20, range(NUM_EXPS))
    np.save('results/wireless_tlr_20.npy', res)

    with multiprocessing.Pool(processes=NUM_PROCS) as pool:
        res = pool.map(run_experiment_tlr_40, range(NUM_EXPS))
    np.save('results/wireless_tlr_40.npy', res)

    with multiprocessing.Pool(processes=NUM_PROCS) as pool:
        res = pool.map(run_experiment_dqn_1, range(NUM_EXPS))
    np.save('results/wireless_dqn_1.npy', res)

    with multiprocessing.Pool(processes=NUM_PROCS) as pool:
        res = pool.map(run_experiment_dqn_16, range(NUM_EXPS))
    np.save('results/wireless_dqn_16.npy', res)

    G_dqn_1 = np.load('results/wireless_dqn_1.npy')
    G_dqn_16 = np.load('results/wireless_dqn_16.npy')
    G_tlr_20 = np.load('results/wireless_tlr_20.npy')
    G_tlr_40 = np.load('results/wireless_tlr_40.npy')

    m_dqn_1 = np.median(G_dqn_1, axis=0)
    m_dqn_16 = np.median(G_dqn_16, axis=0)
    m_tlr_20 = np.median(G_tlr_20, axis=0)
    m_tlr_40 = np.median(G_tlr_40, axis=0)

    with plt.style.context(['science'], ['ieee']):
        matplotlib.rcParams.update({'font.size': 20})
        fig = plt.figure(figsize=[8, 4])
        plt.plot(m_dqn_1, c='b', label='DQN 131,152 params. sm.')
        plt.plot(m_dqn_16, c='r', label='DQN 131,152 params. la.')
        plt.plot(m_tlr_20, c='g', label='TLR 1,720 params.')
        plt.plot(m_tlr_40, c='y', label='TLR 3,440 params.')

        plt.gca().yaxis.set_major_formatter(OOMFormatter(3, "%1.1f"))
        plt.gca().ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))

        plt.yticks([-3_000, -1_000, 1_000])
        plt.xticks([0, 250, 500, 750, 1_000])

        plt.xlim(0, 1_000)
        plt.legend(loc="lower right", fontsize=12)

        plt.xlabel("Episodes")
        plt.ylabel("Return")

        plt.grid()
        plt.tight_layout()
        fig.savefig('figures/fig_8.jpg', dpi=600)

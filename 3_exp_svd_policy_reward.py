from multiprocess import Pool

import gym
import gym_classics
import numpy as np
import matplotlib.pyplot as plt

from src.algorithms.policy_iteration import PolicyIteration, PolicyIterationClassic
from src.algorithms.q_learning import QLearning
from src.utils.utils import Discretizer

from src.environments.pendulum import CustomPendulumEnv
from src.environments.cartpole import CustomContinuousCartPoleEnv
from src.environments.mountaincar import CustomContinuous_MountainCarEnv

def run_test_reward_pi(env, q, max_steps=10000):
    state = env.reset()

    if not isinstance(state, int):
        state = state[0]

    cum_r = 0
    for t in range(max_steps):
        action = np.argmax(q[state, :])
        state, reward, done = env.step(action)[:3]
        cum_r += reward
        if done:
            break
    return cum_r

def run_test_reward_rl(env, q, discretizer, max_steps=10000):
    state = env.reset()

    state_idx = discretizer.get_state_index(state)
    cum_r = 0
    for t in range(max_steps): 
        action_idx = np.argmax(q[state_idx + (slice(None),)])
        action = discretizer.get_action_from_index(action_idx)
        state, reward, done, _ = env.step(action)
        state_idx = discretizer.get_state_index(state)
        cum_r += reward
        if done:
            break
    return cum_r

def run_experiment(learner):
    learner.train(run_greedy_frequency=10)
    return learner

NODES = 50

rank = 1
env = gym.make('FrozenLake8x8-v1')
learners = [PolicyIteration(env=env, max_iter=10000, k=1000, gamma=0.95) for _ in range(NODES)]
with Pool(NODES) as pool:
    learners_trained = pool.map(lambda learner: learner.run(), learners)
learners_trained = [learner - np.mean(learner) for learner in learners_trained]
reward_frozenlake = np.median([run_test_reward_pi(env, learner) for learner in learners_trained])
learners_decomposed = [np.linalg.svd(learner) for learner in learners_trained]
learners_estimated = [u[:, :rank]@np.diag(sigma)[:rank, :rank]@v[:rank, :] for u, sigma, v in learners_decomposed]
reward_frozenlake_estimated = np.median([run_test_reward_pi(env, learner) for learner in learners_estimated])

rank = 5
env = gym.make('Racetrack1-v0')
learners = [PolicyIterationClassic(env=env, max_iter=10000, k=1000, gamma=0.99) for _ in range(NODES)]
with Pool(NODES) as pool:
    learners_trained = pool.map(lambda learner: learner.run(), learners)
learners_trained = [learner - np.mean(learner) for learner in learners_trained]
reward_racetrack = np.median([run_test_reward_pi(env, learner) for learner in learners_trained])
learners_decomposed = [np.linalg.svd(learner) for learner in learners_trained]
learners_estimated = [u[:, :rank]@np.diag(sigma)[:rank, :rank]@v[:rank, :] for u, sigma, v in learners_decomposed]
reward_racetrack_estimated = np.median([run_test_reward_pi(env, learner) for learner in learners_estimated])

rank = 3
env = gym.make('JacksCarRental-v0')
learners = [PolicyIterationClassic(env=env, max_iter=10000, k=1000, gamma=0.99) for _ in range(NODES)]
with Pool(NODES) as pool:
    learners_trained = pool.map(lambda learner: learner.run(), learners)
learners_trained = [learner - np.mean(learner) for learner in learners_trained]
reward_rentalcar = np.median([run_test_reward_pi(env, learner) for learner in learners_trained])
learners_decomposed = [np.linalg.svd(learner) for learner in learners_trained]
learners_estimated = [u[:, :rank]@np.diag(sigma)[:rank, :rank]@v[:rank, :] for u, sigma, v in learners_decomposed]
reward_rentalcar_estimated = np.median([run_test_reward_pi(env, learner) for learner in learners_estimated])

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

learners = [QLearning(
        env=env,
        discretizer=discretizer,
        episodes=15000,
        max_steps=100,
        epsilon=1.0,
        alpha=0.1,
        gamma=0.9,
        decay=0.99995,
        min_epsilon=0.0
    ) for _ in range(NODES)]

with Pool(NODES) as pool:
    learners_trained = pool.map(run_experiment, learners)

nS = np.prod(discretizer.n_states)
nA = np.prod(discretizer.n_actions)
nD = discretizer.n_states.tolist() + discretizer.n_actions.tolist()

q_matrices = [q_learner.Q.reshape(nS, nA).copy() for q_learner in learners_trained]
reward_pendulum = np.median([run_test_reward_rl(env, q.reshape(nD), discretizer, 100) for q in q_matrices])
q_matrices = [q - np.mean(q) for q in q_matrices]

rank = 10
q_decomposed = [np.linalg.svd(q) for q in q_matrices]
q_estimated = [u[:, :rank]@np.diag(sigma)[:rank, :rank]@v[:rank, :] for u, sigma, v in q_decomposed]
reward_pendulum_estimated = np.median([run_test_reward_rl(env, q.reshape(nD), discretizer, 100) for q in q_estimated])

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

learners = [QLearning(
        env=env,
        discretizer=discretizer,
        episodes=40000,
        max_steps=100,
        epsilon=0.4,
        alpha=0.4,
        gamma=0.9,
        decay=0.999999,
        min_epsilon=0.0
    ) for _ in range(NODES)]

with Pool(NODES) as pool:
    learners_trained = pool.map(run_experiment, learners)

nS = np.prod(discretizer.n_states)
nA = np.prod(discretizer.n_actions)
nD = discretizer.n_states.tolist() + discretizer.n_actions.tolist()

q_matrices = [q_learner.Q.reshape(nS, nA).copy() for q_learner in learners_trained]
reward_cartpole = np.median([run_test_reward_rl(env, q.reshape(nD), discretizer, 100) for q in q_matrices])
q_matrices = [q - np.mean(q) for q in q_matrices]

rank = 3
q_decomposed = [np.linalg.svd(q) for q in q_matrices]
q_estimated = [u[:, :rank]@np.diag(sigma)[:rank, :rank]@v[:rank, :] for u, sigma, v in q_decomposed]
reward_cartpole_estimated = np.median([run_test_reward_rl(env, q.reshape(nD), discretizer, 100) for q in q_estimated])

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

learners = [QLearning(
        env=env,
        discretizer=discretizer,
        episodes=20000,
        max_steps=2000,
        epsilon=1.0,
        alpha=0.1,
        gamma=0.99,
        decay=0.9999999,
        min_epsilon=0.0
    ) for _ in range(NODES)]

with Pool(NODES) as pool:
    learners_trained = pool.map(run_experiment, learners)

nS = np.prod(discretizer.n_states)
nA = np.prod(discretizer.n_actions)
nD = discretizer.n_states.tolist() + discretizer.n_actions.tolist()

q_matrices = [q_learner.Q.reshape(nS, nA).copy() for q_learner in learners_trained]
reward_mountain = np.median([run_test_reward_rl(env, q.reshape(nD), discretizer, 2000) for q in q_matrices])
q_matrices = [q - np.mean(q) for q in q_matrices]

rank = 12
q_decomposed = [np.linalg.svd(q) for q in q_matrices]
q_estimated = [u[:, :rank]@np.diag(sigma)[:rank, :rank]@v[:rank, :] for u, sigma, v in q_decomposed]
reward_mountain_estimated = np.median([run_test_reward_rl(env, q.reshape(nD), discretizer, 2000) for q in q_estimated])
raw = np.array([
    reward_frozenlake,
    reward_racetrack,
    reward_rentalcar,
    reward_pendulum,
    reward_cartpole,
    reward_mountain
    ])

truncated = np.array([
    reward_frozenlake_estimated,
    reward_racetrack_estimated,
    reward_rentalcar_estimated,
    reward_pendulum_estimated,
    reward_cartpole_estimated,
    reward_mountain_estimated
    ])

labels = ["Frozenlake", "Racetrack", "Rentalcar", "Pendulum", "Cartpole", "Mountaincar"]
error = np.abs(100*(raw - truncated)/raw)
with plt.style.context(['science'], ['ieee']):
    fig, ax = plt.subplots()
    ax.bar(labels, error)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("$\mathrm{NCRE}$ (\%)")
    fig.savefig('figures/fig_3.jpg', dpi=300)

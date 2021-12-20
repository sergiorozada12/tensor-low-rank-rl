from multiprocess import Pool

import numpy as np
import pandas as pd

from src.algorithms.q_learning import QLearning
from src.algorithms.mlr_learning import MatrixLowRankLearning
from src.algorithms.tlr_learning import TensorLowRankLearning

from src.environments.pendulum import CustomPendulumEnv
from src.environments.cartpole import CustomContinuousCartPoleEnv
from src.environments.mountaincar import CustomContinuous_MountainCarEnv
from src.environments.goddard import CustomGoddardEnv

from src.utils.utils import Discretizer

NODES = 1

def run_experiment(learner):
    learner.train(run_greedy_frequency=10)
    return learner

"""
# Pendulum
env = CustomPendulumEnv()

discretizer_low = Discretizer(
    min_points_states=[-1, -5],
    max_points_states=[1, 5],
    bucket_states=[20, 20],
    min_points_actions=[-2],
    max_points_actions=[2],
    bucket_actions=[2]
)

discretizer_high = Discretizer(
    min_points_states=[-1, -5],
    max_points_states=[1, 5],
    bucket_states=[20, 20],
    min_points_actions=[-2],
    max_points_actions=[2],
    bucket_actions=[20]
)

q_learners_low = [QLearning(
    env=env,
    discretizer=discretizer_low,
    episodes=5000,
    max_steps=100,
    epsilon=1.0,
    alpha=0.1,
    gamma=0.9,
    decay=0.999999,
    min_epsilon=0.0
) for _ in range(NODES)]

q_learners_high = [QLearning(
    env=env,
    discretizer=discretizer_high,
    episodes=5000,
    max_steps=100,
    epsilon=1.0,
    alpha=0.1,
    gamma=0.9,
    decay=0.999999,
    min_epsilon=0.0
) for _ in range(NODES)]

mlr_learners = [
    MatrixLowRankLearning(
        env=env,
        discretizer=discretizer_high,
        episodes=5000,
        max_steps=100,
        epsilon=1.0,
        alpha=0.01,
        gamma=0.9,
        decay=0.999999,
        min_epsilon=0.0,
        init_ord=1.0,
        k=4
    ) for _ in range(NODES)]

tlr_learners = [
    TensorLowRankLearning(
        env=env,
        discretizer=discretizer_high,
        episodes=5000,
        max_steps=100,
        epsilon=1.0,
        alpha=0.005,
        gamma=0.9,
        decay=0.999999,
        min_epsilon=0.0,
        init_ord=1.0,
        k=4
    ) for _ in range(NODES)]

with Pool(NODES) as pool:
    q_learners_low_trained = pool.map(run_experiment, q_learners_low)

with Pool(NODES) as pool:
    q_learners_high_trained = pool.map(run_experiment, q_learners_high)

with Pool(NODES) as pool:
    mlr_learners_trained = pool.map(run_experiment, mlr_learners)

with Pool(NODES) as pool:
    tlr_learners_trained = pool.map(run_experiment, tlr_learners)

N = 70

pend_steps_q_low = np.median([pd.Series(learner.greedy_steps).rolling(N).median() for learner in q_learners_low_trained], axis=0)
pend_steps_q_high = np.median([pd.Series(learner.greedy_steps).rolling(N).median() for learner in q_learners_high_trained], axis=0)
pend_steps_mlr = np.median([pd.Series(learner.greedy_steps).rolling(N).median() for learner in mlr_learners_trained], axis=0)
pend_steps_tlr = np.median([pd.Series(learner.greedy_steps).rolling(N).median() for learner in tlr_learners_trained], axis=0)

pend_reward_q_low = np.median([np.mean(learner.greedy_cumulative_reward[-10:]) for learner in q_learners_low_trained])
pend_reward_q_high = np.median([np.mean(learner.greedy_cumulative_reward[-10:]) for learner in q_learners_high_trained])
pend_reward_mlr = np.median([np.mean(learner.greedy_cumulative_reward[-10:]) for learner in mlr_learners_trained])
pend_reward_tlr = np.median([np.mean(learner.greedy_cumulative_reward[-10:]) for learner in tlr_learners_trained])

# Cartpole
env = CustomContinuousCartPoleEnv()

discretizer_low = Discretizer(
    min_points_states=[-4.8, -0.5, -0.42, -0.9],
    max_points_states=[4.8, 0.5, 0.42, 0.9],
    bucket_states=[10, 10, 20, 20],
    min_points_actions=[-1],
    max_points_actions=[1],
    bucket_actions=[4]
)

discretizer_high = Discretizer(
    min_points_states=[-4.8, -0.5, -0.42, -0.9],
    max_points_states=[4.8, 0.5, 0.42, 0.9],
    bucket_states=[10, 10, 20, 20],
    min_points_actions=[-1],
    max_points_actions=[1],
    bucket_actions=[10]
)

q_learners_low = [QLearning(
    env=env,
    discretizer=discretizer_low,
    episodes=40000,
    max_steps=100,
    epsilon=0.4,
    alpha=0.1,
    gamma=0.9,
    decay=0.999999,
    min_epsilon=0.0
) for _ in range(NODES)]

q_learners_high = [QLearning(
    env=env,
    discretizer=discretizer_high,
    episodes=40000,
    max_steps=100,
    epsilon=0.4,
    alpha=0.1,
    gamma=0.9,
    decay=0.999999,
    min_epsilon=0.0
) for _ in range(NODES)]

mlr_learners = [MatrixLowRankLearning(
    env=env,
    discretizer=discretizer_high,
    episodes=40000,
    max_steps=100,
    epsilon=0.4,
    alpha=0.01,
    gamma=0.9,
    decay=0.999999,
    min_epsilon=0.0,
    init_ord=1.0,
    k=4
) for _ in range(NODES)]

tlr_learners = [TensorLowRankLearning(
    env=env,
    discretizer=discretizer_high,
    episodes=40000,
    max_steps=100,
    epsilon=0.4,
    alpha=0.001,
    gamma=0.9,
    decay=0.999999,
    min_epsilon=0.0,
    init_ord=1.0,
    k=10
) for _ in range(NODES)]

with Pool(NODES) as pool:
    q_learners_low_trained = pool.map(run_experiment, q_learners_low)

with Pool(NODES) as pool:
    q_learners_high_trained = pool.map(run_experiment, q_learners_high)

with Pool(NODES) as pool:
    mlr_learners_trained = pool.map(run_experiment, mlr_learners)

with Pool(NODES) as pool:
    tlr_learners_trained = pool.map(run_experiment, tlr_learners)

N = 500

cart_steps_q_low = np.median([pd.Series(learner.greedy_steps).rolling(N).median() for learner in q_learners_low_trained], axis=0)
cart_steps_q_high = np.median([pd.Series(learner.greedy_steps).rolling(N).median() for learner in q_learners_high_trained], axis=0)
cart_steps_mlr = np.median([pd.Series(learner.greedy_steps).rolling(N).median() for learner in mlr_learners_trained], axis=0)
cart_steps_tlr = np.median([pd.Series(learner.greedy_steps).rolling(N).median() for learner in tlr_learners_trained], axis=0)

cart_reward_q_low = np.median([np.mean(learner.greedy_cumulative_reward[-10:]) for learner in q_learners_low_trained])
cart_reward_q_high = np.median([np.mean(learner.greedy_cumulative_reward[-10:]) for learner in q_learners_high_trained])
cart_reward_mlr = np.median([np.mean(learner.greedy_cumulative_reward[-10:]) for learner in mlr_learners_trained])
cart_reward_tlr = np.median([np.mean(learner.greedy_cumulative_reward[-10:]) for learner in tlr_learners_trained])

print(cart_reward_q_low, cart_reward_q_high, cart_reward_mlr, cart_reward_tlr)
print(cart_steps_q_low.shape)"""


# MountainCar
env = CustomContinuous_MountainCarEnv()

discretizer_low = Discretizer(
    min_points_states=[-1.2, -0.07],
    max_points_states=[0.6, 0.07],
    bucket_states=[10, 20],
    min_points_actions=[-1.0],
    max_points_actions=[1.0],
    bucket_actions=[2]
)

discretizer_high = Discretizer(
    min_points_states=[-1.2, -0.07],
    max_points_states=[0.6, 0.07],
    bucket_states=[10, 20],
    min_points_actions=[-1.0],
    max_points_actions=[1.0],
    bucket_actions=[10]
)

q_learners_low = [QLearning(
    env=env,
    discretizer=discretizer_low,
    episodes=10000,
    max_steps=2000,
    epsilon=1.0,
    alpha=0.1,
    gamma=0.99,
    decay=0.999999,
    min_epsilon=0.0
) for _ in range(NODES)]

q_learners_high = [QLearning(
    env=env,
    discretizer=discretizer_high,
    episodes=10000,
    max_steps=2000,
    epsilon=1.0,
    alpha=0.1,
    gamma=0.99,
    decay=0.999999,
    min_epsilon=0.0
) for _ in range(NODES)]

mlr_learners = [MatrixLowRankLearning(
    env=env,
    discretizer=discretizer_high,
    episodes=10000,
    max_steps=2000,
    epsilon=1.0,
    alpha=0.001,
    gamma=0.99,
    decay=0.999999,
    min_epsilon=0.0,
    init_ord=1.0,
    k=10
) for _ in range(NODES)]

tlr_learners = [TensorLowRankLearning(
    env=env,
    discretizer=discretizer_high,
    episodes=10000,
    max_steps=2000,
    epsilon=1.0,
    alpha=0.001,
    gamma=0.99,
    decay=0.999999,
    min_epsilon=0.0,
    init_ord=1.0,
    k=10
) for _ in range(NODES)]

with Pool(NODES) as pool:
    q_learners_low_trained = pool.map(run_experiment, q_learners_low)

with Pool(NODES) as pool:
    q_learners_high_trained = pool.map(run_experiment, q_learners_high)

with Pool(NODES) as pool:
    mlr_learners_trained = pool.map(run_experiment, mlr_learners)

with Pool(NODES) as pool:
    tlr_learners_trained = pool.map(run_experiment, tlr_learners)

N = 100

mount_steps_q_low = np.median([pd.Series(learner.greedy_steps).rolling(N).median() for learner in q_learners_low_trained], axis=0)
mount_steps_q_high = np.median([pd.Series(learner.greedy_steps).rolling(N).median() for learner in q_learners_high_trained], axis=0)
mount_steps_mlr = np.median([pd.Series(learner.greedy_steps).rolling(N).median() for learner in mlr_learners_trained], axis=0)
mount_steps_tlr = np.median([pd.Series(learner.greedy_steps).rolling(N).median() for learner in tlr_learners_trained], axis=0)

mount_reward_q_low = np.median([np.mean(learner.greedy_cumulative_reward[-10:]) for learner in q_learners_low_trained])
mount_reward_q_high = np.median([np.mean(learner.greedy_cumulative_reward[-10:]) for learner in q_learners_high_trained])
mount_reward_mlr = np.median([np.mean(learner.greedy_cumulative_reward[-10:]) for learner in mlr_learners_trained])
mount_reward_tlr = np.median([np.mean(learner.greedy_cumulative_reward[-10:]) for learner in tlr_learners_trained])

print(mount_reward_q_low, mount_reward_q_high, mount_reward_mlr, mount_reward_tlr)
print(mount_steps_q_low.shape)


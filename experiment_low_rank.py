import gym
import json
from utils import Experiment
from environments import ContinuousCartPoleEnv, Continuous_MountainCarEnv, PendulumEnv

parameters_file_pend_ql_action = "experiments/pendulum_low_rank_q_learner_action.json"
parameters_file_pend_lr = "experiments/pendulum_low_rank_lr_learner.json"

parameters_file_cart_ql_action = "experiments/cartpole_low_rank_q_learner_action.json"
parameters_file_cart_lr = "experiments/cartpole_low_rank_lr_learner.json"

parameters_file_mountaincar_ql_action = "experiments/mountaincar_low_rank_q_learner_action.json"
parameters_file_mountaincar_lr = "experiments/mountaincar_low_rank_lr_learner.json"

with open(parameters_file_pend_ql_action) as j: parameters_pend_ql_action = json.loads(j.read())
with open(parameters_file_pend_lr) as j: parameters_pend_lr = json.loads(j.read())

with open(parameters_file_cart_ql_action) as j: parameters_cart_ql_action = json.loads(j.read())
with open(parameters_file_cart_lr) as j: parameters_cart_lr = json.loads(j.read())

with open(parameters_file_mountaincar_ql_action) as j: parameters_mountaincar_ql_action = json.loads(j.read())
with open(parameters_file_mountaincar_lr) as j: parameters_mountaincar_lr = json.loads(j.read())

env_pend = PendulumEnv()
env_cart = ContinuousCartPoleEnv()
env_mountaincar = Continuous_MountainCarEnv()

# 1) REWARD
"""
# 1.1) Pendulum
Experiment.run_q_learning_experiments(env_pend,
                                      parameters_pend_ql_action,
                                      "models/low_rank_reward/pendulum_ql_action_bucket_{}_exp_{}.pck")

Experiment.run_lr_learning_experiments(env_pend,
                                       parameters_pend_lr,
                                       "models/low_rank_reward/pendulum_lr_k_bucket_{}_exp_{}.pck")

print("Pendulum DONE")
"""
# 1.2) Cartpole
"""
Experiment.run_q_learning_experiments(env_cart,
                                      parameters_cart_ql_action,
                                      "models/low_rank_reward/cartpole_ql_action_bucket_{}_exp_{}.pck")

Experiment.run_lr_learning_experiments(env_cart,
                                       parameters_cart_lr,
                                       "models/low_rank_reward/cartpole_lr_k_bucket_{}_exp_{}.pck")

print("Cartpole DONE")"""

# 1.3) Mountaincar

Experiment.run_q_learning_experiments(env_mountaincar,
                                      parameters_mountaincar_ql_action,
                                      "models/low_rank_reward/mountaincar_ql_action_bucket_{}_exp_{}.pck")

Experiment.run_lr_learning_experiments(env_mountaincar,
                                       parameters_mountaincar_lr,
                                       "models/low_rank_reward/mountaincar_lr_k_bucket_{}_exp_{}.pck")

print("Mountaincar DONE")

"""

# 2) CONVERGENCE

# 2.1) Pendulum
Experiment.run_q_learning_experiments(env_pend,
                                      parameters_pend_ql_action,
                                      "models/low_rank_convergence/pendulum_ql_action_bucket_{}_exp_{}.pck",
                                      "models/low_rank_reward/pendulum_ql_action_bucket_{}_exp_{}.pck")

Experiment.run_lr_learning_experiments(env_pend,
                                       parameters_pend_lr,
                                       "models/low_rank_convergence/pendulum_lr_k_bucket_{}_exp_{}.pck",
                                       "models/low_rank_reward/pendulum_lr_k_bucket_{}_exp_{}.pck")

print("Pendulum DONE")
"""
# 2.2) Cartpole
"""Experiment.run_q_learning_experiments(env_cart,
                                      parameters_cart_ql_action,
                                      "models/low_rank_convergence/cartpole_conv_ql_action_bucket_{}_exp_{}.pck",
                                      "models/low_rank_reward/cartpole_reward_ql_action_bucket_{}_exp_{}.pck")

Experiment.run_lr_learning_experiments(env_cart,
                                       parameters_cart_lr,
                                       "models/low_rank_convergence/cartpole_conv_lr_k_bucket_{}_exp_{}.pck",
                                       "models/low_rank_reward/pendulum_reward_lr_k_bucket_{}_exp_{}.pck")

print("Cartpole DONE")"""

# 2.3) Mountaincar

Experiment.run_q_learning_experiments(env_mountaincar,
                                      parameters_mountaincar_ql_action,
                                      "models/low_rank_convergence/mountaincar_ql_action_bucket_{}_exp_{}.pck",
                                      "models/low_rank_reward/mountaincar_ql_action_bucket_{}_exp_{}.pck")

Experiment.run_lr_learning_experiments(env_mountaincar,
                                       parameters_mountaincar_lr,
                                       "models/low_rank_convergence/mountaincar_lr_k_bucket_{}_exp_{}.pck",
                                       "models/low_rank_reward/mountaincar_lr_k_bucket_{}_exp_{}.pck")
print("Mountaincar DONE")
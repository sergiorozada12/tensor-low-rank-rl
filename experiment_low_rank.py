import gym
import json
from utils import Experiment
from environments import ContinuousCartPoleEnv, Continuous_MountainCarEnv, PendulumEnv

parameters_file_pend_ql_action = "experiments/pendulum_low_rank_q_learner_action.json"
parameters_file_pend_ql_state = "experiments/pendulum_low_rank_q_learner_state.json"
parameters_file_pend_lr = "experiments/pendulum_low_rank_lr_learner.json"

parameters_file_cart_ql_action = "experiments/cartpole_low_rank_q_learner_action.json"
parameters_file_cart_lr = "experiments/cartpole_low_rank_lr_learner.json"

parameters_file_mountaincar_ql_action = "experiments/mountaincar_low_rank_q_learner_action.json"
parameters_file_mountaincar_lr = "experiments/mountaincar_low_rank_lr_learner.json"

with open(parameters_file_pend_ql_action) as j: parameters_pend_ql_action = json.loads(j.read())
with open(parameters_file_pend_ql_state) as j: parameters_pend_ql_state = json.loads(j.read())
with open(parameters_file_pend_lr) as j: parameters_pend_lr = json.loads(j.read())

with open(parameters_file_cart_ql_action) as j: parameters_cart_ql_action = json.loads(j.read())
with open(parameters_file_cart_lr) as j: parameters_cart_lr = json.loads(j.read())

with open(parameters_file_mountaincar_ql_action) as j: parameters_mountaincar_ql_action = json.loads(j.read())
with open(parameters_file_mountaincar_lr) as j: parameters_mountaincar_lr = json.loads(j.read())

env_pend = PendulumEnv()
env_cart = ContinuousCartPoleEnv()
env_mountaincar = Continuous_MountainCarEnv()

# 1) REWARD
#Experiment.run_q_learning_experiments(env_pend, parameters_pend_ql_action, "models/pendulum_reward_ql_action_bucket_{}_exp_{}.pck")
#Experiment.run_lr_learning_experiments(env_pend, parameters_pend_lr, "models/pendulum_reward_lr_k_bucket_{}_exp_{}.pck")

#Experiment.run_q_learning_experiments(env_cart, parameters_cart_ql_action, "models/cartpole_reward_ql_action_bucket_{}_exp_{}.pck")
#Experiment.run_lr_learning_experiments(env_cart, parameters_cart_lr, "models/cartpole_reward_lr_k_bucket_{}_exp_{}.pck")

#Experiment.run_q_learning_experiments(env_mountaincar, parameters_mountaincar_ql_action, "models/mountaincar_reward_ql_action_bucket_{}_exp_{}.pck")
Experiment.run_lr_learning_experiments(env_mountaincar, parameters_mountaincar_lr, "models/mountaincar_reward_lr_k_bucket_{}_exp_{}.pck")

# 2) CONVERGENCE
#Experiment.run_q_learning_experiments(env_pend, parameters_pend_ql_action, "models/pendulum_conv_ql_action_bucket_{}_exp_{}.pck", "models/pendulum_reward_ql_action_bucket_{}_exp_{}.pck")
#Experiment.run_lr_learning_experiments(env_pend, parameters_pend_lr, "models/pendulum_conv_lr_k_bucket_{}_exp_{}.pck", "models/pendulum_reward_lr_k_bucket_{}_exp_{}.pck")

#Experiment.run_q_learning_experiments(env_cart, parameters_cart_ql_action, "models/cartpole_conv_ql_action_bucket_{}_exp_{}.pck", "models/cartpole_reward_ql_action_bucket_{}_exp_{}.pck")
#Experiment.run_lr_learning_experiments(env_cart, parameters_cart_lr, "models/cartpole_conv_lr_k_bucket_{}_exp_{}.pck", "models/pendulum_reward_lr_k_bucket_{}_exp_{}.pck")

#Experiment.run_q_learning_experiments(env_mountaincar, parameters_mountaincar_ql_action, "models/mountaincar_conv_ql_action_bucket_{}_exp_{}.pck", "models/mountaincar_reward_ql_action_bucket_{}_exp_{}.pck")
#Experiment.run_lr_learning_experiments(env_mountaincar, parameters_mountaincar_lr, "models/mountaincar_conv_lr_k_bucket_{}_exp_{}.pck", "models/mountaincar_reward_lr_k_bucket_{}_exp_{}.pck")

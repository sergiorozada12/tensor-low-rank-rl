import gym
import json
from utils import Experiment
from environments import ContinuousCartPoleEnv, Continuous_MountainCarEnv

parameters_file_pend_ql_action = "experiments/pendulum_low_rank_q_learner_action.json"
parameters_file_pend_ql_state = "experiments/pendulum_low_rank_q_learner_state.json"
parameters_file_pend_lr = "experiments/pendulum_low_rank_lr_learner.json"

with open(parameters_file_pend_ql_action) as j: parameters_pend_ql_action = json.loads(j.read())
with open(parameters_file_pend_ql_state) as j: parameters_pend_ql_state = json.loads(j.read())
with open(parameters_file_pend_lr) as j: parameters_pend_lr = json.loads(j.read())

env_pend = gym.make('Pendulum-v0')

run_q_learning_experiments(env_pend, parameters_file_pend_ql_state, "models/pendulum_ql_state_bucket_{}_exp_{}.pck", False, True)

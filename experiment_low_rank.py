import gym
import json
from utils import Experiment
from environments import ContinuousCartPoleEnv, Continuous_MountainCarEnv

parameters_file_pend_ql = "experiments/pendulum_low_rank_q_learner.json"
parameters_file_pend_lr = "experiments/pendulum_low_rank_lr_learner.json"

with open(parameters_file_pend_ql) as j: parameters_pend_ql = json.loads(j.read())
with open(parameters_file_pend_lr) as j: parameters_pend_lr = json.loads(j.read())

env_pend = gym.make('Pendulum-v0')

for i in range(len(parameters_file_pend_ql["bucket_actions"])):
    parameters = parameters_pend_ql.copy()
    parameters["bucket_actions"] = parameters_pend_ql["bucket_actions"][i]
    for j in range(parameters_file_pend_ql["n_simulations"]):
        path_output = "models/pendulum_ql_bucket_{}_exp_{}.pck".format(i, j)
        Experiment.run_q_learning_experiment(env_pend, parameters, path_output)

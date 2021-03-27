import gym
import json
from utils import Experiment
from environments import PendulumEnv, ContinuousCartPoleEnv, Continuous_MountainCarEnv

parameters_file_pend = "experiments/pendulum_baseline.json"
parameters_file_cart = "experiments/cartpole_baseline.json"
parameters_file_moun = "experiments/mountaincar_baseline.json"

with open(parameters_file_pend) as j: parameters_pend = json.loads(j.read())
with open(parameters_file_cart) as j :parameters_cart = json.loads(j.read())
with open(parameters_file_moun) as j: parameters_moun = json.loads(j.read())

env_pend = PendulumEnv()
env_cart = ContinuousCartPoleEnv()
env_moun = Continuous_MountainCarEnv()

Experiment.run_q_learning_experiment(env_pend, parameters_pend, "models/pendulum_baseline.pck")
Experiment.run_q_learning_experiment(env_cart, parameters_cart, "models/cartpole_baseline.pck")
Experiment.run_q_learning_experiment(env_moun, parameters_moun, "models/mountaincar_baseline.pck")

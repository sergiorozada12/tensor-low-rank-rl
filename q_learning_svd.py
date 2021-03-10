import gym
import json
from utils import Discretizer, Saver
from models import QLearning
from environments import ContinuousCartPoleEnv, Continuous_MountainCarEnv

#parameters_file = "experiments/pendulum_svd.json"
#parameters_file = "experiments/cartpole_svd.json"
parameters_file = "experiments/mountaincar_svd.json"
with open(parameters_file) as j:
    parameters = json.loads(j.read())

#env = gym.make('Pendulum-v0')
#env = ContinuousCartPoleEnv()
env = Continuous_MountainCarEnv()
saver = Saver()

discretizer = Discretizer(min_points_states=parameters["min_states"],
                          max_points_states=parameters["max_states"],
                          bucket_states=parameters["bucket_states"],
                          min_points_actions=parameters["min_actions"],
                          max_points_actions=parameters["max_actions"],
                          bucket_actions=parameters["bucket_actions"])

q_learner = QLearning(env=env,
                      discretizer=discretizer,
                      episodes=parameters["episodes"],
                      max_steps=parameters["max_steps"],
                      epsilon=parameters["epsilon"],
                      alpha=parameters["alpha"],
                      gamma=parameters["gamma"],
                      decay=decay["decay"])

q_learner.train()
#saver.save_to_pickle("models/pendulum_Q_svd.pck", q_learner.Q)
#saver.save_to_pickle("models/cartpole_Q_svd.pck", q_learner.Q)
saver.save_to_pickle("models/mountaincar_Q_svd.pck", q_learner.Q)

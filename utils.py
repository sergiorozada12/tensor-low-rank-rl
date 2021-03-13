import numpy as np
import pickle
from models import QLearning


class Discretizer:
    def __init__(self,
                 min_points_states,
                 max_points_states,
                 bucket_states,
                 min_points_actions,
                 max_points_actions,
                 bucket_actions):
        self.min_points_states = np.array(min_points_states)
        self.max_points_states = np.array(max_points_states)
        self.bucket_states = np.array(bucket_states)
        self.spacing_states = (self.max_points_states - self.min_points_states) / self.bucket_states
        self.range_states = self.max_points_states - self.min_points_states

        self.min_points_actions = np.array(min_points_actions)
        self.max_points_actions = np.array(max_points_actions)
        self.bucket_actions = np.array(bucket_actions)
        self.spacing_actions = (self.max_points_actions - self.min_points_actions) / self.bucket_actions
        self.range_actions = self.max_points_actions - self.min_points_actions

        self.n_states = np.round((self.max_points_states - self.min_points_states) / self.spacing_states + 1).astype(
            int)
        self.n_actions = np.round(
            (self.max_points_actions - self.min_points_actions) / self.spacing_actions + 1).astype(int)
        self.dimensions = np.concatenate((self.n_states, self.n_actions))

    def get_state_index(self, state):
        state = np.clip(state, a_min=self.min_points_states, a_max=self.max_points_states)
        scaling = (state - self.min_points_states) / self.range_states
        state_idx = np.round(scaling * (self.bucket_states - 1)).astype(int)
        return tuple(state_idx.tolist())

    def get_action_index(self, action):
        action = np.clip(action, a_min=self.min_points_actions, a_max=self.max_points_actions)
        scaling = (action - self.min_points_actions) / self.range_actions
        action_idx = np.round(scaling * (self.bucket_actions - 1)).astype(int)
        return tuple(action_idx.tolist())

    def get_action_from_index(self, action_idx):
        return self.min_points_actions + self.spacing_actions / 2 + action_idx * self.spacing_actions


class Saver:
    @staticmethod
    def save_to_pickle(path, obj):
        """
        :param path: str
            Path to store the object.
        :param obj
            Object to store.
        """
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_from_pickle(path):
        """
        :param path: str
            Path of the object to load.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)


class Experiment:

    @staticmethod
    def run_q_learning_experiment(env, parameters, path_output, use_decay=False):

        saver = Saver()
        discretizer = Discretizer(min_points_states=parameters["min_states"],
                                  max_points_states=parameters["max_states"],
                                  bucket_states=parameters["bucket_states"],
                                  min_points_actions=parameters["min_actions"],
                                  max_points_actions=parameters["max_actions"],
                                  bucket_actions=parameters["bucket_actions"])

        decay = parameters["decay"] if use_decay else 1.0

        q_learner = QLearning(env=env,
                              discretizer=discretizer,
                              episodes=parameters["episodes"],
                              max_steps=parameters["max_steps"],
                              epsilon=parameters["epsilon"],
                              alpha=parameters["alpha"],
                              gamma=parameters["gamma"],
                              decay=decay)

        q_learner.train()
        saver.save_to_pickle(path_output, q_learner)

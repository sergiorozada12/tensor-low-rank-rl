import numpy as np
import pickle


class Discretizer:
    def __init__(self,
                 min_points_states,
                 max_points_states,
                 spacing_states,
                 min_points_actions,
                 max_points_actions,
                 spacing_actions):
        self.min_points_states = np.array(min_points_states)
        self.max_points_states = np.array(max_points_states)
        self.spacing_states = np.array(spacing_states)

        self.min_points_actions = np.array(min_points_actions)
        self.max_points_actions = np.array(max_points_actions)
        self.spacing_actions = np.array(spacing_actions)

        self.n_states = np.round((self.max_points_states - self.min_points_states)/self.spacing_states + 1).astype(
            int)
        self.n_actions = np.round(
            (self.max_points_actions - self.min_points_actions)/self.spacing_actions + 1).astype(int)
        self.dimensions = np.concatenate((self.n_states, self.n_actions))

    def get_state_index(self, state):
        state_idx = np.round((state - self.min_points_states)/self.spacing_states).astype(int)
        return tuple(state_idx.tolist())

    def get_action_index(self, action):
        action_idx = np.round((action - self.min_points_actions)/self.spacing_actions).astype(int)
        return tuple(action_idx.tolist())

    def get_action_from_index(self, action_idx):
        if type(action_idx) == int:
            return self.min_points_actions[0] + action_idx*self.spacing_actions[0]
        return self.min_points_actions + action_idx*self.spacing_actions


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
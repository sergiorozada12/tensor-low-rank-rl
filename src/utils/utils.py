import numpy as np


class Discretizer:
    def __init__(
        self,
        min_points_states,
        max_points_states,
        bucket_states,
        min_points_actions,
        max_points_actions,
        bucket_actions,
        discrete_action=False,
        states_structure=None,
        ):

        self.discrete_action = discrete_action

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
        
        self.n_states = np.round(self.bucket_states).astype(int)
        self.n_actions = np.round(self.bucket_actions).astype(int)
        self.dimensions = np.concatenate((self.n_states, self.n_actions))
        
        self.states_structure = states_structure
        
        if self.states_structure != None:
            new_n_states = list()
            for i in range(len(self.states_structure)):
                ndim = self.states_structure[i]
                origin_dim = np.sum(self.states_structure[0:i]) if i>0 else 0
                new_n_states.append(np.prod(self.n_states[origin_dim:origin_dim+ndim]))
            self.n_states = new_n_states
            self.dimensions = np.concatenate((self.n_states, self.n_actions))

    def get_state_index(self, state):
        state = np.clip(state, a_min=self.min_points_states, a_max=self.max_points_states)
        scaling = (state - self.min_points_states) / self.range_states
        state_idx = np.round(scaling * (self.bucket_states - 1)).astype(int)
        if self.states_structure is None:
            return tuple(state_idx.tolist())

        indices = list()
        n_states = np.round(self.bucket_states).astype(int)
        for i in range(len(self.states_structure)):
            ndim = self.states_structure[i]
            origin_dim = np.sum(self.states_structure[0:i]) if i>0 else 0
            new_index = np.ravel_multi_index(
                state_idx[origin_dim:origin_dim+ndim], 
                dims=n_states[origin_dim:origin_dim+ndim]
            )
            indices.append(new_index)
        return tuple(indices)

    def get_action_index(self, action):
        action = np.clip(action, a_min=self.min_points_actions, a_max=self.max_points_actions)
        scaling = (action - self.min_points_actions) / self.range_actions
        action_idx = np.round(scaling * (self.bucket_actions - 1)).astype(int)
        return tuple(action_idx.tolist())

    def get_action_from_index(self, action_idx):
        if self.discrete_action:
            return action_idx[0]
        return self.min_points_actions + self.spacing_actions / 2 + action_idx * self.spacing_actions
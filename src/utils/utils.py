import numpy as np
from collections import namedtuple
import random
import operator

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
Data = namedtuple('Data', ('priority', 'probability', 'weight', 'index'))

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


class ReplayBuffer(object):
    def __init__(
        self,
        capacity,
        seed):

        self.capacity = capacity
        self.position = 0
        self.memory = []
        np.random.seed(seed)

        self.type = 'Uniform'

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample_batch(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def get_buffer_size(self):
        return len(self.memory)

# Kudos to @Gillaume-Cr
class PrioritizedReplayBuffer(object):
    def __init__(self, action_size, buffer_size, batch_size, experiences_per_sampling, seed, compute_weights):
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experiences_per_sampling = experiences_per_sampling

        self.alpha = 0.5
        self.alpha_decay_rate = 0.99
        self.beta = 0.5
        self.beta_growth_rate = 1.001
        self.seed = random.seed(seed)
        self.compute_weights = compute_weights
        self.experience_count = 0

        indexes = []
        datas = []
        for i in range(buffer_size):
            indexes.append(i)
            d = Data(0, 0, 0, i)
            datas.append(d)

        self.memory = {key: Transition for key in indexes}
        self.memory_data = {key: data for key,data in zip(indexes, datas)}
        self.sampled_batches = []
        self.current_batch = 0
        self.priorities_sum_alpha = 0
        self.priorities_max = 1
        self.weights_max = 1

        self.type = 'PER'

    def update_priorities(self, tds, indices):
        for td, index in zip(tds, indices):
            N = min(self.experience_count, self.buffer_size)

            updated_priority = td[0]
            if updated_priority > self.priorities_max:
                self.priorities_max = updated_priority
            
            if self.compute_weights:
                updated_weight = ((N * updated_priority)**(-self.beta))/self.weights_max
                if updated_weight > self.weights_max:
                    self.weights_max = updated_weight
            else:
                updated_weight = 1

            old_priority = self.memory_data[index].priority
            self.priorities_sum_alpha += updated_priority**self.alpha - old_priority**self.alpha
            updated_probability = td[0]**self.alpha / self.priorities_sum_alpha
            data = Data(updated_priority, updated_probability, updated_weight, index) 
            self.memory_data[index] = data

    def update_memory_sampling(self):
        self.current_batch = 0
        values = list(self.memory_data.values())
        random_values = random.choices(
            self.memory_data,
            [data.probability for data in values],
            k=self.experiences_per_sampling
        )
        self.sampled_batches = [random_values[i:i + self.batch_size] 
                                    for i in range(0, len(random_values), self.batch_size)]

    def update_parameters(self):
        self.alpha *= self.alpha_decay_rate
        self.beta *= self.beta_growth_rate
        if self.beta > 1:
            self.beta = 1
        N = min(self.experience_count, self.buffer_size)
        self.priorities_sum_alpha = 0
        sum_prob_before = 0
        for element in self.memory_data.values():
            sum_prob_before += element.probability
            self.priorities_sum_alpha += element.priority**self.alpha
        sum_prob_after = 0
        for element in self.memory_data.values():
            probability = element.priority**self.alpha/self.priorities_sum_alpha
            sum_prob_after += probability
            weight = 1
            if self.compute_weights:
                weight = ((N*element.probability)**(-self.beta))/self.weights_max
            d = Data(element.priority, probability, weight, element.index)
            self.memory_data[element.index] = d

    def push(self, *args):
        self.experience_count += 1
        index = self.experience_count % self.buffer_size

        if self.experience_count > self.buffer_size:
            temp = self.memory_data[index]
            self.priorities_sum_alpha -= temp.priority**self.alpha
            if temp.priority == self.priorities_max:
                self.memory_data[index].priority = 0
                self.priorities_max = max(self.memory_data.items(), key=operator.itemgetter(1)).priority
            if self.compute_weights:
                if temp.weight == self.weights_max:
                    self.memory_data[index].weight = 0
                    self.weights_max = max(self.memory_data.items(), key=operator.itemgetter(2)).weight

        priority = self.priorities_max
        weight = self.weights_max
        self.priorities_sum_alpha += priority ** self.alpha
        probability = priority ** self.alpha / self.priorities_sum_alpha
        e = Transition(*args)
        self.memory[index] = e
        d = Data(priority, probability, weight, index)
        self.memory_data[index] = d
            
    def sample_batch(self, batch_size):
        sampled_batch = self.sampled_batches[self.current_batch]
        self.current_batch += 1
        batch = [self.memory.get(sample.index) for sample in sampled_batch]
        return batch

    def __len__(self):
        return len(self.memory)

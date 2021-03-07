import numpy as np


class QLearning:
    def __init__(self,
                 env,
                 discretizer,
                 episodes,
                 max_steps,
                 epsilon,
                 alpha,
                 gamma):

        self.env = env
        self.discretizer = discretizer
        self.episodes = episodes
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.Q = np.zeros(self.discretizer.dimensions)

        self.training_steps = []
        self.training_cumulative_reward = []
        self.greedy_steps = []
        self.greedy_cumulative_reward = []

    def get_random_action(self):
        return self.env.action_space.sample()

    def get_greedy_action(self, state):
        state_idx = self.discretizer.get_state_index(state)
        action_idx = np.argmax(self.Q[state_idx + (slice(None),)])
        return self.discretizer.get_action_from_index(action_idx)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.get_random_action()
        return self.get_greedy_action(state)

    def update_q_matrix(self, state, action, state_prime, reward):
        state_idx = self.discretizer.get_state_index(state)
        state_prime_idx = self.discretizer.get_state_index(state_prime)
        action_idx = self.discretizer.get_action_index(action)

        target_q = reward + self.gamma * np.max(self.Q[state_prime_idx + (slice(None),)])
        q = self.Q[state_idx + action_idx]

        error_signal = target_q - q
        self.Q[state_idx + action_idx] += self.alpha * error_signal

    def run_episode(self, is_train=True, is_greedy=False):
        state = self.env.reset()
        cumulative_reward = 0

        for step in range(self.max_steps):
            action = self.get_greedy_action(state) if is_greedy else self.choose_action(state)
            state_prime, reward, done, _ = self.env.step(action)

            if is_train:
                self.update_q_matrix(state, action, state_prime, reward)

            if done:
                break

            state = state_prime
            cumulative_reward += reward

        return step + 1, cumulative_reward

    def run_training_episode(self):
        n_steps, cumulative_reward = self.run_episode(is_train=True, is_greedy=False)
        self.training_steps.append(n_steps)
        self.training_cumulative_reward.append(cumulative_reward)

    def run_greedy_episode(self):
        n_steps, cumulative_reward = self.run_episode(is_train=False, is_greedy=True)
        self.greedy_steps.append(n_steps)
        self.greedy_cumulative_reward.append(cumulative_reward)

    def run_testing_episode(self):
        return run_episode(is_train=False, is_greedy=True)

    def train(self, run_greedy_frequency=None):
        if run_greedy_frequency:
            for episode in range(self.episodes):
                self.run_training_episode()

                if (episode % run_greedy_frequency) == 0:
                    self.run_greedy_episode()
        else:
            for _ in range(self.episodes):
                self.run_training_episode()


class LowRankLearning:
    def __init__(self,
                 env,
                 discretizer,
                 episodes,
                 max_steps,
                 epsilon,
                 alpha,
                 gamma,
                 k):

        self.env = env
        self.discretizer = discretizer
        self.episodes = episodes
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.L = np.random.rand(self.discretizer.n_states + [k]) * 1e-3
        self.R = np.random.rand([k] + self.discretizer.n_actions) * 1e-3

        self.training_steps = []
        self.training_cumulative_reward = []
        self.greedy_steps = []
        self.greedy_cumulative_reward = []

    def get_random_action(self):
        return self.env.action_space.sample()

    def get_greedy_action(self, state):
        state_idx = self.discretizer.get_state_index(state)
        action_idx = np.argmax(self.L[state_idx + (slice(None),)].reshape(1, -1) @ self.R)
        return self.discretizer.get_action_from_index(action_idx)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.get_random_action()
        return self.get_greedy_action(state)

    def update_q_matrix(self, state, action, state_prime, reward):
        state_idx = self.discretizer.get_state_index(state)
        state_prime_idx = self.discretizer.get_state_index(state_prime)
        action_idx = self.discretizer.get_action_index(action)

        target_q = reward + self.gamma * np.max(self.L[state_idx + (slice(None),)].reshape(1, -1) @ self.R)
        q = np.dot(self.L[state_idx + (slice(None),)], self.R[(slice(None),) + action_idx])

        error_signal = target_q - q
        grad_R = self.R[(slice(None),) + action_idx]
        grad_L = self.L[state_idx + (slice(None),)]

        self.L[state_idx + (slice(None),)] += self.alpha * error_signal * grad_R / np.linalg.norm(grad_R)
        self.R[(slice(None),) + action_idx] += self.alpha * error_signal * grad_L / np.linalg.norm(grad_L)

    def run_episode(self, is_train=True, is_greedy=False):
        state = self.env.reset()
        cumulative_reward = 0

        for step in range(self.max_steps):
            action = self.get_greedy_action(state) if is_greedy else self.choose_action(state)
            state_prime, reward, done, _ = self.env.step(action)

            if is_train:
                self.update_q_matrix(state, action, state_prime, reward)

            if done:
                break

            state = state_prime
            cumulative_reward += reward

        return step + 1, cumulative_reward

    def run_training_episode(self):
        n_steps, cumulative_reward = self.run_episode(is_train=True, is_greedy=False)
        self.training_steps.append(n_steps)
        self.training_cumulative_reward.append(cumulative_reward)

    def run_greedy_episode(self):
        n_steps, cumulative_reward = self.run_episode(is_train=False, is_greedy=True)
        self.greedy_steps.append(n_steps)
        self.greedy_cumulative_reward.append(cumulative_reward)

    def run_testing_episode(self):
        return run_episode(is_train=False, is_greedy=True)

    def train(self, run_greedy_frequency=None):
        if run_greedy_frequency:
            for episode in range(self.episodes):
                self.run_training_episode()

                if (episode % run_greedy_frequency) == 0:
                    self.run_greedy_episode()
        else:
            for episode in range(self.episodes):
                self.run_training_episode()

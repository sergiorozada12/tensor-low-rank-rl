import numpy as np
import tensorly as tl


class QLearning:
    def __init__(
        self,
        env,
        discretizer,
        episodes,
        max_steps,
        epsilon,
        alpha,
        gamma,
        decay=1.0,
        min_epsilon=0.0,
        Q_ground_truth=None
        ):

        self.env = env
        self.discretizer = discretizer
        self.episodes = episodes
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.decay = decay
        self.min_epsilon = min_epsilon

        self.Q = np.zeros(self.discretizer.dimensions)
        self.Q_gt = Q_ground_truth

        if self.Q_gt is not None:
            self.Q_gt_norm = np.linalg.norm(Q_ground_truth.flatten(), ord=2)
            self.Q_norms = []

            self.estimate_deviation()

        self.training_steps = []
        self.training_cumulative_reward = []
        self.greedy_steps = []
        self.greedy_cumulative_reward = []

    def estimate_deviation(self):
        norm = np.linalg.norm(self.Q_gt.flatten() - self.Q.flatten(), ord=2)/self.Q_gt_norm
        self.Q_norms.append(norm)

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

        if self.Q_gt is not None:
            self.estimate_deviation()

    def run_episode(self, is_train=True, is_greedy=False):
        state = self.env.reset()
        cumulative_reward = 0

        for step in range(self.max_steps):
            action = self.get_greedy_action(state) if is_greedy else self.choose_action(state)
            state_prime, reward, done, _ = self.env.step(action)
            cumulative_reward += reward

            if is_train:
                self.update_q_matrix(state, action, state_prime, reward)

            if done:
                break

            state = state_prime

            if (not is_greedy) & is_train & (self.epsilon > self.min_epsilon):
                self.epsilon *= self.decay

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
        return self.run_episode(is_train=False, is_greedy=True)

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
    def __init__(
        self,
        env,
        discretizer,
        episodes,
        max_steps,
        epsilon,
        alpha,
        gamma,
        k,
        decay=1.0,
        decay_alpha=1.0,
        init_ord=1,
        min_epsilon=0.0,
        Q_hat_ground_truth=None
        ):

        self.env = env
        self.discretizer = discretizer
        self.episodes = episodes
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.decay = decay
        self.decay_alpha = decay_alpha
        self.min_epsilon = min_epsilon

        self.L = np.random.rand(*(list(self.discretizer.n_states) + [k]))*init_ord
        self.R = np.random.rand(*([k] + list(self.discretizer.n_actions)))*init_ord

        self.Q_hat_gt = Q_hat_ground_truth

        if self.Q_hat_gt is not None:
            self.Q_hat_gt_norm = np.linalg.norm(Q_hat_ground_truth.flatten(), ord=2)
            self.Q_hat_norms = []

            self.estimate_deviation()

        self.training_steps = []
        self.training_cumulative_reward = []
        self.greedy_steps = []
        self.greedy_cumulative_reward = []

    def estimate_deviation(self):
        self.Q_hat = self.L @ self.R
        norm = np.linalg.norm(self.Q_hat_gt.flatten() - self.Q_hat.flatten(), ord=2)/self.Q_hat_gt_norm
        self.Q_hat_norms.append(norm)

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

        target_q = reward + self.gamma * np.max(self.L[state_prime_idx + (slice(None),)].reshape(1, -1) @ self.R)
        q = np.dot(self.L[state_idx + (slice(None),)], self.R[(slice(None),) + action_idx])

        error_signal = target_q - q
        grad_R = self.R[(slice(None),) + action_idx]
        grad_L = self.L[state_idx + (slice(None),)]

        self.L[state_idx + (slice(None),)] += self.alpha * error_signal * grad_R / np.linalg.norm(grad_R)
        self.R[(slice(None),) + action_idx] += self.alpha * error_signal * grad_L / np.linalg.norm(grad_L)

        if self.Q_hat_gt is not None:
            self.estimate_deviation()

    def run_episode(self, is_train=True, is_greedy=False):
        state = self.env.reset()
        cumulative_reward = 0

        for step in range(self.max_steps):
            action = self.get_greedy_action(state) if is_greedy else self.choose_action(state)
            state_prime, reward, done, _ = self.env.step(action)
            cumulative_reward += reward

            if is_train:
                self.update_q_matrix(state, action, state_prime, reward)

            if done:
                break

            state = state_prime

            if (not is_greedy) & is_train & (self.epsilon > self.min_epsilon):
                self.epsilon *= self.decay

            if (not is_greedy) & is_train:
                self.alpha *= self.decay_alpha

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
        return self.run_episode(is_train=False, is_greedy=True)

    def train(self, run_greedy_frequency=None):
        if run_greedy_frequency:
            for episode in range(self.episodes):
                self.run_training_episode()

                if (episode % run_greedy_frequency) == 0:
                    self.run_greedy_episode()
        else:
            for _ in range(self.episodes):
                self.run_training_episode()


class TensorLowRankLearning:
    def __init__(
        self,
        env,
        discretizer,
        episodes,
        max_steps,
        epsilon,
        alpha,
        gamma,
        k,
        decay=1.0,
        decay_alpha=1.0,
        init_ord=1,
        min_epsilon=0.0
        ):

        self.env = env
        self.discretizer = discretizer
        self.episodes = episodes
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.decay = decay
        self.decay_alpha = decay_alpha
        self.min_epsilon = min_epsilon
        self.k = k

        self.factors = [np.random.rand(dim, self.k) * init_ord for dim in self.discretizer.dimensions]
        self.factor_indices = np.arange(len(self.factors))
        self.non_factor_indices = [np.delete(self.factor_indices, i).tolist() for i in self.factor_indices]

        self.training_steps = []
        self.training_cumulative_reward = []
        self.greedy_steps = []
        self.greedy_cumulative_reward = []

    def get_random_action(self):
        return self.env.action_space.sample()

    def get_greedy_action(self, state):
        state_idx = self.discretizer.get_state_index(state)
        q_simplified = self.get_q_from_state_idx(state_idx).reshape(self.discretizer.n_actions)
        action_idx = np.unravel_index(np.argmax(q_simplified), q_simplified.shape)
        return self.discretizer.get_action_from_index(action_idx)

    def get_q_from_state_idx(self, state_idx):
        state_value = np.ones(self.k)
        for idx, factor in enumerate(self.factors[:len(state_idx)]):
            state_value *= factor[state_idx[idx], :]
        action_khatri = tl.tenalg.khatri_rao(self.factors[len(state_idx):], reverse=False)
        action_state = np.sum(state_value * action_khatri, axis=1)
        return action_state

    def get_q_from_state_action_idx(self, state_idx, action_idx):
        indices = state_idx + action_idx
        q_value = np.ones(self.k)
        for idx, factor in enumerate(self.factors):
            q_value *= factor[indices[idx], :]
        return np.sum(q_value)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.get_random_action()
        return self.get_greedy_action(state)

    def update_q_matrix(self, state, action, state_prime, reward):

        state_idx = self.discretizer.get_state_index(state)
        state_prime_idx = self.discretizer.get_state_index(state_prime)
        action_idx = self.discretizer.get_action_index(action)

        q_simplified = self.get_q_from_state_idx(state_prime_idx)
        target_q = reward + self.gamma * np.max(q_simplified)
        q = self.get_q_from_state_action_idx(state_idx, action_idx)

        error_signal = target_q - q

        tensor_indices = state_idx + action_idx

        for factor_idx in self.factor_indices:
            grad_factor = np.ones(self.k)
            for non_factor_idx in self.non_factor_indices[factor_idx]:
                grad_factor *= self.factors[non_factor_idx][tensor_indices[non_factor_idx], :]

            update = -error_signal * grad_factor / np.linalg.norm(grad_factor)
            self.factors[factor_idx][tensor_indices[factor_idx], :] -= self.alpha * update

    def run_episode(self, is_train=True, is_greedy=False):
        state = self.env.reset()
        cumulative_reward = 0

        for step in range(self.max_steps):
            action = self.get_greedy_action(state) if is_greedy else self.choose_action(state)
            state_prime, reward, done, _ = self.env.step(action)
            cumulative_reward += reward

            if is_train:
                self.update_q_matrix(state, action, state_prime, reward)

            if done:
                break

            state = state_prime

            if (not is_greedy) & is_train & (self.epsilon > self.min_epsilon):
                self.epsilon *= self.decay

            if (not is_greedy):
                self.alpha *= self.decay_alpha

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
        return self.run_episode(is_train=False, is_greedy=True)

    def train(self, run_greedy_frequency=None):
        if run_greedy_frequency:
            for episode in range(self.episodes):
                self.run_training_episode()

                if (episode % run_greedy_frequency) == 0:
                    self.run_greedy_episode()
        else:
            for _ in range(self.episodes):
                self.run_training_episode()

import numpy as np


class MatrixLowRankLearning:
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
        Q_hat_ground_truth=None,
        bias=0.0,
        normalize_columns=False,
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
        self.normalize_columns = normalize_columns
        self.k = k

        self.L = (np.random.rand(*(list(self.discretizer.n_states) + [k])) - bias)*init_ord
        self.R = (np.random.rand(*([k] + list(self.discretizer.n_actions))) - bias)*init_ord

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

    def normalize(self):
        L = self.L.reshape(-1, self.k)
        norm_l = np.linalg.norm(L, axis=0)

        R = self.R.reshape(self.k, -1)
        norm_r = np.linalg.norm(R, axis=1)

        factor_l = np.sqrt(norm_r)/np.sqrt(norm_l)
        factor_r = np.sqrt(norm_l)/np.sqrt(norm_r)

        self.L = (L*factor_l).reshape(self.L.shape)
        self.R = (R.T*factor_r).T.reshape(self.R.shape)

    def update_q_matrix(self, state, action, state_prime, reward, done):
        state_idx = self.discretizer.get_state_index(state)
        state_prime_idx = self.discretizer.get_state_index(state_prime)
        action_idx = self.discretizer.get_action_index(action)

        q_next = np.max(self.L[state_prime_idx + (slice(None),)].reshape(1, -1) @ self.R) if not done else 0
        target_q = reward + self.gamma * q_next
        q = np.dot(self.L[state_idx + (slice(None),)], self.R[(slice(None),) + action_idx])

        error_signal = target_q - q
        grad_R = self.R[(slice(None),) + action_idx]
        grad_L = self.L[state_idx + (slice(None),)]

        self.L[state_idx + (slice(None),)] += self.alpha * error_signal * grad_R / np.linalg.norm(grad_R)
        self.R[(slice(None),) + action_idx] += self.alpha * error_signal * grad_L / np.linalg.norm(grad_L)

        if self.normalize_columns:
            self.normalize()

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
                self.update_q_matrix(state, action, state_prime, reward, done)

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
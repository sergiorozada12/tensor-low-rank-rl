import numpy as np
import time


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

    def update_q_matrix(self, state, action, state_prime, reward, done):
        state_idx = self.discretizer.get_state_index(state)
        state_prime_idx = self.discretizer.get_state_index(state_prime)
        action_idx = self.discretizer.get_action_index(action)

        q_next = np.max(self.Q[state_prime_idx + (slice(None),)]) if not done else 0
        target_q = reward + self.gamma * q_next
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
                self.update_q_matrix(state, action, state_prime, reward, done)

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

    def evaluate_final_policy(self):
        rewards = []
        for _ in range(1_000):
            _, cumulative_reward = self.run_episode(is_train=False, is_greedy=True)
            rewards.append(cumulative_reward)
        self.mean_reward = np.mean(rewards)
        self.std_reward = np.std(rewards)

    def train(self, run_greedy_frequency=None):
        if run_greedy_frequency:
            for episode in range(self.episodes):
                self.run_training_episode()

                if (episode % run_greedy_frequency) == 0:
                    self.run_greedy_episode()
        else:
            for _ in range(self.episodes):
                self.run_training_episode()

        self.evaluate_final_policy()

    def measure_mean_runtime(self):
        state = self.env.reset()
        action = self.choose_action(state)
        state_prime, reward, done, _ = self.env.step(action)

        start_time = time.time()
        for _ in range(100_000):
            self.update_q_matrix(state, action, state_prime, reward, done)
        end_time = time.time()
        return end_time - start_time    

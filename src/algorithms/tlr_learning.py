import numpy as np
import tensorly as tl


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
        min_epsilon=0.0,
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
        self.k = k
        self.normalize_columns = normalize_columns

        self.factors = [(np.random.rand(dim, self.k) - bias) * init_ord for dim in self.discretizer.dimensions]
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
        if self.discretizer.discrete_action:
            return action_idx[0]
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

    def normalize(self):
        n_factors = len(self.factors)

        power_denominator = (n_factors - 1)/n_factors
        power_numerator = 1/n_factors

        norms_denominator = [np.linalg.norm(factor, axis=0)**power_denominator for factor in self.factors]
        norms_numerator = [np.linalg.norm(factor, axis=0)**power_numerator for factor in self.factors]

        for i in range(n_factors):
            numerator = np.prod([norms_numerator[j] for j in range(n_factors) if j != i], axis=0)
            scaler = numerator/norms_denominator[i]
            self.factors[i] *= scaler

    def update_q_matrix(self, state, action, state_prime, reward, done):

        state_idx = self.discretizer.get_state_index(state)
        state_prime_idx = self.discretizer.get_state_index(state_prime)
        action_idx = self.discretizer.get_action_index(action)

        q_next = np.max(self.get_q_from_state_idx(state_prime_idx)) if not done else 0
        target_q = reward + self.gamma * np.max(q_next)
        q = self.get_q_from_state_action_idx(state_idx, action_idx)

        error_signal = target_q - q

        tensor_indices = state_idx + action_idx

        new_factors = self.factors[:]
        for factor_idx in self.factor_indices:
            grad_factor = np.ones(self.k)
            for non_factor_idx in self.non_factor_indices[factor_idx]:
                grad_factor *= self.factors[non_factor_idx][tensor_indices[non_factor_idx], :]

            update = -error_signal * grad_factor / np.linalg.norm(grad_factor)
            new_factors[factor_idx][tensor_indices[factor_idx], :] -= self.alpha * update
        self.factors = new_factors[:]
        if self.normalize_columns:
            self.normalize()

    def run_episode(self, is_train=True, is_greedy=False):
        state = self.env.reset()
        cumulative_reward = 0

        if len(state.shape) > 1:
            state = state.flatten()

        for step in range(self.max_steps):
            action = self.get_greedy_action(state) if is_greedy else self.choose_action(state)
            state_prime, reward, done, _ = self.env.step(action)
            cumulative_reward += reward

            if len(state_prime.shape) > 1:
                state_prime = state_prime.flatten()

            if is_train:
                self.update_q_matrix(state, action, state_prime, reward, done)

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

                if episode > 0 and episode % 1000 == 0:
                    print(episode, self.greedy_cumulative_reward[-100])

                if (episode % run_greedy_frequency) == 0:
                    self.run_greedy_episode()

                if episode > int(0.1*self.episodes) and int(np.mean(self.greedy_steps[-int(0.05*self.episodes):])) == self.max_steps:
                    self.greedy_cumulative_reward = [self.greedy_cumulative_reward[-1]]*self.episodes
                    self.greedy_steps = [self.greedy_steps[-1]]*self.episodes
                    break
        else:
            for _ in range(self.episodes):
                self.run_training_episode()

        self.evaluate_final_policy()

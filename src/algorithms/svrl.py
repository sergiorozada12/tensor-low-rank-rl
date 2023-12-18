import numpy as np
import time
import torch

from src.models.mf import MF


torch.set_num_threads(1)


class Svrl:
    def __init__(
        self,
        env,
        model_online,
        model_target,
        discretizer,
        episodes,
        max_steps,
        epsilon,
        alpha,
        gamma,
        buffer,
        batch_size,
        decay,
        p_mask,
        k,
        update_freq,
        iter,
        writer=None,
        prioritized_experience=False,
        alpha_mf=0.1,
        beta_mf=0.01
    ):

        self.env = env
        self.model_online = model_online
        self.model_target = model_target
        self.discretizer = discretizer

        self.episodes = episodes
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.decay = decay
        self.buffer = buffer
        self.prioritized_experience = prioritized_experience
        self.batch_size = batch_size

        self.p_mask = p_mask
        self.k = k
        self.update_freq = update_freq
        self.iter = iter
        self.alpha_mf = alpha_mf
        self.beta_mf = beta_mf

        self.training_steps = []
        self.training_cumulative_reward = []
        self.greedy_steps = []
        self.greedy_cumulative_reward = []

        self.iteration_idx = 0
        self.episode = 0

        self.writer = writer
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model_online.parameters(), lr=alpha)

        self.model_online
        self.model_target

    def get_random_action(self):
        return self.env.action_space.sample()

    def get_greedy_action(self, state):
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float)
            q_values = self.model_online.forward(state_tensor)
            action_idx = q_values.abs().argmax().item()

        if self.discretizer.discrete_action:
            return action_idx
        return self.discretizer.get_action_from_index(action_idx)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.get_random_action()
        return self.get_greedy_action(state)

    def weighted_mse_loss(self, input, target, weight):
        weight = torch.as_tensor(weight, dtype=torch.float32)
        return torch.sum(weight*(input - target)**2)

    def interpolate_q_target(self, q):
        q = q.detach().numpy()
        mask = np.random.rand(*q.shape) < self.p_mask
        q_sampled = q * mask

        mf = MF(q_sampled, K=self.k, alpha=self.alpha_mf, beta=self.beta_mf, iterations=self.iter)
        _ = mf.train()
        return torch.tensor(mf.full_matrix(), dtype=torch.float32)

    def update_model(self):
        if len(self.buffer) <= self.batch_size:
            return

        if self.prioritized_experience:
            sample, weights, batch_idxes = self.buffer.sample_batch(self.batch_size, 1.0)
        else:
            sample = self.buffer.sample_batch(self.batch_size)

        self.model_online.zero_grad()

        state = torch.stack([s.state for s in sample])
        next_state = torch.stack([s.next_state for s in sample])
        action_idx = torch.as_tensor([self.discretizer.get_action_index(s.action)[0] for s in sample], dtype=torch.int64).unsqueeze(1)
        reward = torch.as_tensor([s.reward for s in sample], dtype=torch.float32)
        done_mask = torch.as_tensor([0 if s.done else 1 for s in sample], dtype=torch.float32)

        q = torch.squeeze(self.model_online.forward(state).gather(1, action_idx))
        q_next = torch.squeeze(self.model_target.forward(state))
        _, action_next_idx = q_next_interpolated = self.interpolate_q_target(q_next).max(dim=1, keepdim=True)

        q_next = torch.squeeze(self.model_online.forward(next_state).gather(dim=1, index=action_next_idx))*done_mask
        q_target = reward + self.gamma*q_next

        if self.prioritized_experience:
            loss = self.weighted_mse_loss(q, q_target, weights)
        else:
            loss = self.criterion(q, q_target)
        loss.backward()
        self.optimizer.step()

        for p1, p2 in zip(self.model_target.parameters(), self.model_online.parameters()):
            p1.data.copy_(.001*p2.data + (1 - .001)*p1.data)

        if self.prioritized_experience:
            td_error = torch.flatten(q_target - q).tolist()
            new_priorities = np.abs(td_error) + 1e-6
            self.buffer.update_priorities(batch_idxes, new_priorities)

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
                self.iteration_idx += 1
                state_tensor = torch.tensor(state, dtype=torch.float32, requires_grad=False)
                state_prime_tensor = torch.tensor(state_prime, dtype=torch.float32, requires_grad=False)
                self.buffer.push(state_tensor, action, state_prime_tensor, reward, done)

                if self.iteration_idx % self.update_freq == 0 and step != 0:
                    self.update_model()

            if done:
                break

            state = state_prime

            if (not is_greedy) & is_train:
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

    def train(self, run_greedy_frequency=1):
        if run_greedy_frequency:
            while self.episode < self.episodes:
                self.run_training_episode()

                if self.episode > int(0.1*self.episodes) and int(np.mean(self.greedy_steps[-int(0.05*self.episodes):])) == self.max_steps:
                    self.greedy_cumulative_reward = [self.greedy_cumulative_reward[-1]]*int(self.episodes/run_greedy_frequency)
                    self.greedy_steps = [self.greedy_steps[-1]]*int(self.episodes/run_greedy_frequency)
                    break

                if (self.episode % run_greedy_frequency) == 0:
                    self.run_greedy_episode()

                self.episode += 1

        else:
            for _ in range(self.episodes):
                self.run_training_episode()

        self.evaluate_final_policy()

        if self.writer:
            self.writer.flush()

    def measure_mean_runtime(self):
        state = self.env.reset()
        action = self.choose_action(state)
        state_prime, reward, done, _ = self.env.step(action)
        if len(state_prime.shape) > 1:
            state_prime = state_prime.flatten()

        for _ in range(100):
            state_tensor = torch.tensor(state, dtype=torch.float32, requires_grad=False)
            state_prime_tensor = torch.tensor(state_prime, dtype=torch.float32, requires_grad=False)
            self.buffer.push(state_tensor, action, state_prime_tensor, reward, done)

        start_time = time.time()
        for _ in range(100_000):
            self.update_model()
        end_time = time.time()
        return end_time - start_time

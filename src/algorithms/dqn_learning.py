import numpy as np
import torch


class DqnLearning:
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
        writer=None,
        prioritized_experience=False,
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

        self.training_steps = []
        self.training_cumulative_reward = []
        self.greedy_steps = []
        self.greedy_cumulative_reward = []

        self.iteration_idx = 0

        self.writer = writer
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model_online.parameters(), lr=alpha)

    def get_random_action(self):
        return self.env.action_space.sample()

    def get_greedy_action(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float)
            q_values = self.model_online.forward(state_tensor)
            action_idx = q_values.abs().argmax().item()
        return self.discretizer.get_action_from_index(action_idx)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.get_random_action()
        return self.get_greedy_action(state)

    def write_training_metrics(self, loss, q):
        if not self.writer:
            return

        if (self.iteration_idx % 10000) == 0:
            for tag, value in self.model_online.named_parameters():
                if value.grad is not None:
                    self.writer.add_histogram(tag + "/grad", value.grad.cpu(), self.iteration_idx)
                    self.writer.add_histogram(tag + "/weight", value, self.iteration_idx)

        self.writer.add_scalar("Loss/train", loss, self.iteration_idx)
        self.writer.add_histogram("Q/train", q, self.iteration_idx)
        self.iteration_idx += 1

    def write_env_metrics_train(self, episode):
        if not self.writer:
            return

        self.writer.add_scalar("Reward/train", self.training_cumulative_reward[-1], episode)
        self.writer.add_scalar("Steps/train", self.training_steps[-1], episode)

    def write_env_metrics_greedy(self, episode):
        if not self.writer:
            return

        self.writer.add_scalar("Reward/greedy", self.greedy_cumulative_reward[-1], episode)
        self.writer.add_scalar("Steps/greedy", self.greedy_steps[-1], episode)

    def weighted_mse_loss(self, input, target, weight):
        weight = torch.tensor(weight, requires_grad=False, dtype=torch.float32)
        return torch.sum(weight*(input - target)**2)

    def update_model(self):
        if len(self.buffer) < self.batch_size:
            return

        if self.prioritized_experience:
            sample, weights, batch_idxes = self.buffer.sample_batch(self.batch_size, 1.0)
        else:
            sample = self.buffer.sample_batch(self.batch_size)

        self.optimizer.zero_grad()

        state = torch.stack([s.state for s in sample])
        next_state = torch.stack([s.next_state for s in sample])
        action_idx = torch.tensor([self.discretizer.get_action_index(s.action)[0] for s in sample], dtype=torch.int64, requires_grad=False).unsqueeze(1)
        reward = torch.tensor([s.reward for s in sample], dtype=torch.float32, requires_grad=False)
        done_mask = torch.tensor([0 if s.done else 1 for s in sample], dtype=torch.float32, requires_grad=False)

        q = torch.squeeze(self.model_online.forward(state).gather(1, action_idx))
        _, action_next_idx = self.model_target.forward(next_state).max(dim=1, keepdim=True)
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

        self.write_training_metrics(loss, q_next)

        if self.prioritized_experience:
            td_error = torch.flatten(q_target - q).tolist()
            new_priorities = np.abs(td_error) + 1e-6
            self.buffer.update_priorities(batch_idxes, new_priorities)

    def run_episode(self, is_train=True, is_greedy=False):
        state = self.env.reset()
        cumulative_reward = 0

        for step in range(self.max_steps):
            action = self.get_greedy_action(state) if is_greedy else self.choose_action(state)
            state_prime, reward, done, _ = self.env.step(action)
            cumulative_reward += reward

            if is_train:
                state_tensor = torch.tensor(state, dtype=torch.float32, requires_grad=False)
                state_prime_tensor = torch.tensor(state_prime, dtype=torch.float32, requires_grad=False)
                self.buffer.push(state_tensor, action, state_prime_tensor, reward, done)
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

    def train(self, run_greedy_frequency=None):
        if run_greedy_frequency:
            for episode in range(self.episodes):
                self.run_training_episode()
                self.write_env_metrics_train(episode)

                if (episode % run_greedy_frequency) == 0:
                    self.run_greedy_episode()
                    self.write_env_metrics_greedy(episode)

        else:
            for _ in range(self.episodes):
                self.run_training_episode()
                self.write_env_metrics(episode)

        if self.writer:
            self.writer.flush()

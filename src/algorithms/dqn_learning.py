import numpy as np
import torch


class DqnLearning:
    def __init__(
        self,
        env,
        model,
        writer,
        discretizer,
        episodes,
        max_steps,
        epsilon,
        alpha,
        gamma,
        buffer,
        batch_size,
        decay,
    ):

        self.env = env
        self.buffer = buffer
        self.model = model
        self.discretizer = discretizer

        self.episodes = episodes
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.decay = decay
        self.buffer = buffer
        self.batch_size = batch_size

        self.training_steps = []
        self.training_cumulative_reward = []
        self.greedy_steps = []
        self.greedy_cumulative_reward = []

        self.iteration_idx = 0

        self.writer = writer
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=alpha)

    def get_random_action(self):
        return self.env.action_space.sample()

    def get_greedy_action(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float)
            q_values = self.model.forward(state_tensor)
            action_idx = q_values.abs().argmax().item()
        return self.discretizer.get_action_from_index(action_idx)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.get_random_action()
        return self.get_greedy_action(state)

    def update_model(self):
        if len(self.buffer) < self.batch_size:
            return

        sample = self.buffer.sample_batch(self.batch_size)
        self.optimizer.zero_grad()
        
        state = torch.stack([s.state for s in sample])
        next_state = torch.stack([s.next_state for s in sample])
        action_idx = torch.tensor([self.discretizer.get_action_index(s.action)[0] for s in sample], dtype=torch.int64, requires_grad=False).unsqueeze(1)
        #print(action_idx)
        reward = torch.tensor([s.reward for s in sample], dtype=torch.float32, requires_grad=False)
        done_mask = torch.tensor([0 if s.done else 1 for s in sample], dtype=torch.float32, requires_grad=False)
        
        q = torch.squeeze(self.model.forward(state).gather(1, action_idx))
        q_next = self.model.forward(next_state).amax(dim=1)*done_mask
        q_target = reward + self.gamma*q_next

        loss = self.criterion(q, q_target)
        loss.backward()
        self.optimizer.step()
        #print(list(self.model.layers[0].parameters())[0][0, :])

        self.writer.add_scalar("Loss/train", loss, self.iteration_idx)
        self.writer.flush()
        self.iteration_idx += 1

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

                if (episode % run_greedy_frequency) == 0:
                    self.run_greedy_episode()
        else:
            for _ in range(self.episodes):
                self.run_training_episode()

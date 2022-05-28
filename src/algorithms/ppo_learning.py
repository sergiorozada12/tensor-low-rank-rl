import numpy as np
import torch


class PpoLearning:
    def __init__(
        self,
        env,
        model_actor,
        model_critic,
        discretizer,
        episodes,
        max_steps,
        gamma,
        clip_param,
        writer=None
    ):

        self.env = env
        self.model_actor = model_actor
        self.model_critic = model_critic
        self.discretizer = discretizer

        self.episodes = episodes
        self.max_steps = max_steps
        self.gamma = gamma
        self.clip_param = clip_param

        self.training_steps = []
        self.training_cumulative_reward = []
        self.greedy_steps = []
        self.greedy_cumulative_reward = []

        self.iteration_idx = 0

        self.writer = writer
        self.criterion_actor = torch.nn.MSELoss()
        self.criterion_critic = torch.nn.MSELoss()
        self.optimizer_actor = torch.optim.Adam(self.model_actor.parameters(), lr=7e-3)
        self.optimizer_critic = torch.optim.Adam(self.model_critic.parameters(), lr=7e-3)

    def get_action(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float)
            probs = self.model_actor.forward(state_tensor)
            action_idx = np.random.random_sample(probs)
        return self.discretizer.get_action_from_index(action_idx), probs

    def update_actor(
        self,
        states,
        actions,
        rewards,
        dones,
        values
    ):
        g = 0
        lmbda = 0.95
        returns = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma*values[i + 1]*(1 - dones[i]) - values[i]
            g = delta + self.gamma* lmbda*(1 - dones[i])*g
            returns.append(g + values[i])

        returns.reverse()
        adv = np.array(returns, dtype=np.float32) - values[:-1]
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        returns = np.array(returns, dtype=np.float32)
        return states, actions, returns, adv

    def update_models(
        self,
        states,
        rewards,
        actions,
        probs,
        dones,
        values
    ):

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

        states, rewards, actions, probs, dones, values = [], [], [], [], [], []

        for step in range(self.max_steps):
            action, p = self.get_action(state)
            value = self.model_critic(state)

            state_prime, reward, done, _ = self.env.step(action)
            cumulative_reward += reward

            states.append(state)
            rewards.append(reward)
            actions.append(action)
            probs.append(p)
            dones.append(done)
            values.append(value)

            if done:
                break

            state = state_prime

        if is_train:
            self.update_models(
                states,
                rewards,
                actions,
                probs,
                dones,
                values
            )

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
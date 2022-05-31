import numpy as np
import torch
import scipy


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
            logits = self.model_actor.forward(state_tensor)
            action_idx = torch.distributions.categorical.Categorical(logits=logits).sample()
        return self.discretizer.get_action_from_index(action_idx), logits

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
        dones,
        logits,
    ):

        with torch.no_grad():
            values = self.model_critic(torch.tensor(states))

        deltas = rewards[:-1] + self.gamma*values[1:] - values[:-1]
        advantages = scipy.signal.lfilter([1], [1, float(-self.gamma*self.lmda)], deltas[::-1], axis=0)[::-1]
        advantage_mean, advantage_std = (np.mean(advantages), np.std(advantages),)
        advantages = (advantages - advantage_mean) / advantage_std

        logprobabilities = torch.nn.LogSoftmax(logits).gather(1, actions)

        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            mask = 1 - dones[i]
            delta = rewards[i] + self.gamma*values[i + 1]*mask - values[i]
            gae = delta + self.gamma*self.lmbda*mask*gae
            returns.insert(0, gae + values[i])

        adv = np.array(returns) - values[:-1]
        advantages = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

        newpolicy_probs = y_pred
        ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))
        critic_loss = K.mean(K.square(rewards - values))
        total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * K.mean(
            -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
        return total_loss

        


        buf_logprob = -(buf_noise.pow(2).__mul__(0.5) + 
                      self.act.a_std_log +                              
                      self.act.sqrt_2pi_log).sum(1) 
        # compute the reward and advantage
        buf_r_sum, buf_advantage = self.compute_reward(buf_len, 
        buf_reward, buf_mask, buf_value) 

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

    def run_episode(self, is_train=True):
        state = self.env.reset()
        cumulative_reward = 0

        states, rewards, actions, dones, logits = [], [], [], [], []

        for step in range(self.max_steps):
            action, logit = self.get_action(state)

            state_prime, reward, done, _ = self.env.step(action)
            cumulative_reward += reward

            states.append(state)
            rewards.append(reward)
            actions.append(action)
            dones.append(done)
            logits.append(logit)

            if done:
                break

            state = state_prime

        if is_train:
            self.update_models(
                states,
                rewards,
                actions,
                dones,
                logits,
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
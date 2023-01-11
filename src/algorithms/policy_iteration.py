import numpy as np


class PolicyIteration:
    def __init__(self, env, max_iter, k, gamma):
        self.env = env
        self.max_iter = max_iter
        self.k = k
        self.gamma = gamma

        self.nA, self.nS = self.env.action_space.n, self.env.observation_space.n
        self.P, self.R = self.get_model()

    def run(self):
        policy = (1.0/self.nA)*np.ones([self.nS, self.nA])
        policy_opt, value_function_opt = self.policy_iteration(policy)
        q = np.einsum('ijk,ijk->ij', self.P, self.R + self.gamma * value_function_opt[None,None,:])
        return q

    def get_model(self):
        P = np.zeros([self.nS, self.nA, self.nS])
        R = np.zeros([self.nS, self.nA, self.nS])
        for s in range(self.nS):
            for a in range(self.nA):
                transitions = self.env.P[s][a]
                for p_trans, next_s, rew, _ in transitions:
                    P[s, a, next_s] += p_trans
                    R[s, a, next_s] = rew
                P[s, a, :] /= np.sum(P[s, a, :])
        return P, R

    def policy_iteration(self, policy):
        for _ in range(self.max_iter):
            policy = policy.copy()

            v = self.policy_evaluation(policy)
            policy_prime = self.policy_improvement(v)

            if np.array_equal(policy_prime, policy):
                return policy, v

            policy = policy_prime.copy()

    def policy_evaluation(self, policy):
        mean_R = self.calculate_mean_reward(policy)
        mean_P = self.calculate_mean_transition(policy)

        value_function = np.zeros(mean_R.shape)

        for i in range(self.k):
            value_function = mean_R + self.gamma * np.dot(mean_P, value_function)

        return value_function

    def policy_improvement(self, v):
        q = np.einsum('ijk,ijk->ij', self.P, self.R + self.gamma * v[None,None,:])
        policy = np.argmax(q, axis=1)
        return policy

    def vectorize_policy(self, policy):
        new_policy = np.zeros([self.nS, self.nA])
        for s in range(self.nS):
            new_policy[s, policy[s]] = 1.0
        return new_policy

    def calculate_mean_reward(self, policy):
        if len(policy.shape) == 1:
            policy = self.vectorize_policy(policy)
        return np.einsum('ijk,ijk,ij ->i', self.R, self.P, policy)

    def calculate_mean_transition(self, policy):
        if(len(policy.shape)==1):
            policy = self.vectorize_policy(policy)
        return np.einsum('ijk,ij -> ik', self.P, policy)


class PolicyIterationClassic:
    def __init__(self, env, max_iter, k, gamma):
        self.env = env
        self.max_iter = max_iter
        self.k = k
        self.gamma = gamma

        self.nA, self.nS = len(env.actions()), len(env.states())
        self.P, self.R = self.get_model()

    def run(self):
        policy = (1.0/self.nA)*np.ones([self.nS, self.nA])
        policy_opt, value_function_opt = self.policy_iteration(policy)
        q = np.einsum('ijk,ijk->ij', self.P, self.R + self.gamma * value_function_opt[None,None,:])
        return q

    def get_model(self):
        P = np.zeros([self.nS, self.nA, self.nS])
        R = np.zeros([self.nS, self.nA, self.nS])
        for s in range(self.nS):
            for a in range(self.nA):
                states, rewards, dones, probabilities = self.env.model(s, a)
                transitions = [(states[i], rewards[i], dones[i], probabilities[i]) for i in range(states.size)]
                for next_s, rew, done, p_trans in transitions:
                    P[s, a, next_s] += p_trans
                    R[s, a, next_s] = rew
                P[s, a, :] /= np.sum(P[s, a, :])
        return P, R

    def policy_iteration(self, policy):
        for _ in range(self.max_iter):
            policy = policy.copy()

            v = self.policy_evaluation(policy)
            policy_prime = self.policy_improvement(v)

            if np.array_equal(policy_prime, policy):
                return policy, v

            policy = policy_prime.copy()

    def policy_evaluation(self, policy):
        mean_R = self.calculate_mean_reward(policy)
        mean_P = self.calculate_mean_transition(policy)

        value_function = np.zeros(mean_R.shape)

        for i in range(self.k):
            value_function = mean_R + self.gamma * np.dot(mean_P, value_function)

        return value_function

    def policy_improvement(self, v):
        q = np.einsum('ijk,ijk->ij', self.P, self.R + self.gamma * v[None,None,:])
        policy = np.argmax(q, axis=1)
        return policy

    def vectorize_policy(self, policy):
        new_policy = np.zeros([self.nS, self.nA])
        for s in range(self.nS):
            new_policy[s, policy[s]] = 1.0
        return new_policy

    def calculate_mean_reward(self, policy):
        if len(policy.shape) == 1:
            policy = self.vectorize_policy(policy)
        return np.einsum('ijk,ijk,ij ->i', self.R, self.P, policy)

    def calculate_mean_transition(self, policy):
        if(len(policy.shape)==1):
            policy = self.vectorize_policy(policy)
        return np.einsum('ijk,ij -> ik', self.P, policy)
import os
import json
import pickle
from platform import architecture

from pathos.multiprocessing import ProcessingPool as Pool

import numpy as np
import pandas as pd

from src.models.mlp import Mlp

from src.algorithms.q_learning import QLearning
from src.algorithms.dqn_learning import DqnLearning
from src.algorithms.svrl import Svrl
from src.algorithms.mlr_learning import MatrixLowRankLearning
from src.algorithms.tlr_learning import TensorLowRankLearning

from src.utils.utils import Discretizer, PrioritizedReplayBuffer, ReplayBuffer


class Experiment:
    def __init__(self, name, env, nodes, recover=False, run_freq=10):
        self.env = env
        self.nodes = nodes
        self.name = name
        self.run_freq = run_freq

        with open(f'parameters/{name}', 'r') as f:
            self.parameters = json.load(f)

        self.discretizer = self._get_discretizer()

        if recover:
            self.models = self._get_models_from_checkpoints()
        else:
            self.models = self._get_models()

    def _get_discretizer(self):
        states_structure = self.parameters.get('states_structure', None)
        discrete_action = self.parameters.get('discrete_action', False)
        return Discretizer(
            min_points_states=self.parameters['min_points_states'],
            max_points_states=self.parameters['max_points_states'],
            bucket_states=self.parameters['bucket_states'],
            min_points_actions=self.parameters['min_points_actions'],
            max_points_actions=self.parameters['max_points_actions'],
            bucket_actions=self.parameters['bucket_actions'],
            states_structure=states_structure,
            discrete_action=discrete_action
        )

    def _get_models(self):
        if self.parameters['type'] == 'q-model':
            return self._get_q_models()
        elif self.parameters['type'] == 'mlr-model':
            return self._get_mlr_models()
        elif self.parameters['type'] == 'tlr-model':
            return self._get_tlr_models()
        elif self.parameters['type'] == 'dqn-model':
            return self._get_dqn_models()
        elif self.parameters['type'] == 'svrl-model':
            return self._get_svrl_models()
        return None

    def _get_models_from_checkpoints(self):
        models = []
        for path in os.listdir('nn_checkpoints'):

            with open(os.path.join('nn_checkpoints', path), 'rb') as f:
                model = pickle.load(f)

            if len(model.training_steps) > len(model.greedy_steps):
                model.training_steps.pop()
                model.training_cumulative_reward.pop()
            elif len(model.training_steps) > model.episode:
                model.episode += 1

            models.append(model)

            os.remove(os.path.join('nn_checkpoints', path))

    def _get_q_models(self):
        return [QLearning(
            env=self.env,
            discretizer=self.discretizer,
            episodes=self.parameters['episodes'],
            max_steps=self.parameters['max_steps'],
            epsilon=self.parameters['epsilon'],
            alpha=self.parameters['alpha'],
            gamma=self.parameters['gamma'],
            decay=self.parameters['decay'],
        ) for _ in range(self.nodes)]

    def _get_mlr_models(self):
        return [MatrixLowRankLearning(
            env=self.env,
            discretizer=self.discretizer,
            episodes=self.parameters['episodes'],
            max_steps=self.parameters['max_steps'],
            epsilon=self.parameters['epsilon'],
            alpha=self.parameters['alpha'],
            gamma=self.parameters['gamma'],
            decay=self.parameters['decay'],
            k=self.parameters['k']
        ) for _ in range(self.nodes)]

    def _get_tlr_models(self):
        bias = self.parameters.get('bias', 0.0)
        return [TensorLowRankLearning(
            env=self.env,
            discretizer=self.discretizer,
            episodes=self.parameters['episodes'],
            max_steps=self.parameters['max_steps'],
            epsilon=self.parameters['epsilon'],
            alpha=self.parameters['alpha'],
            gamma=self.parameters['gamma'],
            decay=self.parameters['decay'],
            k=self.parameters['k'],
            bias=bias
        ) for _ in range(self.nodes)]

    def _get_dqn_models(self):
        model_online = Mlp(
            len(self.parameters['bucket_states']),
            self.parameters['arch'],
            self.discretizer.n_actions[0]
        )

        model_target = Mlp(
            len(self.parameters['bucket_states']),
            self.parameters['arch'],
            self.discretizer.n_actions[0]
        )

        model_target.load_state_dict(model_online.state_dict())

        if self.parameters['prioritized_experience']:
            buffer = PrioritizedReplayBuffer(self.parameters['buffer_size'], 1.0)
        else:
            buffer = ReplayBuffer(self.parameters['buffer_size'])

        return [DqnLearning(
            env=self.env,
            discretizer=self.discretizer,
            model_online=model_online,
            model_target=model_target,
            buffer=buffer,
            batch_size=self.parameters['batch_size'],
            episodes=self.parameters['episodes'],
            max_steps=self.parameters['max_steps'],
            epsilon=self.parameters['epsilon'],
            alpha=self.parameters['alpha'],
            gamma=self.parameters['gamma'],
            decay=self.parameters['decay'],
            prioritized_experience=self.parameters['prioritized_experience'],
        ) for _ in range(self.nodes)]

    def _get_svrl_models(self):
        model_online = Mlp(
            len(self.parameters['bucket_states']),
            self.parameters['arch'],
            self.discretizer.n_actions[0]
        )

        model_target = Mlp(
            len(self.parameters['bucket_states']),
            self.parameters['arch'],
            self.discretizer.n_actions[0]
        )

        model_target.load_state_dict(model_online.state_dict())

        if self.parameters['prioritized_experience']:
            buffer = PrioritizedReplayBuffer(self.parameters['buffer_size'], 1.0)
        else:
            buffer = ReplayBuffer(self.parameters['buffer_size'])

        return [Svrl(
            env=self.env,
            discretizer=self.discretizer,
            model_online=model_online,
            model_target=model_target,
            buffer=buffer,
            batch_size=self.parameters['batch_size'],
            episodes=self.parameters['episodes'],
            max_steps=self.parameters['max_steps'],
            epsilon=self.parameters['epsilon'],
            alpha=self.parameters['alpha'],
            gamma=self.parameters['gamma'],
            decay=self.parameters['decay'],
            prioritized_experience=self.parameters['prioritized_experience'],
            p_mask=self.parameters['p_mask'],
            update_freq=self.parameters['update_freq'],
            iter=self.parameters['iter'],
            k=self.parameters['k'],
            alpha_mf=self.parameters['alpha_mf'],
            beta_mf=self.parameters['beta_mf']
        ) for _ in range(self.nodes)]

    def run_experiment(self, learner):
        learner.train(run_greedy_frequency=self.run_freq)
        return learner

    def run_experiments(self, window):
        with Pool(self.nodes) as pool:
            models = pool.map(self.run_experiment, self.models)

        steps = []
        max_length = self.parameters['episodes']
        for model in models:
            s = model.greedy_steps
            steps.append(pd.Series(s).rolling(window).median())

        steps = np.median(steps, axis=0)
        rewards = [learner.mean_reward for learner in models]
        data = {'steps': list(steps), 'rewards': rewards}

        with open(f'results/{self.name}', 'w') as f:
            json.dump(data, f)

        for path in os.listdir('nn_checkpoints'):
            os.remove(os.path.join('nn_checkpoints', path))


class ExperimentScale:
    def __init__(self, name, env, nodes, run_freq=10):
        self.env = env
        self.nodes = nodes
        self.name = name
        self.run_freq = run_freq

        with open(f'parameters/{name}', 'r') as f:
            self.parameters = json.load(f)

    def _get_discretizer(self, bucket_actions):
        states_structure = self.parameters.get('states_structure', None)
        discrete_action = self.parameters.get('discrete_action', False)
        return Discretizer(
            min_points_states=self.parameters['min_points_states'],
            max_points_states=self.parameters['max_points_states'],
            bucket_states=self.parameters['bucket_states'],
            min_points_actions=self.parameters['min_points_actions'],
            max_points_actions=self.parameters['max_points_actions'],
            bucket_actions=bucket_actions,
            states_structure=states_structure,
            discrete_action=discrete_action
        )

    def _get_q_models(self):
        return [QLearning(
            env=self.env,
            discretizer=self.discretizer,
            episodes=self.parameters['episodes'],
            max_steps=self.parameters['max_steps'],
            epsilon=self.parameters['epsilon'],
            alpha=self.parameters['alpha'],
            gamma=self.parameters['gamma'],
            decay=self.parameters['decay'],
        ) for _ in range(self.nodes)]

    def _get_mlr_models(self, k):
        return [MatrixLowRankLearning(
            env=self.env,
            discretizer=self.discretizer,
            episodes=self.parameters['episodes'],
            max_steps=self.parameters['max_steps'],
            epsilon=self.parameters['epsilon'],
            alpha=self.parameters['alpha'],
            gamma=self.parameters['gamma'],
            decay=self.parameters['decay'],
            k=k
        ) for _ in range(self.nodes)]

    def _get_tlr_models(self, k):
        bias = self.parameters.get('bias', 0.0)
        return [TensorLowRankLearning(
            env=self.env,
            discretizer=self.discretizer,
            episodes=self.parameters['episodes'],
            max_steps=self.parameters['max_steps'],
            epsilon=self.parameters['epsilon'],
            alpha=self.parameters['alpha'],
            gamma=self.parameters['gamma'],
            decay=self.parameters['decay'],
            k=k,
            bias=bias
        ) for _ in range(self.nodes)]

    def _get_dqn_models(self, architecture):
        model_online = Mlp(
            len(self.parameters['bucket_states']),
            architecture,
            self.discretizer.n_actions[0]
        )

        model_target = Mlp(
            len(self.parameters['bucket_states']),
            architecture,
            self.discretizer.n_actions[0]
        )

        model_target.load_state_dict(model_online.state_dict())

        if self.parameters['prioritized_experience']:
            buffer = PrioritizedReplayBuffer(self.parameters['buffer_size'], 1.0)
        else:
            buffer = ReplayBuffer(self.parameters['buffer_size'])

        return [DqnLearning(
            env=self.env,
            discretizer=self.discretizer,
            model_online=model_online,
            model_target=model_target,
            buffer=buffer,
            batch_size=self.parameters['batch_size'],
            episodes=self.parameters['episodes'],
            max_steps=self.parameters['max_steps'],
            epsilon=self.parameters['epsilon'],
            alpha=self.parameters['alpha'],
            gamma=self.parameters['gamma'],
            decay=self.parameters['decay'],
            prioritized_experience=self.parameters['prioritized_experience'],
        ) for _ in range(self.nodes)]

    def _get_svrl_models(self, architecture):
        model_online = Mlp(
            len(self.parameters['bucket_states']),
            architecture,
            self.discretizer.n_actions[0]
        )

        model_target = Mlp(
            len(self.parameters['bucket_states']),
            architecture,
            self.discretizer.n_actions[0]
        )

        model_target.load_state_dict(model_online.state_dict())

        if self.parameters['prioritized_experience']:
            buffer = PrioritizedReplayBuffer(self.parameters['buffer_size'], 1.0)
        else:
            buffer = ReplayBuffer(self.parameters['buffer_size'])

        return [Svrl(
            env=self.env,
            discretizer=self.discretizer,
            model_online=model_online,
            model_target=model_target,
            buffer=buffer,
            batch_size=self.parameters['batch_size'],
            episodes=self.parameters['episodes'],
            max_steps=self.parameters['max_steps'],
            epsilon=self.parameters['epsilon'],
            alpha=self.parameters['alpha'],
            gamma=self.parameters['gamma'],
            decay=self.parameters['decay'],
            prioritized_experience=self.parameters['prioritized_experience'],
            p_mask=self.parameters['p_mask'],
            update_freq=self.parameters['update_freq'],
            iter=self.parameters['iter'],
            k=self.parameters['k'],
            alpha_mf=self.parameters['alpha_mf'],
            beta_mf=self.parameters['beta_mf']
        ) for _ in range(self.nodes)]

    def run_experiment(self, learner):
        learner.train(run_greedy_frequency=self.run_freq)
        return learner

    def _run_q_experiments(self):
        
        results = {
            'parameters': [],
            'reward': [],
            'reward_std': []
        }

        for bucket_actions in self.parameters['bucket_actions']:
            self.discretizer = self._get_discretizer(bucket_actions)
            models = self._get_q_models()

            with Pool(self.nodes) as pool:
                trained_models = pool.map(self.run_experiment, models)

            rewards = np.mean([learner.mean_reward for learner in trained_models])
            rewards_std = np.std([learner.mean_reward for learner in trained_models])
            params = trained_models[0].Q.size

            results['parameters'].append(params)
            results['reward'].append(rewards)
            results['reward_std'].append(rewards_std)

        with open(f'results/{self.name}', 'w') as f:
            json.dump(results, f)

    def _run_mlr_experiments(self):
        results = {
            'parameters': [],
            'reward': [],
            'reward_std': []
        }

        self.discretizer = self._get_discretizer(self.parameters['bucket_actions'])
        for k in self.parameters['k']:
            models = self._get_mlr_models(k)

            with Pool(self.nodes) as pool:
                trained_models = pool.map(self.run_experiment, models)

            rewards = np.mean([learner.mean_reward for learner in trained_models])
            rewards_std = np.std([learner.mean_reward for learner in trained_models])
            params = trained_models[0].L.size + trained_models[0].R.size

            results['parameters'].append(params)
            results['reward'].append(rewards)
            results['reward_std'].append(rewards_std)

        with open(f'results/{self.name}', 'w') as f:
            json.dump(results, f)

    def _run_tlr_experiments(self):
        
        results = {
            'parameters': [],
            'reward': [],
            'reward_std': []
        }

        self.discretizer = self._get_discretizer(self.parameters['bucket_actions'])
        for k in self.parameters['k']:
            models = self._get_tlr_models(k)

            with Pool(self.nodes) as pool:
                trained_models = pool.map(self.run_experiment, models)

            rewards = np.mean([learner.mean_reward for learner in trained_models])
            rewards_std = np.std([learner.mean_reward for learner in trained_models])
            params = sum([factor.size for factor in trained_models[0].factors])

            results['parameters'].append(params)
            results['reward'].append(rewards)
            results['reward_std'].append(rewards_std)

        with open(f'results/{self.name}', 'w') as f:
            json.dump(results, f)

    def _run_dqn_experiments(self):

        results = {
            'parameters': [],
            'reward': [],
            'reward_std': []
        }

        self.discretizer = self._get_discretizer(self.parameters['bucket_actions'])
        for arch in self.parameters['arch']:
            models = self._get_dqn_models(arch)

            with Pool(self.nodes) as pool:
                trained_models = pool.map(self.run_experiment, models)

            rewards = [learner.mean_reward for learner in trained_models]
            params = sum(arch)

            results['parameters'].append(params)
            results['rewards'].append(rewards)

        with open(f'results/{self.name}', 'w') as f:
            json.dump(results, f)

    def _run_svrl_experiments(self):

        results = {
            'parameters': [],
            'reward': [],
            'reward_std': []
        }

        self.discretizer = self._get_discretizer(self.parameters['bucket_actions'])
        for arch in self.parameters['arch']:
            models = self._get_svrl_models(arch)

            with Pool(self.nodes) as pool:
                trained_models = pool.map(self.run_experiment, models)

            rewards = [learner.mean_reward for learner in trained_models]
            params = sum(arch)

            results['parameters'].append(params)
            results['rewards'].append(rewards)

        with open(f'results/{self.name}', 'w') as f:
            json.dump(results, f)

    def run_experiments(self):
        if self.parameters['type'] == 'q-model':
            self._run_q_experiments()
        elif self.parameters['type'] == 'mlr-model':
            self._run_mlr_experiments()
        elif self.parameters['type'] == 'tlr-model':
            self._run_tlr_experiments()
        elif self.parameters['type'] == 'dqn-model':
            self._run_dqn_experiments()
        elif self.parameters['type'] == 'svrl-model':
            self._run_svrl_experiments()


class ExperimentHighway:
    def __init__(self, name, env, nodes, recover=False, run_freq=10):
        self.env = env
        self.nodes = nodes
        self.name = name
        self.run_freq = run_freq

        with open(f'parameters/{name}', 'r') as f:
            self.parameters = json.load(f)

        self.discretizer = self._get_discretizer()

        if recover:
            self.models = self._get_models_from_checkpoints()
        else:
            self.models = self._get_models()

    def _get_discretizer(self):
        states_structure = self.parameters.get('states_structure', None)
        discrete_action = self.parameters.get('discrete_action', False)
        return Discretizer(
            min_points_states=self.parameters['min_points_states'],
            max_points_states=self.parameters['max_points_states'],
            bucket_states=self.parameters['bucket_states'],
            min_points_actions=self.parameters['min_points_actions'],
            max_points_actions=self.parameters['max_points_actions'],
            bucket_actions=self.parameters['bucket_actions'],
            states_structure=states_structure,
            discrete_action=discrete_action
        )

    def _get_models(self):
        if self.parameters['type'] == 'q-model':
            return self._get_q_models()
        elif self.parameters['type'] == 'mlr-model':
            return self._get_mlr_models()
        elif self.parameters['type'] == 'tlr-model':
            return self._get_tlr_models()
        elif self.parameters['type'] == 'dqn-model':
            return self._get_dqn_models()
        return None

    def _get_models_from_checkpoints(self):
        models = []
        for path in os.listdir('nn_checkpoints'):

            with open(os.path.join('nn_checkpoints', path), 'rb') as f:
                model = pickle.load(f)

            if len(model.training_steps) > len(model.greedy_steps):
                model.training_steps.pop()
                model.training_cumulative_reward.pop()
            elif len(model.training_steps) > model.episode:
                model.episode += 1

            models.append(model)

            os.remove(os.path.join('nn_checkpoints', path))

    def _get_q_models(self):
        return [QLearning(
            env=self.env,
            discretizer=self.discretizer,
            episodes=self.parameters['episodes'],
            max_steps=self.parameters['max_steps'],
            epsilon=self.parameters['epsilon'],
            alpha=self.parameters['alpha'],
            gamma=self.parameters['gamma'],
            decay=self.parameters['decay'],
        ) for _ in range(self.nodes)]

    def _get_mlr_models(self):
        return [MatrixLowRankLearning(
            env=self.env,
            discretizer=self.discretizer,
            episodes=self.parameters['episodes'],
            max_steps=self.parameters['max_steps'],
            epsilon=self.parameters['epsilon'],
            alpha=self.parameters['alpha'],
            gamma=self.parameters['gamma'],
            decay=self.parameters['decay'],
            k=self.parameters['k']
        ) for _ in range(self.nodes)]

    def _get_tlr_models(self):
        bias = self.parameters.get('bias', 0.0)
        return [TensorLowRankLearning(
            env=self.env,
            discretizer=self.discretizer,
            episodes=self.parameters['episodes'],
            max_steps=self.parameters['max_steps'],
            epsilon=self.parameters['epsilon'],
            alpha=self.parameters['alpha'],
            gamma=self.parameters['gamma'],
            decay=self.parameters['decay'],
            k=self.parameters['k'],
            bias=bias
        ) for _ in range(self.nodes)]

    def _get_dqn_models(self):
        model_online = Mlp(
            len(self.parameters['bucket_states']),
            self.parameters['arch'],
            self.discretizer.n_actions[0]
        )

        model_target = Mlp(
            len(self.parameters['bucket_states']),
            self.parameters['arch'],
            self.discretizer.n_actions[0]
        )

        model_target.load_state_dict(model_online.state_dict())

        if self.parameters['prioritized_experience']:
            buffer = PrioritizedReplayBuffer(self.parameters['buffer_size'], 1.0)
        else:
            buffer = ReplayBuffer(self.parameters['buffer_size'])

        return [DqnLearning(
            env=self.env,
            discretizer=self.discretizer,
            model_online=model_online,
            model_target=model_target,
            buffer=buffer,
            batch_size=self.parameters['batch_size'],
            episodes=self.parameters['episodes'],
            max_steps=self.parameters['max_steps'],
            epsilon=self.parameters['epsilon'],
            alpha=self.parameters['alpha'],
            gamma=self.parameters['gamma'],
            decay=self.parameters['decay'],
            prioritized_experience=self.parameters['prioritized_experience'],
        ) for _ in range(self.nodes)]

    def run_experiment(self, learner):
        learner.train(run_greedy_frequency=self.run_freq)
        return learner

    def run_experiments(self, window):
        with Pool(self.nodes) as pool:
            models = pool.map(self.run_experiment, self.models)

        rewards = np.median([pd.Series(learner.greedy_cumulative_reward).rolling(window).median() for learner in models], axis=0)
        final_rewards = [learner.mean_reward for learner in models]
        data = {'rewards': list(rewards), 'final_rewards': final_rewards}

        with open(f'results/{self.name}', 'w') as f:
            json.dump(data, f)

        for path in os.listdir('nn_checkpoints'):
            os.remove(os.path.join('nn_checkpoints', path))


class ExperimentComplexity:
    def __init__(self, name, env):
        self.env = env
        self.name = name

        with open(f'parameters/{name}', 'r') as f:
            self.parameters = json.load(f)

    def _get_discretizer(self, bucket_a):
        states_structure = self.parameters.get('states_structure', None)
        discrete_action = self.parameters.get('discrete_action', False)
        return Discretizer(
            min_points_states=self.parameters['min_points_states'],
            max_points_states=self.parameters['max_points_states'],
            bucket_states=self.parameters['bucket_states'],
            min_points_actions=self.parameters['min_points_actions'],
            max_points_actions=self.parameters['max_points_actions'],
            bucket_actions=bucket_a,
            states_structure=states_structure,
            discrete_action=discrete_action
        )

    def _get_models(self):
        if self.parameters['type'] == 'q-model':
            return self._get_q_models()
        elif self.parameters['type'] == 'mlr-model':
            return self._get_mlr_models()
        elif self.parameters['type'] == 'tlr-model':
            return self._get_tlr_models()
        elif self.parameters['type'] == 'dqn-model':
            return self._get_dqn_models()
        elif self.parameters['type'] == 'svrl-model':
            return self._get_svrl_models()
        return None

    def _get_q_models(self):
        return [QLearning(
            env=self.env,
            discretizer=self.discretizer,
            episodes=self.parameters['episodes'],
            max_steps=self.parameters['max_steps'],
            epsilon=self.parameters['epsilon'],
            alpha=self.parameters['alpha'],
            gamma=self.parameters['gamma'],
            decay=self.parameters['decay'],
        )]

    def _get_mlr_models(self):
        return [MatrixLowRankLearning(
            env=self.env,
            discretizer=self.discretizer,
            episodes=self.parameters['episodes'],
            max_steps=self.parameters['max_steps'],
            epsilon=self.parameters['epsilon'],
            alpha=self.parameters['alpha'],
            gamma=self.parameters['gamma'],
            decay=self.parameters['decay'],
            k=k
        ) for k in self.parameters['k']]

    def _get_tlr_models(self):
        bias = self.parameters.get('bias', 0.0)
        return [TensorLowRankLearning(
            env=self.env,
            discretizer=self.discretizer,
            episodes=self.parameters['episodes'],
            max_steps=self.parameters['max_steps'],
            epsilon=self.parameters['epsilon'],
            alpha=self.parameters['alpha'],
            gamma=self.parameters['gamma'],
            decay=self.parameters['decay'],
            k=k,
            bias=bias
        ) for k in self.parameters['k']]

    def _get_dqn_models(self):
        models = []
        for arch in self.parameters['arch']:
            model_online = Mlp(
                len(self.parameters['bucket_states']),
                arch,
                self.discretizer.n_actions[0]
            )

            model_target = Mlp(
                len(self.parameters['bucket_states']),
                arch,
                self.discretizer.n_actions[0]
            )

            model_target.load_state_dict(model_online.state_dict())

            if self.parameters['prioritized_experience']:
                buffer = PrioritizedReplayBuffer(self.parameters['buffer_size'], 1.0)
            else:
                buffer = ReplayBuffer(self.parameters['buffer_size'])

            models.append(DqnLearning(
                env=self.env,
                discretizer=self.discretizer,
                model_online=model_online,
                model_target=model_target,
                buffer=buffer,
                batch_size=self.parameters['batch_size'],
                episodes=self.parameters['episodes'],
                max_steps=self.parameters['max_steps'],
                epsilon=self.parameters['epsilon'],
                alpha=self.parameters['alpha'],
                gamma=self.parameters['gamma'],
                decay=self.parameters['decay'],
                prioritized_experience=self.parameters['prioritized_experience'],
            ))
        return models

    def _get_svrl_models(self):
        models = []
        for arch in self.parameters['arch']:
            model_online = Mlp(
                len(self.parameters['bucket_states']),
                arch,
                self.discretizer.n_actions[0]
            )

            model_target = Mlp(
                len(self.parameters['bucket_states']),
                arch,
                self.discretizer.n_actions[0]
            )

            model_target.load_state_dict(model_online.state_dict())

            if self.parameters['prioritized_experience']:
                buffer = PrioritizedReplayBuffer(self.parameters['buffer_size'], 1.0)
            else:
                buffer = ReplayBuffer(self.parameters['buffer_size'])

            models.append(Svrl(
                env=self.env,
                discretizer=self.discretizer,
                model_online=model_online,
                model_target=model_target,
                buffer=buffer,
                batch_size=self.parameters['batch_size'],
                episodes=self.parameters['episodes'],
                max_steps=self.parameters['max_steps'],
                epsilon=self.parameters['epsilon'],
                alpha=self.parameters['alpha'],
                gamma=self.parameters['gamma'],
                decay=self.parameters['decay'],
                prioritized_experience=self.parameters['prioritized_experience'],
                p_mask=self.parameters['p_mask'],
                update_freq=self.parameters['update_freq'],
                iter=self.parameters['iter'],
                k=self.parameters['k'],
                alpha_mf=self.parameters['alpha_mf'],
                beta_mf=self.parameters['beta_mf']
            ))
        return models

    def run_experiments(self):
        buckets_a = self.parameters['bucket_actions']
        if self.parameters['type'] != 'q-model':
            buckets_a = [buckets_a]

        for ba in buckets_a:
            self.discretizer = self._get_discretizer(ba)
            self.models= self._get_models()
            for model in self.models:
                time = model.measure_mean_runtime()
                print(f"{self.name} - {time} seconds", flush=True)
        print("----------------------------------------------------------------------", flush=True)

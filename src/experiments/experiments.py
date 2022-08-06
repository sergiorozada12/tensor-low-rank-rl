import json
from pathos.multiprocessing import ProcessingPool as Pool

import numpy as np
import pandas as pd

from src.models.mlp import Mlp

from src.algorithms.q_learning import QLearning
from src.algorithms.dqn_learning import DqnLearning
from src.algorithms.mlr_learning import MatrixLowRankLearning
from src.algorithms.tlr_learning import TensorLowRankLearning

from src.utils.utils import Discretizer, PrioritizedReplayBuffer, ReplayBuffer


class Experiment:
    def __init__(self, name, env, nodes):
        self.env = env
        self.nodes = nodes
        self.name = name

        with open(f'parameters/{name}', 'r') as f:
            self.parameters = json.load(f)

        self.discretizer = self._get_discretizer()
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

    @staticmethod
    def run_experiment(learner):
        learner.train(run_greedy_frequency=1)
        return learner

    def run_experiments(self, window):
        """with Pool(self.nodes) as pool:
            models_1 = pool.map(self.run_experiment, self.models[:50])

        with Pool(self.nodes) as pool:
            models_2 = pool.map(self.run_experiment, self.models[50:])
        
        models = models_1 + models_2"""

        with Pool(self.nodes) as pool:
            models = pool.map(self.run_experiment, self.models)

        steps = np.median([pd.Series(learner.greedy_steps).rolling(window).median() for learner in models], axis=0)
        rewards = np.median([np.mean(learner.greedy_cumulative_reward[-10:]) for learner in models])
        data = {'steps': list(steps), 'rewards': rewards}

        with open(f'results/{self.name}', 'w') as f:
            json.dump(data, f)

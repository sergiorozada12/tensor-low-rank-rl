import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
from models import QLearning, LowRankLearning

from functools import partial
import multiprocessing


class Discretizer:
    def __init__(self,
                 min_points_states,
                 max_points_states,
                 bucket_states,
                 min_points_actions,
                 max_points_actions,
                 bucket_actions):
        self.min_points_states = np.array(min_points_states)
        self.max_points_states = np.array(max_points_states)
        self.bucket_states = np.array(bucket_states)
        self.spacing_states = (self.max_points_states - self.min_points_states) / self.bucket_states
        self.range_states = self.max_points_states - self.min_points_states

        self.min_points_actions = np.array(min_points_actions)
        self.max_points_actions = np.array(max_points_actions)
        self.bucket_actions = np.array(bucket_actions)
        self.spacing_actions = (self.max_points_actions - self.min_points_actions) / self.bucket_actions
        self.range_actions = self.max_points_actions - self.min_points_actions

        self.n_states = np.round(self.bucket_states).astype(int)
        self.n_actions = np.round(self.bucket_actions).astype(int)
        self.dimensions = np.concatenate((self.n_states, self.n_actions))

    def get_state_index(self, state):
        state = np.clip(state, a_min=self.min_points_states, a_max=self.max_points_states)
        scaling = (state - self.min_points_states) / self.range_states
        state_idx = np.round(scaling * (self.bucket_states - 1)).astype(int)
        return tuple(state_idx.tolist())

    def get_action_index(self, action):
        action = np.clip(action, a_min=self.min_points_actions, a_max=self.max_points_actions)
        scaling = (action - self.min_points_actions) / self.range_actions
        action_idx = np.round(scaling * (self.bucket_actions - 1)).astype(int)
        return tuple(action_idx.tolist())

    def get_action_from_index(self, action_idx):
        return self.min_points_actions + self.spacing_actions / 2 + action_idx * self.spacing_actions


class DiscretizerReshape:
    def __init__(self,
                 min_points_states,
                 max_points_states,
                 bucket_states,
                 min_points_actions,
                 max_points_actions,
                 bucket_actions):

        self.min_points_states = np.array(min_points_states)
        self.max_points_states = np.array(max_points_states)
        self.bucket_states = np.array(bucket_states)
        self.spacing_states = (self.max_points_states - self.min_points_states) / self.bucket_states
        self.range_states = self.max_points_states - self.min_points_states

        self.min_points_actions = np.array(min_points_actions)
        self.max_points_actions = np.array(max_points_actions)
        self.bucket_actions = np.array(bucket_actions)
        self.spacing_actions = (self.max_points_actions - self.min_points_actions) / self.bucket_actions
        self.range_actions = self.max_points_actions - self.min_points_actions

        self.n_state_dimensions = np.round(self.bucket_states).astype(int)
        self.n_action_dimensions = np.round(self.bucket_actions).astype(int)

        self.dimensions = np.concatenate((self.n_state_dimensions, self.n_action_dimensions))

        n_elements = np.prod(self.dimensions)
        n_cols = np.ceil(np.sqrt(n_elements))
        n_rows = np.ceil(n_elements/n_cols)

        self.n_states = n_rows
        self.n_actions = n_cols

        indices = np.array(list(itertools.product(self.n_states, self.n_actions)))

    def get_state_index(self, state):
        state = np.clip(state, a_min=self.min_points_states, a_max=self.max_points_states)
        scaling = (state - self.min_points_states) / self.range_states
        state_idx = np.round(scaling * (self.bucket_states - 1)).astype(int)
        return tuple(state_idx.tolist())

    def get_action_index(self, action):
        action = np.clip(action, a_min=self.min_points_actions, a_max=self.max_points_actions)
        scaling = (action - self.min_points_actions) / self.range_actions
        action_idx = np.round(scaling * (self.bucket_actions - 1)).astype(int)
        return tuple(action_idx.tolist())

    def get_action_from_index(self, action_idx):
        return self.min_points_actions + self.spacing_actions / 2 + action_idx * self.spacing_actions


class Saver:
    @staticmethod
    def save_to_pickle(path, obj):
        """
        :param path: str
            Path to store the object.
        :param obj
            Object to store.
        """
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_from_pickle(path):
        """
        :param path: str
            Path of the object to load.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)


class Experiment:

    @staticmethod
    def run_q_learning_experiment(env, parameters, path_output, Q_gt=None):
        saver = Saver()
        discretizer = Discretizer(min_points_states=parameters["min_states"],
                                  max_points_states=parameters["max_states"],
                                  bucket_states=parameters["bucket_states"],
                                  min_points_actions=parameters["min_actions"],
                                  max_points_actions=parameters["max_actions"],
                                  bucket_actions=parameters["bucket_actions"])

        decay = parameters["decay"] if "decay" in parameters.keys() else 1.0
        min_epsilon = parameters["min_epsilon"] if "min_epsilon" in parameters.keys() else 1.0

        q_learner = QLearning(env=env,
                              discretizer=discretizer,
                              episodes=parameters["episodes"],
                              max_steps=parameters["max_steps"],
                              epsilon=parameters["epsilon"],
                              alpha=parameters["alpha"],
                              gamma=parameters["gamma"],
                              decay=decay,
                              min_epsilon=min_epsilon,
                              Q_ground_truth=Q_gt)

        q_learner.train(run_greedy_frequency=10)
        saver.save_to_pickle(path_output, q_learner)

    @staticmethod
    def run_lr_learning_experiment(env, parameters, path_output, Q_hat_gt=None):
        saver = Saver()
        discretizer = Discretizer(min_points_states=parameters["min_states"],
                                  max_points_states=parameters["max_states"],
                                  bucket_states=parameters["bucket_states"],
                                  min_points_actions=parameters["min_actions"],
                                  max_points_actions=parameters["max_actions"],
                                  bucket_actions=parameters["bucket_actions"])

        decay = parameters["decay"] if "decay" in parameters.keys() else 1.0
        init_ord = parameters["init_ord"] if "init_ord" in parameters.keys() else 1.0
        min_epsilon = parameters["min_epsilon"] if "min_epsilon" in parameters.keys() else 1.0

        lr_learner = LowRankLearning(env=env,
                                     discretizer=discretizer,
                                     episodes=parameters["episodes"],
                                     max_steps=parameters["max_steps"],
                                     epsilon=parameters["epsilon"],
                                     alpha=parameters["alpha"],
                                     gamma=parameters["gamma"],
                                     k=parameters["k"],
                                     decay=decay,
                                     init_ord=init_ord,
                                     min_epsilon=min_epsilon,
                                     Q_hat_ground_truth=Q_hat_gt)

        lr_learner.train(run_greedy_frequency=10)
        saver.save_to_pickle(path_output, lr_learner)

    @staticmethod
    def wrapper_parallel_ql_experiment(index_exp, env, params, path_output_base, path_ground_truth, index_buck):
        saver = Saver()

        path_output = path_output_base.format(index_buck, index_exp)
        Q_gt = saver.load_from_pickle(path_ground_truth.format(index_buck, index_exp)).Q if path_ground_truth else None
        Experiment.run_q_learning_experiment(env=env,
                                             parameters=params,
                                             path_output=path_output,
                                             Q_gt=Q_gt)

    @staticmethod
    def wrapper_parallel_lr_experiment(index_exp, env, params, path_output_base, path_ground_truth, index_buck):
        saver = Saver()

        path_output = path_output_base.format(index_buck, index_exp)
        model_gt = saver.load_from_pickle(path_ground_truth.format(i, j)) if path_ground_truth else None
        Q_hat_gt = model_gt.L @ model_gt.R if model_gt else None

        Experiment.run_lr_learning_experiment(env=env,
                                              parameters=params,
                                              path_output=path_output,
                                              Q_hat_gt=Q_hat_gt)

    @staticmethod
    def run_q_learning_experiments(env, parameters, path_output_base, path_ground_truth=None):

        varying_action = True if len(parameters["bucket_actions"]) > 1 else False
        length = len(parameters["bucket_actions"]) if varying_action else len(parameters["bucket_states"])

        for i in range(length):
            parameters_to_experiment = parameters.copy()

            if varying_action:
                parameters_to_experiment["bucket_actions"] = parameters["bucket_actions"][i]
            else:
                parameters_to_experiment["bucket_states"] = parameters["bucket_states"][i]

            wrapper = partial(Experiment.wrapper_parallel_ql_experiment,
                              env=env,
                              params=parameters_to_experiment,
                              path_output_base=path_output_base,
                              path_ground_truth=path_ground_truth,
                              index_buck=i)

            exp_indices = list(range(parameters["n_simulations"]))

            with multiprocessing.Pool(processes=2) as p:
                p.map(wrapper, exp_indices)


    @staticmethod
    def run_lr_learning_experiments(env, parameters, path_output_base, path_ground_truth=None):

        for i in range(len(parameters["k"])):
            parameters_to_experiment = parameters.copy()
            parameters_to_experiment["k"] = parameters["k"][i]

            wrapper = partial(Experiment.wrapper_parallel_lr_experiment,
                              env=env,
                              params=parameters_to_experiment,
                              path_output_base=path_output_base,
                              path_ground_truth=path_ground_truth,
                              index_buck=i)

            exp_indices = list(range(parameters["n_simulations"]))

            with multiprocessing.Pool(processes=2) as p:
                p.map(wrapper, exp_indices)


class Plotter:

    @staticmethod
    def plot_svd(paths):

        saver = Saver()

        fig, axes = plt.subplots(nrows=1, ncols=len(paths))
        for i, path in enumerate(paths):
            matrix = saver.load_from_pickle(path).Q
            _, sigma, _ = np.linalg.svd(matrix.reshape(-1, matrix.shape[-1]))

            axes[i].bar(np.arange(len(sigma)), sigma)
        plt.show()

    @staticmethod
    def plot_rewards(base_paths_arr, experiment_paths_arr):
        saver = Saver()
        fig, axes = plt.subplots(nrows=1, ncols=len(base_paths_arr))
        for index in range(len(base_paths_arr)):
            base_paths = base_paths_arr[index]
            experiment_paths = experiment_paths_arr[index]

            for i in range(len(base_paths)):
                path_base = base_paths[i]
                with open(experiment_paths[i]) as j: params = json.loads(j.read())

                n_simulations = params["n_simulations"]
                if "lr" in path_base:
                    n_experiments = len(params["k"])
                else:
                    n_experiments = max(len(params["bucket_actions"]), len(params["bucket_states"]))

                parameters = np.zeros((n_simulations, n_experiments))
                rewards = np.zeros((n_simulations, n_experiments))

                for j in range(n_experiments):
                    for k in range(n_simulations):
                        model = saver.load_from_pickle(path_base.format(j, k))
                        if "lr" in path_base:
                            parameters[k, j] = np.prod(model.L.shape) + np.prod(model.R.shape)
                        else:
                            parameters[k, j] = np.prod(model.Q.shape)
                        rewards[k, j] = np.median(model.greedy_cumulative_reward[-10:])
                axes[index].plot(np.mean(parameters, axis=0), np.median(rewards, axis=0))
        plt.show()

    @staticmethod
    def plot_convergence(base_paths_arr, experiment_paths_arr, step_diff, th):
        saver = Saver()
        fig, axes = plt.subplots(nrows=1, ncols=len(base_paths_arr))
        for index in range(len(base_paths_arr)):
            base_paths = base_paths_arr[index]
            experiment_paths = experiment_paths_arr[index]

            for i in range(len(base_paths)):
                path_base = base_paths[i]
                with open(experiment_paths[i]) as j:
                    params = json.loads(j.read())

                n_simulations = params["n_simulations"]
                if "lr" in path_base:
                    n_experiments = len(params["k"])
                else:
                    n_experiments = max(len(params["bucket_actions"]), len(params["bucket_states"]))

                parameters = np.zeros((n_simulations, n_experiments))
                convergences = np.zeros((n_simulations, n_experiments))

                for j in range(n_experiments):
                    for k in range(n_simulations):
                        model = saver.load_from_pickle(path_base.format(j, k))
                        if "lr" in path_base:
                            parameters[k, j] = np.prod(model.L.shape) + np.prod(model.R.shape)
                            for m in range(len(model.Q_hat_norms)):
                                if (m + step_diff) == len(model.Q_hat_norms):
                                    convergences[k, j] = len(model.Q_hat_norms)
                                    break

                                if (model.Q_hat_norms[m] - model.Q_hat_norms[m + step_diff]) < th:
                                    convergences[k, j] = m
                                    break
                        else:
                            parameters[k, j] = np.prod(model.Q.shape)
                            for m in range(len(model.Q_norms)):
                                if (m + step_diff) == len(model.Q_norms):
                                    convergences[k, j] = len(model.Q_norms)
                                    break

                                if (model.Q_norms[m] - model.Q_norms[m + step_diff]) < th:
                                    convergences[k, j] = m
                                    break
                axes[index].plot(np.mean(parameters, axis=0), np.mean(convergences, axis=0))
        plt.show()



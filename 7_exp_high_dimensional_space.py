from src.environments.highway import CustomHighwayEnv
from src.experiments.experiments import Experiment

env_highway = CustomHighwayEnv()

N_NODES = 20
PATH_TLR = 'highway_tlr.json'
PATH_DQN_SMALL = 'highway_dqn_batch_small.json'
PATH_DQN_LARGE = 'highway_dqn_batch_large.json'


if __name__ == "__main__":
    experiment_tlr = Experiment(PATH_TLR, env_highway, N_NODES)
    experiment_tlr.run_experiments(window=100)

    experiment_dqn_small = Experiment(PATH_DQN_SMALL, env_highway, N_NODES)
    experiment_dqn_small.run_experiments(window=100)

    experiment_dqn_large = Experiment(PATH_DQN_LARGE, env_highway, N_NODES)
    experiment_dqn_large.run_experiments(window=100)

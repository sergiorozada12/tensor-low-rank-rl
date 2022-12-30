from src.environments.highway import CustomHighwayEnv
from src.experiments.experiments import Experiment

env_highway = CustomHighwayEnv()

N_NODES = 20
PATH_TLR = 'highway_tlr.json'
PATH_DQN = 'highway_dqn.json'


if __name__ == "__main__":
    experiment_tlr = Experiment(PATH_TLR, env_highway, N_NODES)
    experiment_tlr.run_experiments(window=100)

    experiment_dqn = Experiment(PATH_DQN, env_highway, N_NODES)
    experiment_dqn.run_experiments(window=100)

from src.environments.highway import CustomHighwayEnv
from src.experiments.experiments import ExperimentHighway

env_highway = CustomHighwayEnv()

N_NODES = 20
PATH_TLR_SMALL = 'highway_tlr_small.json'
PATH_TLR_LARGE = 'highway_tlr_large.json'
PATH_DQN_LARGE = 'highway_dqn_large.json'
PATH_DQN_SMALL = 'highway_dqn_small.json'


if __name__ == "__main__":
    
    experiment_tlr = ExperimentHighway(PATH_TLR_SMALL, env_highway, N_NODES)
    experiment_tlr.run_experiments(window=100)

    experiment_tlr = ExperimentHighway(PATH_TLR_LARGE, env_highway, N_NODES)
    experiment_tlr.run_experiments(window=100)

    experiment_dqn = ExperimentHighway(PATH_DQN_SMALL, env_highway, N_NODES)
    experiment_dqn.run_experiments(window=100)

    experiment_dqn = ExperimentHighway(PATH_DQN_LARGE, env_highway, N_NODES)
    experiment_dqn.run_experiments(window=100)

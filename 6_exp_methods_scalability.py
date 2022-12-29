import os

from src.environments.pendulum import CustomPendulumEnv
from src.environments.cartpole import CustomContinuousCartPoleEnv
from src.environments.mountaincar import CustomContinuous_MountainCarEnv
from src.environments.goddard import CustomGoddardEnv

from src.experiments.experiments import ExperimentScale

env_pendulum = CustomPendulumEnv()
env_cartpole = CustomContinuousCartPoleEnv()
env_mountaincar = CustomContinuous_MountainCarEnv()
env_rocket = CustomGoddardEnv()


N_NODES = 100


if __name__ == "__main__":
    # Pendulum
    experiments = [f for f in os.listdir('parameters') if 'pendulum' in f]
    experiments_done = [f for f in os.listdir('results') if 'pendulum' in f]
    for name in experiments:
        if name in experiments_done:
            continue
        experiment = ExperimentScale(name, env_pendulum, N_NODES)
        experiment.run_experiments()

    # Cartpole
    experiments = [f for f in os.listdir('parameters') if 'cartpole' in f]
    experiments_done = [f for f in os.listdir('results') if 'cartpole' in f]
    for name in experiments:
        if name in experiments_done:
            continue
        experiment = ExperimentScale(name, env_cartpole, N_NODES)
        experiment.run_experiments()

    # Mountaincar
    experiments = [f for f in os.listdir('parameters') if 'mountaincar' in f]
    experiments_done = [f for f in os.listdir('results') if 'mountaincar' in f]
    for name in experiments:
        if name in experiments_done:
            continue
        experiment = ExperimentScale(name, env_mountaincar, N_NODES)
        experiment.run_experiments()

    # Rocket
    experiments = [f for f in os.listdir('parameters') if 'rocket' in f]
    experiments_done = [f for f in os.listdir('results') if 'rocket' in f]
    for name in experiments:
        if name in experiments_done:
            continue
        experiment = ExperimentScale(name, env_rocket, N_NODES)
        experiment.run_experiments()

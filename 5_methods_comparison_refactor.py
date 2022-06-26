import os

from src.environments.pendulum import CustomPendulumEnv
from src.environments.cartpole import CustomContinuousCartPoleEnv
from src.environments.mountaincar import CustomContinuous_MountainCarEnv
from src.environments.goddard import CustomGoddardEnv

from src.experiments.experiments import Experiment

env_pendulum = CustomPendulumEnv()
env_cartpole = CustomContinuousCartPoleEnv()
env_mountaincar = CustomContinuous_MountainCarEnv()
env_rocket = CustomGoddardEnv()


N_NODES = 1


if __name__ == "__main__":
    # Pendulum
    experiments = [f for f in os.listdir('parameters') if 'pendulum' in f]
    experiments_done = [f for f in os.listdir('results') if 'pendulum' in f]
    for name in experiments:
        if name in experiments_done:
            continue
        experiment = Experiment(name, env_pendulum, N_NODES)
        experiment.run_experiments(window=70)

    # Cartpole
    experiments = [f for f in os.listdir('parameters') if 'cartpole' in f]
    experiments_done = [f for f in os.listdir('results') if 'cartpole' in f]
    for name in experiments:
        if name in experiments_done:
            continue
        experiment = Experiment(name, env_cartpole, N_NODES)
        experiment.run_experiments(window=500)

    # Mountaincar
    experiments = [f for f in os.listdir('parameters') if 'mountaincar' in f]
    experiments_done = [f for f in os.listdir('results') if 'mountaincar' in f]
    for name in experiments:
        if name in experiments_done:
            continue
        print(name)
        experiment = Experiment(name, env_mountaincar, N_NODES)
        experiment.run_experiments(window=100)

"""
    # Rocket
    experiments = [f for f in os.listdir('parameters') if 'rocket' in f]
    experiments_done = [f for f in os.listdir('results') if 'rocket' in f]
    for name in experiments:
        if name in experiments_done:
            continue
        experiment = Experiment(name, env_rocket, N_NODES)
        experiment.run_experiments(window=50)
"""

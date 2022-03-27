import os

from src.environments.pendulum import CustomPendulumEnv
from src.environments.cartpole import CustomContinuousCartPoleEnv
from src.environments.mountaincar import CustomContinuous_MountainCarEnv
from src.environments.goddard import CustomGoddardEnv

from src.experiments.experiments import Experiment

env_pendulum = CustomPendulumEnv()
env_cartpole = CustomContinuousCartPoleEnv()
env_mountaincar = CustomContinuous_MountainCarEnv()
env_goddard = CustomGoddardEnv()

if __name__ == "__main__":
    # Pendulum
    experiments = [f for f in os.listdir('parameters') if 'pendulum' in f]
    for name in experiments:
        experiment = Experiment(name, env_pendulum, 1)
        experiment.run_experiments(window=70)

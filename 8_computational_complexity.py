import os

from src.environments.pendulum import CustomPendulumEnv
from src.environments.cartpole import CustomContinuousCartPoleEnv
from src.environments.mountaincar import CustomContinuous_MountainCarEnv
from src.environments.goddard import CustomGoddardEnv

from src.experiments.experiments import ExperimentComplexity

env_pendulum = CustomPendulumEnv()
env_cartpole = CustomContinuousCartPoleEnv()
env_mountaincar = CustomContinuous_MountainCarEnv()
env_rocket = CustomGoddardEnv()


N_NODES = 100


if __name__ == "__main__":
    # Pendulum
    experiments = [f for f in os.listdir('parameters') if 'pendulum' in f and 'scale' in f]
    for name in experiments:
        experiment = ExperimentComplexity(name, env_pendulum)
        experiment.run_experiments()

    # Cartpole
    experiments = [f for f in os.listdir('parameters') if 'cartpole' in f and 'scale' in f]
    for name in experiments:
        experiment = ExperimentComplexity(name, env_cartpole)
        experiment.run_experiments()

    # Mountaincar
    experiments = [f for f in os.listdir('parameters') if 'mountaincar' in f and 'scale' in f]
    for name in experiments:
        experiment = ExperimentComplexity(name, env_mountaincar)
        experiment.run_experiments()

    # Rocket
    experiments = [f for f in os.listdir('parameters') if 'rocket' in f and 'scale' in f]
    for name in experiments:
        experiment = ExperimentComplexity(name, env_rocket)
        experiment.run_experiments()

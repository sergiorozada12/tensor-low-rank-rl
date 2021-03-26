from utils import Saver, Plotter

saver = Saver()

Plotter.plot_svd(["models/pendulum_Q_svd.pck",
                  "models/cartpole_Q_svd.pck",
                  "models/mountaincar_Q_svd.pck"])

##############################################################################

base_paths = ["models/pendulum_ql_action_bucket_{}_exp_{}.pck",
              "models/pendulum_lr_k_bucket_{}_exp_{}.pck"]

experiment_paths = ["experiments/pendulum_low_rank_q_learner_action.json",
                    "experiments/pendulum_low_rank_lr_learner.json"]

Plotter.plot_rewards(base_paths, experiment_paths)

#############################################################################

base_paths = ["models/cartpole_ql_action_bucket_{}_exp_{}.pck",
              "models/cartpole_lr_k_bucket_{}_exp_{}.pck"]

experiment_paths = ["experiments/cartpole_low_rank_q_learner_action.json",
                    "experiments/cartpole_low_rank_lr_learner.json"]

Plotter.plot_rewards(base_paths, experiment_paths)

#############################################################################

base_paths = ["models/mountaincar_ql_action_bucket_{}_exp_{}.pck"]

experiment_paths = ["experiments/mountaincar_low_rank_q_learner_action.json"]

Plotter.plot_rewards(base_paths, experiment_paths)

from utils import Saver, Plotter

saver = Saver()

# PLOT 1
Plotter.plot_svd(["models/low_rank_reward/pendulum_ql_action_bucket_5_exp_0.pck",
                  "models/low_rank_reward/cartpole_ql_action_bucket_4_exp_0.pck",
                  "models/low_rank_reward/mountaincar_ql_action_bucket_4_exp_0.pck"])

# PLOT 2
base_paths_pend = ["models/low_rank_reward/pendulum_ql_action_bucket_{}_exp_{}.pck",
                   "models/low_rank_reward/pendulum_lr_k_bucket_{}_exp_{}.pck"]

experiment_paths_pend = ["experiments/low_rank_reward/pendulum_low_rank_q_learner_action.json",
                         "experiments/low_rank_reward/pendulum_low_rank_lr_learner.json"]

base_paths_cart = ["models/low_rank_reward/cartpole_ql_action_bucket_{}_exp_{}.pck",
                   "models/cartpole_lr_k_bucket_{}_exp_{}.pck"]

experiment_paths_cart = ["experiments/low_rank_reward/cartpole_low_rank_q_learner_action.json",
                         "experiments/cartpole_low_rank_lr_learner.json"]

base_paths_mount = ["models/low_rank_reward/mountaincar_ql_action_bucket_{}_exp_{}.pck",
                    "models/low_rank_reward/mountaincar_lr_k_bucket_{}_exp_{}.pck"]

experiment_paths_mount = ["experiments/low_rank_reward/mountaincar_low_rank_q_learner_action.json",
                          "experiments/low_rank_reward/mountaincar_low_rank_lr_learner.json"]

Plotter.plot_rewards([base_paths_pend, base_paths_cart, base_paths_mount],
                     [experiment_paths_pend, experiment_paths_cart, experiment_paths_mount])

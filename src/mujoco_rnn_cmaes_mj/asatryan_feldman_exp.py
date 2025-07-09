# %%
"""
.####.##.....##.########...#######..########..########..######.
..##..###...###.##.....##.##.....##.##.....##....##....##....##
..##..####.####.##.....##.##.....##.##.....##....##....##......
..##..##.###.##.########..##.....##.########.....##.....######.
..##..##.....##.##........##.....##.##...##......##..........##
..##..##.....##.##........##.....##.##....##.....##....##....##
.####.##.....##.##.........#######..##.....##....##.....######.
"""

import matplotlib
matplotlib.use("Agg")
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plants import SequentialReacher
from environments import SequentialReachingEnv
from networks import RNN
from utils import *

import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

# %%
"""
.##........#######.....###....########.
.##.......##.....##...##.##...##.....##
.##.......##.....##..##...##..##.....##
.##.......##.....##.##.....##.##.....##
.##.......##.....##.#########.##.....##
.##.......##.....##.##.....##.##.....##
.########..#######..##.....##.########.
"""
# Set the XML file for the SequentialReacher
# xml_file = "arm_with_pulley_flexor.xml"
# xml_file = "arm_with_pulley_extensor.xml"
xml_file = "arm_no_pulley.xml"

reacher = SequentialReacher(plant_xml_file=xml_file)
# # Ensure the XML file contains a geometry named "weight" or use an existing geometry name
# reacher = SequentialReacher(plant_xml_file="arm_just_arm.xml")  # Update XML if needed

rnn = RNN(
    input_size=3 + reacher.num_sensors,
    hidden_size=25,
    output_size=reacher.num_actuators,
    activation=tanh,
    alpha=1,  # reacher.model.opt.timestep / 10e-3,
)
target_duration = 60
env = SequentialReachingEnv(
    plant=reacher,
    target_duration={
        "mean": target_duration,
        "min": target_duration,
        "max": target_duration,
    },
    num_targets=2,
    loss_weights={
        "euclidean": 1,
        "manhattan": 0,
        "energy": 0,
        "ridge": 0.001,
        "lasso": 0,
    },
)
models_dir = "/Users/joseph/Documents/GitHub/animat/models"
# models_dir = "../../models"
gen_idx = 9000  # Specify the generation index you want to load
model_file = f"optimizer_gen_{gen_idx}_cmaesv2.pkl"

# model_file = "optimizer_gen_5000_tau10_rnn50.pkl"
# print("Current working directory:", os.getcwd())
with open(os.path.join(models_dir, model_file), "rb") as f:
    optimizer = pickle.load(f)
best_rnn = rnn.from_params(optimizer.mean)

# Swap input weights at indices 1 and 2 in best_rnn
best_rnn.W_in[:, [1, 2]] = best_rnn.W_in[:, [2, 1]]

# %%
"""
.########..########.##....##.########..########.########.
.##.....##.##.......###...##.##.....##.##.......##.....##
.##.....##.##.......####..##.##.....##.##.......##.....##
.########..######...##.##.##.##.....##.######...########.
.##...##...##.......##..####.##.....##.##.......##...##..
.##....##..##.......##...###.##.....##.##.......##....##.
.##.....##.########.##....##.########..########.##.....##
"""
# best_rnn.W_in[:,:3] = 0
# plt.figure(figsize=(10, 10))
# sns.heatmap(best_rnn.W_in, cmap="viridis", cbar=True)
# plt.title("Input Weights")
# plt.xlabel("Input Features")
# plt.ylabel("Hidden Units")

# env.feldman(
#     best_rnn,
#     weight_mod=0.5,
#     weight_density=100,
#     seed=0,
#     render=False,
#     log=True,
# )
# print("Simulation complete")
# save_path = "/Users/joseph/My Drive/Champalimaud/rotations/Joe/figures/"
# # save_path = "C:\\Users\\User\\Desktop\\tests"
# env.plot(save_path=save_path)
# print("Plotting complete")

# %%
env.passive(
    best_rnn,
    weight_mod=0.5,
    weight_density=100,
    seed=0,
    render=False,
    log=True,
)

save_path = "/Users/joseph/My Drive/Champalimaud/rotations/Joe/figures/"
# save_path = "C:\\Users\\User\\Desktop\\tests"
pickle_path = os.path.join(
    save_path, "avg_torque_vs_avg_angle_with_flexor_and_extensor.fig.pickle"
)
env.plot_passive(save_path=save_path, pickle_path=pickle_path)

# # %%

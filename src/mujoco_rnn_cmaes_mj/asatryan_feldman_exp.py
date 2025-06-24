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
reacher = SequentialReacher(plant_xml_file="arm_with_pulley.xml")
rnn = RNN(
    input_size=3 + reacher.num_sensors,
    hidden_size=25,
    output_size=reacher.num_actuators,
    activation=tanh,
    alpha=1,  # reacher.model.opt.timestep / 10e-3,
)
env = SequentialReachingEnv(
    plant=reacher,
    target_duration={"mean": 60, "min": 60, "max": 60},
    num_targets=2,
    loss_weights={
        "euclidean": 1,
        "manhattan": 0,
        "energy": 0,
        "ridge": 0.001,
        "lasso": 0,
    },
)
models_dir = "../../models"
gen_idx = 9000  # Specify the generation index you want to load
model_file = f"optimizer_gen_{gen_idx}_cmaesv2.pkl"

# model_file = "optimizer_gen_5000_tau10_rnn50.pkl"
print("Current working directory:", os.getcwd())
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
env.feldman(best_rnn, weight_mod=3, seed=0, render=True, log=True)
env.plot()


# %%

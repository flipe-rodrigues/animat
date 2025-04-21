import gymnasium as gym
import custom_envs
import torch
from train import RNNController


def test_rnn():
    env = gym.make(
        "CustomMuJoCo-v0", xml_path="path/to/your_model.xml", render_mode="human"
    )
    rnn = RNNController()
    best_params = torch.load("best_rnn_params.pth")
    rnn.set_params(best_params)

    obs, _ = env.reset()
    hidden = torch.zeros(1, 1, rnn.hidden_size)

    for _ in range(1000):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        action, hidden = rnn(obs_tensor, hidden)
        action = action.squeeze().detach().numpy()

        obs, _, done, _, _ = env.step(action)
        env.render()

        if done:
            break

    env.close()


# Run the test
test_rnn()

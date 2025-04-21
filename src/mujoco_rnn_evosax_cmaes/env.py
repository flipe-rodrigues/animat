import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

# -- Load your MJCF model --
def create_env(xml_path: str) -> mjx.Model:
    model = mjx.put_model(mujoco.MjModel.from_xml_path(xml_path))
    return model

# -- Reset the environment to default state --
def reset_env(model: mjx.Model) -> mjx.Data:
    data = mjx.data_from_model(model)
    return data

# -- Get 12-dimensional observation vector from sensors --
def get_obs(data: mjx.Data) -> jnp.ndarray:
    return data.sensordata[:12]  # Assumes 12 sensors exist in XML

# -- Apply muscle activation and step simulation forward --
def step_env(model: mjx.Model, data: mjx.Data, action: jnp.ndarray) -> mjx.Data:
    action = jnp.clip(action, 0.0, 1.0)  # Ensure within [0, 1]
    data = data.replace(ctrl=action)
    data = mjx.step(model, data)
    return data

# -- Get 3D position of the end effector --
def get_end_effector_pos(data: mjx.Data, end_effector_name="limb_tip") -> jnp.ndarray:
    # Assumes the end effector body is named "limb_tip" in the XML
    body_id = data.model.body(name=end_effector_name).id
    return data.xpos[body_id]

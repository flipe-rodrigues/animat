"""
MuJoCo Plant Interface

A wrapper around MuJoCo that provides:
- XML parsing and model introspection
- Physics simulation for muscle-driven systems
- Sensor calibration utilities

The environment uses this without needing to know MuJoCo internals.
"""

import numpy as np
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.constants import (
    DEFAULT_TIMESTEP,
    DEFAULT_CALIBRATION_EPISODES,
    DEFAULT_CALIBRATION_STEPS,
    DEFAULT_WORKSPACE_SAMPLES,
)


# ============================================================================
# Model Information Dataclasses
# ============================================================================

@dataclass
class JointInfo:
    """Information about a joint."""
    name: str
    joint_type: str = 'hinge'
    axis: Tuple[float, float, float] = (0, 0, 1)
    range: Optional[Tuple[float, float]] = None  # in degrees
    stiffness: float = 0.0
    damping: float = 0.0


@dataclass
class MuscleInfo:
    """Information about a muscle actuator."""
    name: str
    tendon_name: str = ''
    force: float = 100.0
    ctrl_range: Tuple[float, float] = (0.0, 1.0)


@dataclass
class SensorInfo:
    """Information about a sensor."""
    name: str
    sensor_type: str  # actuatorpos, actuatorvel, actuatorfrc
    target: str  # actuator name


@dataclass
class BodyInfo:
    """Information about a body."""
    name: str
    pos: Tuple[float, float, float] = (0, 0, 0)
    is_mocap: bool = False


@dataclass
class ParsedModel:
    """Complete parsed model information."""
    model_name: str
    joints: List[JointInfo] = field(default_factory=list)
    muscles: List[MuscleInfo] = field(default_factory=list)
    sensors: List[SensorInfo] = field(default_factory=list)
    bodies: List[BodyInfo] = field(default_factory=list)
    timestep: float = DEFAULT_TIMESTEP
    
    @property
    def num_joints(self) -> int:
        return len(self.joints)
    
    @property
    def num_muscles(self) -> int:
        return len(self.muscles)
    
    @property
    def num_sensors(self) -> int:
        return len(self.sensors)
    
    @property
    def muscle_names(self) -> List[str]:
        return [m.name for m in self.muscles]
    
    @property
    def joint_names(self) -> List[str]:
        return [j.name for j in self.joints]


@dataclass
class PlantState:
    """Current state of the plant."""
    muscle_lengths: np.ndarray
    muscle_velocities: np.ndarray
    muscle_forces: np.ndarray
    hand_position: np.ndarray


# ============================================================================
# XML Parsing
# ============================================================================

def _parse_tuple(s: str, n: int = 3) -> Tuple:
    """Parse space-separated string into tuple of floats."""
    values = [float(x) for x in s.split()]
    if len(values) < n:
        values.extend([0.0] * (n - len(values)))
    return tuple(values[:n])


def parse_mujoco_xml(xml_path: str) -> ParsedModel:
    """
    Parse a MuJoCo XML file and extract model information.
    
    Args:
        xml_path: Path to the MuJoCo XML file
        
    Returns:
        ParsedModel with joints, muscles, sensors, bodies
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    model_name = root.get('model', 'unnamed_model')
    
    # Timestep
    timestep = 0.01
    option = root.find('option')
    if option is not None:
        timestep = float(option.get('timestep', '0.01'))
    
    # Defaults
    default_muscle = {}
    default_joint = {}
    defaults = root.find('default')
    if defaults is not None:
        muscle_default = defaults.find('muscle')
        if muscle_default is not None and muscle_default.get('ctrlrange'):
            default_muscle['ctrl_range'] = _parse_tuple(muscle_default.get('ctrlrange'), 2)
        
        joint_default = defaults.find('joint')
        if joint_default is not None:
            default_joint['stiffness'] = float(joint_default.get('stiffness', '0'))
            default_joint['damping'] = float(joint_default.get('damping', '0'))
    
    # Parse bodies and joints recursively
    joints = []
    bodies = []
    
    def parse_body(body_elem):
        name = body_elem.get('name', 'unnamed')
        pos = _parse_tuple(body_elem.get('pos', '0 0 0'), 3)
        is_mocap = body_elem.get('mocap', 'false').lower() == 'true'
        bodies.append(BodyInfo(name=name, pos=pos, is_mocap=is_mocap))
        
        for joint_elem in body_elem.findall('joint'):
            range_str = joint_elem.get('range')
            joints.append(JointInfo(
                name=joint_elem.get('name', 'unnamed_joint'),
                joint_type=joint_elem.get('type', 'hinge'),
                axis=_parse_tuple(joint_elem.get('axis', '0 0 1'), 3),
                range=_parse_tuple(range_str, 2) if range_str else None,
                stiffness=float(joint_elem.get('stiffness', default_joint.get('stiffness', 0))),
                damping=float(joint_elem.get('damping', default_joint.get('damping', 0))),
            ))
        
        for child in body_elem.findall('body'):
            parse_body(child)
    
    worldbody = root.find('worldbody')
    if worldbody is not None:
        for body_elem in worldbody.findall('body'):
            parse_body(body_elem)
    
    # Parse muscles
    muscles = []
    actuator_section = root.find('actuator')
    if actuator_section is not None:
        for muscle_elem in actuator_section.findall('muscle'):
            ctrl_range = default_muscle.get('ctrl_range', (0.0, 1.0))
            if muscle_elem.get('ctrlrange'):
                ctrl_range = _parse_tuple(muscle_elem.get('ctrlrange'), 2)
            
            muscles.append(MuscleInfo(
                name=muscle_elem.get('name', 'unnamed_muscle'),
                tendon_name=muscle_elem.get('tendon', ''),
                force=float(muscle_elem.get('force', '100')),
                ctrl_range=ctrl_range,
            ))
    
    # Parse sensors
    sensors = []
    sensor_section = root.find('sensor')
    if sensor_section is not None:
        for sensor_elem in sensor_section:
            if sensor_elem.tag is ET.Comment:
                continue
            
            sensor_type = sensor_elem.tag
            target = ''
            if sensor_type in ['actuatorpos', 'actuatorvel', 'actuatorfrc']:
                target = sensor_elem.get('actuator', '')
            elif sensor_type in ['jointpos', 'jointvel']:
                target = sensor_elem.get('joint', '')
            
            sensors.append(SensorInfo(
                name=sensor_elem.get('name', 'unnamed_sensor'),
                sensor_type=sensor_type,
                target=target,
            ))
    
    return ParsedModel(
        model_name=model_name,
        joints=joints,
        muscles=muscles,
        sensors=sensors,
        bodies=bodies,
        timestep=timestep,
    )


def get_model_dimensions(parsed_model: ParsedModel) -> Dict[str, int]:
    """Calculate dimensions needed for neural network architecture."""
    return {
        'num_joints': parsed_model.num_joints,
        'num_muscles': parsed_model.num_muscles,
        'num_total_sensors': parsed_model.num_sensors,
        'num_proprioceptive_inputs': parsed_model.num_muscles * 3,
        'num_outputs': parsed_model.num_muscles * 3,
    }


# ============================================================================
# MuJoCo Plant Wrapper
# ============================================================================

class MuJoCoPlant:
    """
    MuJoCo physics wrapper for muscle-driven arm.
    
    Provides a clean interface to the physics simulation,
    hiding MuJoCo implementation details.
    """
    
    def __init__(self, xml_path: str, render_mode: Optional[str] = None):
        import mujoco
        
        self.xml_path = xml_path
        self.render_mode = render_mode
        
        # Parse model info
        self.parsed_model = parse_mujoco_xml(xml_path)
        
        # Load MuJoCo
        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self._mj_data = mujoco.MjData(self._mj_model)
        
        # Dimensions
        self.num_muscles = self.parsed_model.num_muscles
        self.num_joints = self.parsed_model.num_joints
        self.num_sensors = self.parsed_model.num_sensors
        self.dt = self._mj_model.opt.timestep
        
        # Body IDs
        self._hand_body_id = self._find_body_id('hand')
        self._target_body_id = self._find_body_id('target')
        self._target_mocap_id = (
            self._mj_model.body_mocapid[self._target_body_id]
            if self._target_body_id >= 0 else -1
        )
        
        self._renderer = None
    
    @property
    def joints(self) -> List[JointInfo]:
        """Joint information for sampling configurations."""
        return self.parsed_model.joints
    
    def _find_body_id(self, name: str) -> int:
        import mujoco
        try:
            return mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        except:
            return -1
    
    def reset(self, joint_angles: Optional[np.ndarray] = None) -> PlantState:
        """Reset simulation to initial state."""
        import mujoco
        
        mujoco.mj_resetData(self._mj_model, self._mj_data)
        
        if joint_angles is not None:
            self._mj_data.qpos[:self.num_joints] = joint_angles
        
        self._mj_data.qvel[:] = 0
        mujoco.mj_forward(self._mj_model, self._mj_data)
        
        return self.get_state()
    
    def step(self, muscle_activations: np.ndarray) -> PlantState:
        """Step simulation with muscle activations [0, 1]."""
        import mujoco
        
        self._mj_data.ctrl[:self.num_muscles] = np.clip(muscle_activations, 0.0, 1.0)
        mujoco.mj_step(self._mj_model, self._mj_data)
        
        return self.get_state()
    
    def get_state(self) -> PlantState:
        """Get current plant state."""
        return PlantState(
            muscle_lengths=self._mj_data.actuator_length[:self.num_muscles].copy(),
            muscle_velocities=self._mj_data.actuator_velocity[:self.num_muscles].copy(),
            muscle_forces=self._mj_data.actuator_force[:self.num_muscles].copy(),
            hand_position=self.get_hand_position(),
        )
    
    def get_hand_position(self) -> np.ndarray:
        """Get end effector position."""
        if self._hand_body_id >= 0:
            return self._mj_data.xpos[self._hand_body_id].copy()
        return np.zeros(3)
    
    def get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities."""
        return self._mj_data.qvel[:self.num_joints].copy()
    
    def set_target_position(self, position: np.ndarray):
        """Set mocap target position for visualization."""
        if self._target_mocap_id >= 0:
            self._mj_data.mocap_pos[self._target_mocap_id] = position
    
    def estimate_workspace(
        self, 
        num_samples: int = DEFAULT_WORKSPACE_SAMPLES,
    ) -> Dict[str, any]:
        """
        Estimate reachable workspace by sampling joint configurations.
        
        Returns both bounding box (for backward compatibility) and the actual
        sampled positions for more accurate target sampling.
        
        Args:
            num_samples: Number of joint configurations to sample
            
        Returns:
            Dictionary with:
            - 'x', 'y', 'z': Bounding box as (min, max) tuples
            - 'positions': Array of sampled reachable positions (N x 3)
        """
        import mujoco
        
        # Get joint limits
        joint_ranges = []
        for joint in self.parsed_model.joints:
            if joint.range is not None:
                joint_ranges.append((np.deg2rad(joint.range[0]), np.deg2rad(joint.range[1])))
            else:
                joint_ranges.append((-np.pi, np.pi))
        
        positions = []
        original_qpos = self._mj_data.qpos.copy()
        
        for _ in range(num_samples):
            # Sample random joint configuration within limits
            qpos = np.array([np.random.uniform(lo, hi) for lo, hi in joint_ranges])
            self._mj_data.qpos[:self.num_joints] = qpos
            mujoco.mj_forward(self._mj_model, self._mj_data)
            positions.append(self.get_hand_position())
        
        # Restore original state
        self._mj_data.qpos[:] = original_qpos
        mujoco.mj_forward(self._mj_model, self._mj_data)
        
        positions = np.array(positions)
        
        # Return both bounds (for compatibility) and actual positions
        return {
            'x': (float(positions[:, 0].min()), float(positions[:, 0].max())),
            'y': (float(positions[:, 1].min()), float(positions[:, 1].max())),
            'z': (float(positions[:, 2].min()), float(positions[:, 2].max())),
            'positions': positions,
        }
    
    def render(self) -> Optional[np.ndarray]:
        """Render current state to RGB array."""
        if self.render_mode is None:
            return None
        
        import mujoco
        
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._mj_model, 480, 640)
            # Set up camera for top-down view, zoomed out
            self._camera = mujoco.MjvCamera()
            self._camera.azimuth = 90  # Rotation around vertical
            self._camera.elevation = -90  # Top-down view (looking straight down)
            self._camera.distance = 1.5  # Zoomed out (larger = further)
            self._camera.lookat[:] = [0.0, -0.33, 0.0]  # Center on arm workspace
        
        self._renderer.update_scene(self._mj_data, self._camera)
        frame = self._renderer.render()
        
        if self.render_mode == 'human':
            import cv2
            cv2.imshow('Muscle Arm', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        
        return frame
    
    def close(self):
        """Clean up resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# ============================================================================
# Calibration
# ============================================================================

def calibrate_sensors(
    xml_path: str,
    num_episodes: int = DEFAULT_CALIBRATION_EPISODES,
    max_steps: int = DEFAULT_CALIBRATION_STEPS,
) -> Dict[str, np.ndarray]:
    """
    Calibrate sensor statistics by running random episodes.
    
    Args:
        xml_path: Path to MuJoCo XML
        num_episodes: Number of calibration episodes
        max_steps: Steps per episode
        
    Returns:
        Dictionary with mean and std for each sensor type
    """
    plant = MuJoCoPlant(xml_path)
    
    all_lengths = []
    all_velocities = []
    all_forces = []
    
    for _ in range(num_episodes):
        joint_ranges = []
        for joint in plant.parsed_model.joints:
            if joint.range is not None:
                joint_ranges.append((np.deg2rad(joint.range[0]), np.deg2rad(joint.range[1])))
            else:
                joint_ranges.append((-np.pi, np.pi))
        
        initial_angles = np.array([np.random.uniform(lo, hi) for lo, hi in joint_ranges])
        plant.reset(initial_angles)
        
        for _ in range(max_steps):
            action = np.random.uniform(0, 1, plant.num_muscles)
            state = plant.step(action)
            
            all_lengths.append(state.muscle_lengths)
            all_velocities.append(state.muscle_velocities)
            all_forces.append(state.muscle_forces)
    
    plant.close()
    
    all_lengths = np.array(all_lengths)
    all_velocities = np.array(all_velocities)
    all_forces = np.array(all_forces)
    
    return {
        'length_mean': all_lengths.mean(axis=0),
        'length_std': all_lengths.std(axis=0) + 1e-8,
        'velocity_mean': all_velocities.mean(axis=0),
        'velocity_std': all_velocities.std(axis=0) + 1e-8,
        'force_mean': all_forces.mean(axis=0),
        'force_std': all_forces.std(axis=0) + 1e-8,
    }


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        model = parse_mujoco_xml(sys.argv[1])
        
        print(f"\n{'='*60}")
        print(f"Model: {model.model_name}")
        print(f"{'='*60}")
        print(f"Timestep: {model.timestep}s")
        
        print(f"\nJoints ({model.num_joints}):")
        for j in model.joints:
            print(f"  - {j.name}: {j.joint_type}, range={j.range}")
        
        print(f"\nMuscles ({model.num_muscles}):")
        for m in model.muscles:
            print(f"  - {m.name}: force={m.force}N")
        
        print(f"\nSensors ({model.num_sensors}):")
        for s in model.sensors:
            print(f"  - {s.name}: {s.sensor_type} -> {s.target}")
        
        print(f"\nBodies ({len(model.bodies)}):")
        for b in model.bodies:
            mocap = " (mocap)" if b.is_mocap else ""
            print(f"  - {b.name}{mocap}")
        
        print(f"\nDimensions:")
        for k, v in get_model_dimensions(model).items():
            print(f"  {k}: {v}")
    else:
        print("Usage: python plant.py <path_to_mujoco_xml>")
    def close(self):
        """Clean up resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

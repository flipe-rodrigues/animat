"""
MuJoCo Model Parser
Dynamically parses MuJoCo XML to identify muscles, sensors, joints, and bodies.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class JointInfo:
    """Information about a joint in the model."""
    name: str
    joint_type: str
    axis: Tuple[float, float, float]
    range: Optional[Tuple[float, float]] = None
    stiffness: float = 0.0
    damping: float = 0.0


@dataclass
class MuscleInfo:
    """Information about a muscle actuator."""
    name: str
    tendon_name: str
    force: float
    ctrl_range: Tuple[float, float] = (0.0, 1.0)


@dataclass
class SensorInfo:
    """Information about a sensor."""
    name: str
    sensor_type: str  # actuatorpos, actuatorvel, actuatorfrc, etc.
    target: str  # actuator/joint/site name


@dataclass
class BodyInfo:
    """Information about a body."""
    name: str
    pos: Tuple[float, float, float]
    is_mocap: bool = False


@dataclass
class ParsedModel:
    """Complete parsed model information."""
    model_name: str
    joints: List[JointInfo] = field(default_factory=list)
    muscles: List[MuscleInfo] = field(default_factory=list)
    sensors: List[SensorInfo] = field(default_factory=list)
    bodies: List[BodyInfo] = field(default_factory=list)
    timestep: float = 0.01
    
    # Derived properties
    @property
    def n_joints(self) -> int:
        return len(self.joints)
    
    @property
    def n_muscles(self) -> int:
        return len(self.muscles)
    
    @property
    def n_sensors(self) -> int:
        return len(self.sensors)
    
    @property
    def muscle_names(self) -> List[str]:
        return [m.name for m in self.muscles]
    
    @property
    def joint_names(self) -> List[str]:
        return [j.name for j in self.joints]
    
    def get_sensors_by_type(self, sensor_type: str) -> List[SensorInfo]:
        """Get all sensors of a given type."""
        return [s for s in self.sensors if s.sensor_type == sensor_type]
    
    def get_length_sensors(self) -> List[SensorInfo]:
        """Get muscle length sensors (actuatorpos)."""
        return self.get_sensors_by_type('actuatorpos')
    
    def get_velocity_sensors(self) -> List[SensorInfo]:
        """Get muscle velocity sensors (actuatorvel)."""
        return self.get_sensors_by_type('actuatorvel')
    
    def get_force_sensors(self) -> List[SensorInfo]:
        """Get muscle force sensors (actuatorfrc)."""
        return self.get_sensors_by_type('actuatorfrc')
    
    def get_hand_body(self) -> Optional[BodyInfo]:
        """Find the end effector (hand) body."""
        for body in self.bodies:
            if 'hand' in body.name.lower():
                return body
        # Return last non-mocap body if no explicit hand
        non_mocap = [b for b in self.bodies if not b.is_mocap]
        return non_mocap[-1] if non_mocap else None
    
    def get_target_body(self) -> Optional[BodyInfo]:
        """Find the target (mocap) body."""
        for body in self.bodies:
            if body.is_mocap or 'target' in body.name.lower():
                return body
        return None


def parse_tuple(s: str, n: int = 3) -> Tuple:
    """Parse a space-separated string into a tuple of floats."""
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
        ParsedModel containing all extracted information
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get model name
    model_name = root.get('model', 'unnamed_model')
    
    # Get timestep from option
    timestep = 0.01
    option = root.find('option')
    if option is not None:
        timestep = float(option.get('timestep', '0.01'))
    
    # Parse defaults for muscles
    default_muscle = {}
    default_joint = {}
    defaults = root.find('default')
    if defaults is not None:
        muscle_default = defaults.find('muscle')
        if muscle_default is not None:
            if muscle_default.get('ctrlrange'):
                default_muscle['ctrl_range'] = parse_tuple(muscle_default.get('ctrlrange'), 2)
        
        joint_default = defaults.find('joint')
        if joint_default is not None:
            default_joint['stiffness'] = float(joint_default.get('stiffness', '0'))
            default_joint['damping'] = float(joint_default.get('damping', '0'))
    
    # Parse joints and bodies recursively
    joints = []
    bodies = []
    
    def parse_body(body_elem, parent_pos=(0, 0, 0)):
        """Recursively parse bodies and their joints."""
        name = body_elem.get('name', 'unnamed')
        pos_str = body_elem.get('pos', '0 0 0')
        pos = parse_tuple(pos_str, 3)
        is_mocap = body_elem.get('mocap', 'false').lower() == 'true'
        
        bodies.append(BodyInfo(name=name, pos=pos, is_mocap=is_mocap))
        
        # Parse joints in this body
        for joint_elem in body_elem.findall('joint'):
            joint_name = joint_elem.get('name', 'unnamed_joint')
            joint_type = joint_elem.get('type', 'hinge')
            axis = parse_tuple(joint_elem.get('axis', '0 0 1'), 3)
            
            range_str = joint_elem.get('range')
            joint_range = parse_tuple(range_str, 2) if range_str else None
            
            stiffness = float(joint_elem.get('stiffness', default_joint.get('stiffness', 0)))
            damping = float(joint_elem.get('damping', default_joint.get('damping', 0)))
            
            joints.append(JointInfo(
                name=joint_name,
                joint_type=joint_type,
                axis=axis,
                range=joint_range,
                stiffness=stiffness,
                damping=damping
            ))
        
        # Recurse into child bodies
        for child_body in body_elem.findall('body'):
            parse_body(child_body, pos)
    
    worldbody = root.find('worldbody')
    if worldbody is not None:
        for body_elem in worldbody.findall('body'):
            parse_body(body_elem)
    
    # Parse tendons (for reference)
    tendons = {}
    tendon_section = root.find('tendon')
    if tendon_section is not None:
        for tendon_elem in tendon_section.findall('spatial'):
            tendon_name = tendon_elem.get('name', 'unnamed_tendon')
            tendons[tendon_name] = tendon_elem
    
    # Parse muscles (actuators)
    muscles = []
    actuator_section = root.find('actuator')
    if actuator_section is not None:
        for muscle_elem in actuator_section.findall('muscle'):
            name = muscle_elem.get('name', 'unnamed_muscle')
            tendon_name = muscle_elem.get('tendon', '')
            force = float(muscle_elem.get('force', '100'))
            
            ctrl_range = default_muscle.get('ctrl_range', (0.0, 1.0))
            if muscle_elem.get('ctrlrange'):
                ctrl_range = parse_tuple(muscle_elem.get('ctrlrange'), 2)
            
            muscles.append(MuscleInfo(
                name=name,
                tendon_name=tendon_name,
                force=force,
                ctrl_range=ctrl_range
            ))
    
    # Parse sensors
    sensors = []
    sensor_section = root.find('sensor')
    if sensor_section is not None:
        for sensor_elem in sensor_section:
            # Skip comments
            if sensor_elem.tag is ET.Comment:
                continue
            
            sensor_type = sensor_elem.tag
            name = sensor_elem.get('name', 'unnamed_sensor')
            
            # Determine target based on sensor type
            target = ''
            if sensor_type in ['actuatorpos', 'actuatorvel', 'actuatorfrc']:
                target = sensor_elem.get('actuator', '')
            elif sensor_type in ['jointpos', 'jointvel']:
                target = sensor_elem.get('joint', '')
            elif sensor_type in ['framepos', 'framequat', 'framelinvel']:
                target = sensor_elem.get('objname', '')
            
            sensors.append(SensorInfo(
                name=name,
                sensor_type=sensor_type,
                target=target
            ))
    
    return ParsedModel(
        model_name=model_name,
        joints=joints,
        muscles=muscles,
        sensors=sensors,
        bodies=bodies,
        timestep=timestep
    )


def infer_muscle_sensor_mapping(parsed_model: ParsedModel) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Infer which sensors correspond to which muscles.
    
    Returns:
        Dict mapping muscle name to dict of sensor types ('length', 'velocity', 'force')
    """
    mapping = {}
    
    for muscle in parsed_model.muscles:
        mapping[muscle.name] = {
            'length': None,
            'velocity': None,
            'force': None
        }
        
        # Find sensors that target this muscle
        for sensor in parsed_model.sensors:
            if sensor.target == muscle.name:
                if sensor.sensor_type == 'actuatorpos':
                    mapping[muscle.name]['length'] = sensor.name
                elif sensor.sensor_type == 'actuatorvel':
                    mapping[muscle.name]['velocity'] = sensor.name
                elif sensor.sensor_type == 'actuatorfrc':
                    mapping[muscle.name]['force'] = sensor.name
    
    return mapping


def get_model_dimensions(parsed_model: ParsedModel) -> Dict[str, int]:
    """
    Calculate the dimensions needed for neural network architecture.
    
    Returns:
        Dict with dimension counts
    """
    sensor_mapping = infer_muscle_sensor_mapping(parsed_model)
    
    # Count sensors per type
    n_length_sensors = sum(1 for m in sensor_mapping.values() if m['length'] is not None)
    n_velocity_sensors = sum(1 for m in sensor_mapping.values() if m['velocity'] is not None)
    n_force_sensors = sum(1 for m in sensor_mapping.values() if m['force'] is not None)
    
    return {
        'n_joints': parsed_model.n_joints,
        'n_muscles': parsed_model.n_muscles,
        'n_length_sensors': n_length_sensors,
        'n_velocity_sensors': n_velocity_sensors,
        'n_force_sensors': n_force_sensors,
        'n_total_sensors': parsed_model.n_sensors,
        # Proprioceptive input dim: 3 types of sensors per muscle (Ia, II, Ib)
        # Even if some sensors are missing in XML, we'll use the muscle count
        'n_proprioceptive_inputs': parsed_model.n_muscles * 3,
        # Output dim: alpha MNs + gamma static + gamma dynamic
        'n_outputs': parsed_model.n_muscles * 3,
    }


if __name__ == '__main__':
    # Test with sample XML
    import sys
    if len(sys.argv) > 1:
        model = parse_mujoco_xml(sys.argv[1])
        print(f"Model: {model.model_name}")
        print(f"Timestep: {model.timestep}")
        print(f"\nJoints ({model.n_joints}):")
        for j in model.joints:
            print(f"  - {j.name}: {j.joint_type}, range={j.range}")
        print(f"\nMuscles ({model.n_muscles}):")
        for m in model.muscles:
            print(f"  - {m.name}: tendon={m.tendon_name}, force={m.force}")
        print(f"\nSensors ({model.n_sensors}):")
        for s in model.sensors:
            print(f"  - {s.name}: type={s.sensor_type}, target={s.target}")
        print(f"\nBodies ({len(model.bodies)}):")
        for b in model.bodies:
            print(f"  - {b.name}: pos={b.pos}, mocap={b.is_mocap}")
        
        print(f"\nSensor-Muscle Mapping:")
        mapping = infer_muscle_sensor_mapping(model)
        for muscle, sensors in mapping.items():
            print(f"  {muscle}: {sensors}")
        
        print(f"\nModel Dimensions:")
        dims = get_model_dimensions(model)
        for k, v in dims.items():
            print(f"  {k}: {v}")

import os

from dm_control import composer
from dm_control.composer.observation import observable
from dm_control import mjcf

from utils import get_root_path

class ArmEntity(composer.Entity):
    """A robotic arm that can reach targets."""
    
    def _build(self):
        # Create the MJCF model
        model_path = os.path.join(get_root_path(), "mujoco", "arm.xml")
        self._mjcf_root = mjcf.from_path(model_path)
        
        # Store references to key components
        self._target = self._mjcf_root.find('body', 'target')
        self._hand = self._mjcf_root.find('body', 'hand')
        
    def _build_observables(self):
        return ArmObservables(self)
    
    @property
    def mjcf_model(self):
        return self._mjcf_root
    
    @property
    def actuators(self):
        return tuple(self._mjcf_root.find_all('actuator'))
    
    @property
    def target(self):
        return self._target
    
    @property
    def hand(self):
        return self._hand

class ArmObservables(composer.Observables):
    """Observables for the arm entity."""
    
    @composer.observable
    def muscle_sensors(self):
        # Collect all sensors on the MJCF model
        sensors = self._entity._mjcf_root.find_all('sensor')
        # 'sensordata' is the MuJoCo name for physics.data.sensordata
        return observable.MJCFFeature('sensordata', sensors)

    @composer.observable
    def target_position(self):
        # Returns the mocap_pos of the target body
        return observable.MJCFFeature('mocap_pos', [self._entity._target])
    

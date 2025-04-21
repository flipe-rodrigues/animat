"""
Generate sequences of targets for advanced experiments.
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Callable

class TargetGenerator:
    def __init__(self, seed: int = 42):
        """
        Initialize the target generator.
        
        Args:
            seed: Random seed
        """
        self.key = jax.random.PRNGKey(seed)
    
    def generate_random_targets(self, n_targets: int, min_radius: float = 0.3, max_radius: float = 0.7) -> jnp.ndarray:
        """
        Generate random target positions within a spherical space.
        
        Args:
            n_targets: Number of targets to generate
            min_radius: Minimum radius from origin
            max_radius: Maximum radius from origin
            
        Returns:
            Array of target positions [n_targets, 3]
        """
        targets = []
        
        for _ in range(n_targets):
            # Generate new random key
            self.key, subkey = jax.random.split(self.key)
            k1, k2 = jax.random.split(subkey)
            
            # Random radius between min_radius and max_radius
            radius = min_radius + (max_radius - min_radius) * jax.random.uniform(k1)
            
            # Random spherical coordinates
            theta = jax.random.uniform(k2, minval=0, maxval=2*jnp.pi)  # Azimuthal angle
            phi = jax.random.uniform(k2, minval=0, maxval=jnp.pi)      # Polar angle
            
            # Convert to Cartesian coordinates
            x = radius * jnp.sin(phi) * jnp.cos(theta)
            y = radius * jnp.sin(phi) * jnp.sin(theta)
            z = radius * jnp.cos(phi)
            
            targets.append(jnp.array([x, y, z]))
        
        return jnp.stack(targets)
    
    def generate_grid_targets(self, nx: int = 3, ny: int = 3, nz: int = 3, 
                              x_range: Tuple[float, float] = (-0.5, 0.5),
                              y_range: Tuple[float, float] = (-0.5, 0.5),
                              z_range: Tuple[float, float] = (0.0, 0.7)) -> jnp.ndarray:
        """
        Generate a grid of target positions.
        
        Args:
            nx: Number of points in x dimension
            ny: Number of points in y dimension
            nz: Number of points in z dimension
            x_range: Range of x values (min, max)
            y_range: Range of y values (min, max)
            z_range: Range of z values (min, max)
            
        Returns:
            Array of target positions [nx*ny*nz, 3]
        """
        # Generate evenly spaced grid points
        x = jnp.linspace(x_range[0], x_range[1], nx)
        y = jnp.linspace(y_range[0], y_range[1], ny)
        z = jnp.linspace(z_range[0], z_range[1], nz)
        
        # Create meshgrid
        X, Y, Z = jnp.meshgrid(x, y, z)
        
        # Stack into target positions
        targets = jnp.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        return targets
    
    def generate_circular_targets(self, n_targets: int, radius: float = 0.5, 
                                  center: Tuple[float, float, float] = (0.0, 0.0, 0.4),
                                  plane: str = 'xz') -> jnp.ndarray:
        """
        Generate targets arranged in a circle.
        
        Args:
            n_targets: Number of targets
            radius: Radius of the circle
            center: Center point of the circle (x, y, z)
            plane: Plane in which to generate the circle ('xy', 'xz', or 'yz')
            
        Returns:
            Array of target positions [n_targets, 3]
        """
        # Generate evenly spaced angles
        angles = jnp.linspace(0, 2*jnp.pi, n_targets, endpoint=False)
        
        # Initialize targets
        targets = jnp.zeros((n_targets, 3))
        
        # Set coordinates based on the plane
        if plane == 'xy':
            targets = targets.at[:, 0].set(center[0] + radius * jnp.cos(angles))
            targets = targets.at[:, 1].set(center[1] + radius * jnp.sin(angles))
            targets = targets.at[:, 2].set(center[2])
        elif plane == 'xz':
            targets = targets.at[:, 0].set(center[0] + radius * jnp.cos(angles))
            targets = targets.at[:, 1].set(center[1])
            targets = targets.at[:, 2].set(center[2] + radius * jnp.sin(angles))
        elif plane == 'yz':
            targets = targets.at[:, 0].set(center[0])
            targets = targets.at[:, 1].set(center[1] + radius * jnp.cos(angles))
            targets = targets.at[:, 2].set(center[2] + radius * jnp.sin(angles))
        else:
            raise ValueError(f"Invalid plane: {plane}. Must be 'xy', 'xz', or 'yz'.")
        
        return targets
    
    def generate_sequential_pattern(self, pattern_fn: Callable, n_cycles: int, 
                                    points_per_cycle: int) -> jnp.ndarray:
        """
        Generate a sequence of targets following a pattern function over time.
        
        Args:
            pattern_fn: Function that maps a time value (0 to 1) to a 3D position
            n_cycles: Number of cycles to generate
            points_per_cycle: Number of points per cycle
            
        Returns:
            Array of target positions [n_cycles * points_per_cycle, 3]
        """
        # Generate time values
        times = jnp.linspace(0, n_cycles, n_cycles * points_per_cycle, endpoint=False)
        
        # Apply pattern function to get positions
        # The pattern function should handle the cyclic nature internally
        targets = jax.vmap(pattern_fn)(times)
        
        return targets
    
    def figure_eight(self, t: float, scale: float = 0.5, center: Tuple[float, float, float] = (0.0, 0.0, 0.4),
                     plane: str = 'xz') -> jnp.ndarray:
        """
        Generate a point on a figure-eight pattern at time t.
        
        Args:
            t: Time parameter (0 to 1 for one complete cycle)
            scale: Scale of the figure eight
            center: Center point of the figure eight
            plane: Plane in which to generate the figure eight
            
        Returns:
            3D position on the figure eight
        """
        # Parametric equation of a figure eight
        t_mod = t % 1.0  # Ensure t is in [0, 1)
        angle = 2 * jnp.pi * t_mod
        
        # Calculate position
        pos = jnp.zeros(3)
        
        if plane == 'xy':
            pos = pos.at[0].set(center[0] + scale * jnp.sin(angle))
            pos = pos.at[1].set(center[1] + scale * jnp.sin(angle) * jnp.cos(angle))
            pos = pos.at[2].set(center[2])
        elif plane == 'xz':
            pos = pos.at[0].set(center[0] + scale * jnp.sin(angle))
            pos = pos.at[1].set(center[1])
            pos = pos.at[2].set(center[2] + scale * jnp.sin(angle) * jnp.cos(angle))
        elif plane == 'yz':
            pos = pos.at[0].set(center[0])
            pos = pos.at[1].set(center[1] + scale * jnp.sin(angle))
            pos = pos.at[2].set(center[2] + scale * jnp.sin(angle) * jnp.cos(angle))
        else:
            raise ValueError(f"Invalid plane: {plane}. Must be 'xy', 'xz', or 'yz'.")
        
        return pos
    
    def spiral(self, t: float, scale: float = 0.5, center: Tuple[float, float, float] = (0.0, 0.0, 0.4),
               plane: str = 'xz') -> jnp.ndarray:
        """
        Generate a point on a spiral pattern at time t.
        
        Args:
            t: Time parameter (0 to 1 for one complete cycle)
            scale: Scale of the spiral
            center: Center point of the spiral
            plane: Plane in which to generate the spiral
            
        Returns:
            3D position on the spiral
        """
        # Parametric equation of a spiral
        t_mod = t % 1.0  # Ensure t is in [0, 1)
        angle = 2 * jnp.pi * t_mod
        radius = scale * t_mod
        
        # Calculate position
        pos = jnp.zeros(3)
        
        if plane == 'xy':
            pos = pos.at[0].set(center[0] + radius * jnp.cos(angle))
            pos = pos.at[1].set(center[1] + radius * jnp.sin(angle))
            pos = pos.at[2].set(center[2])
        elif plane == 'xz':
            pos = pos.at[0].set(center[0] + radius * jnp.cos(angle))
            pos = pos.at[1].set(center[1])
            pos = pos.at[2].set(center[2] + radius * jnp.sin(angle))
        elif plane == 'yz':
            pos = pos.at[0].set(center[0])
            pos = pos.at[1].set(center[1] + radius * jnp.cos(angle))
            pos = pos.at[2].set(center[2] + radius * jnp.sin(angle))
        else:
            raise ValueError(f"Invalid plane: {plane}. Must be 'xy', 'xz', or 'yz'.")
        
        return pos
    
    def save_targets(self, targets: jnp.ndarray, output_path: str) -> None:
        """
        Save generated targets to a file.
        
        Args:
            targets: Array of target positions
            output_path: Path to save the targets
        """
        # Convert to numpy for saving
        targets_np = np.array(targets)
        
        # Save using numpy
        np.save(output_path, targets_np)
        print(f"Targets saved to {output_path}")
    
    def load_targets(self, input_path: str) -> jnp.ndarray:
        """
        Load targets from a file.
        
        Args:
            input_path: Path to the targets file
            
        Returns:
            Array of target positions
        """
        # Load numpy array
        targets_np = np.load(input_path)
        
        # Convert to JAX array
        targets = jnp.array(targets_np)
        
        print(f"Loaded {len(targets)} targets from {input_path}")
        
        return targets
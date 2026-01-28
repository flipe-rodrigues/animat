import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import truncated_exponential

MUSCLES = ["deltoid", "latissimus", "biceps", "triceps"]
SENSORS = ["len", "vel", "frc"]


"""
.########.....###....########...######...########.########
....##.......##.##...##.....##.##....##..##..........##...
....##......##...##..##.....##.##........##..........##...
....##.....##.....##.########..##...####.######......##...
....##.....#########.##...##...##....##..##..........##...
....##.....##.....##.##....##..##....##..##..........##...
....##.....##.....##.##.....##..######...########....##...
"""


class TargetSchedule:
    """Encapsulates all target timing information"""

    __slots__ = (
        "positions",
        "onset_times",
        "offset_times",
        "durations",
        "itis",
        "trial_duration",
        "num_targets",
    )

    def __init__(self, positions, durations, itis):
        self.positions = positions
        self.durations = durations
        self.itis = itis
        self.onset_times = np.concatenate([[0], (durations[:-1] + itis[:-1]).cumsum()])
        self.offset_times = self.onset_times + durations
        self.trial_duration = durations.sum() + itis.sum()
        self.num_targets = len(positions)

    def should_disable_target(self, time, target_idx):
        return time >= self.offset_times[target_idx]

    def should_enable_target(self, time, target_idx):
        return target_idx < self.num_targets and time >= self.onset_times[target_idx]


class RewardCalculator:
    """Lightweight reward computation with pre-negated weights"""

    __slots__ = ("distance_weight", "energy_weight")

    def __init__(self, loss_weights):
        self.distance_weight = -loss_weights["distance"]
        self.energy_weight = -loss_weights["energy"]

    def compute(self, distance, energy):
        """Compute reward using pre-negated weights"""
        return distance * self.distance_weight + energy * self.energy_weight


"""
.########.##....##.##.....##
.##.......###...##.##.....##
.##.......####..##.##.....##
.######...##.##.##.##.....##
.##.......##..####..##...##.
.##.......##...###...##.##..
.########.##....##....###...
"""


class SequentialReachingEnv:
    """
    Lightweight environment for sequential reaching tasks.

    This class stores only configuration and is designed to be created once
    per worker. The plant and encoder are passed as arguments to evaluate()
    to avoid redundant object creation.
    """

    def __init__(
        self,
        target_duration_distro,
        iti_distro,
        num_targets,
        randomize_gravity=False,
        loss_weights=None,
        profile=False,
    ):
        """
        Initialize environment configuration.

        Args:
            target_duration_distro: Dict with 'mean', 'min', 'max' for target durations
            iti_distro: Dict with 'mean', 'min', 'max' for inter-trial intervals
            num_targets: Number of targets per evaluation
            randomize_gravity: Whether to randomize gravity direction
            loss_weights: Dict with 'distance' and 'energy' weights
            profile: Whether to collect performance metrics
        """
        # Set defaults
        if loss_weights is None:
            loss_weights = {"distance": 1.0, "energy": 0.1}

        # Validate configuration
        self._validate_config(
            target_duration_distro, iti_distro, num_targets, loss_weights
        )

        # Store configuration
        self.target_duration_distro = target_duration_distro
        self.iti_distro = iti_distro
        self.num_targets = num_targets
        self.randomize_gravity = randomize_gravity
        self.loss_weights = loss_weights
        self.profile = profile

        # State
        self.logger = None
        self._log_capacity = 0
        self._profile_stats = {} if profile else None

    def _validate_config(
        self, target_duration_distro, iti_distro, num_targets, loss_weights
    ):
        """Validate configuration at initialization"""
        required_keys = ["mean", "min", "max"]

        for name, distro in [
            ("target_duration", target_duration_distro),
            ("iti", iti_distro),
        ]:
            if not all(k in distro for k in required_keys):
                raise ValueError(f"{name}_distro must contain {required_keys}")
            if distro["min"] > distro["max"]:
                raise ValueError(f"{name}_distro: min > max")

        if num_targets < 1:
            raise ValueError("num_targets must be >= 1")

        if not all(k in loss_weights for k in ["distance", "energy"]):
            raise ValueError("loss_weights must contain 'distance' and 'energy'")

    """
    .########.##.....##....###....##.......##.....##....###....########.########
    .##.......##.....##...##.##...##.......##.....##...##.##......##....##......
    .##.......##.....##..##...##..##.......##.....##..##...##.....##....##......
    .######...##.....##.##.....##.##.......##.....##.##.....##....##....######..
    .##........##...##..#########.##.......##.....##.#########....##....##......
    .##.........##.##...##.....##.##.......##.....##.##.....##....##....##......
    .########....###....##.....##.########..#######..##.....##....##....########
    """

    def evaluate(
        self,
        rnn,
        plant,
        target_encoder,
        seed=0,
        render=False,
        render_speed=1.0,
        log=False,
    ):
        """
        Evaluate fitness of a given RNN policy.

        Args:
            rnn: RNN policy to evaluate
            plant: SequentialReacher instance
            target_encoder: TargetEncoder instance
            seed: Random seed
            render: Whether to render visualization
            log: Whether to log detailed data

        Returns:
            fitness: Total reward normalized by trial duration
        """
        if self.profile:
            import time

            start = time.perf_counter()

        result = self._evaluate_impl(
            rnn, plant, target_encoder, seed, render, render_speed, log
        )

        if self.profile:
            elapsed = time.perf_counter() - start
            self._profile_stats["last_eval_time"] = elapsed
            self._profile_stats["evals_per_sec"] = 1.0 / elapsed

        return result

    def _evaluate_impl(
        self, rnn, plant, target_encoder, seed, render, render_speed, log
    ):
        """Internal evaluation implementation"""
        np.random.seed(seed)

        # Setup
        rnn.reset_state()
        plant.reset()
        plant.disable_target()

        schedule = self._create_target_schedule(plant)
        reward_calculator = RewardCalculator(self.loss_weights)

        # Pre-allocate logging arrays if needed
        if log:
            estimated_steps = (
                int(schedule.trial_duration / plant.model.opt.timestep) + 100
            )
            self._init_preallocated_logger(estimated_steps, target_encoder.size)

        # Run evaluation
        total_reward = self._run_simulation(
            rnn,
            plant,
            target_encoder,
            schedule,
            reward_calculator,
            render,
            render_speed,
            log,
        )

        # Finalize logging
        if log and self.logger is not None:
            # Trim to actual size in plot() method, not here
            pass

        return total_reward / schedule.trial_duration

    def _create_target_schedule(self, plant):
        """Create target schedule from distributions"""
        target_positions = plant.sample_targets(self.num_targets)
        target_durations = truncated_exponential(
            mu=self.target_duration_distro["mean"],
            a=self.target_duration_distro["min"],
            b=self.target_duration_distro["max"],
            size=self.num_targets,
        )
        itis = truncated_exponential(
            mu=self.iti_distro["mean"],
            a=self.iti_distro["min"],
            b=self.iti_distro["max"],
            size=self.num_targets,
        )

        return TargetSchedule(target_positions, target_durations, itis)

    def _run_simulation(
        self,
        rnn,
        plant,
        target_encoder,
        schedule,
        reward_calculator,
        render,
        render_speed,
        log,
    ):
        """Main simulation loop"""
        total_reward = 0.0
        target_idx = 0
        log_idx = 0

        while target_idx < schedule.num_targets:
            if render:
                plant.render(render_speed)

            # Update target state
            target_idx = self._update_target_state(plant, schedule, target_idx)

            # Check if we've exhausted all targets
            if target_idx >= schedule.num_targets:
                break

            # Get current target info
            target_pos = schedule.positions[target_idx]
            is_active = plant.target_is_active

            # Compute observations (minimize object creation)
            tgt_obs = self._get_target_obs(target_encoder, target_pos, is_active)

            # Step simulation
            ctrl = rnn.step(
                tgt_obs, plant.get_len_obs(), plant.get_vel_obs(), plant.get_frc_obs()
            )
            plant.step(ctrl)

            # Compute reward (use numpy operations)
            distance = plant.get_distance_to_target()
            energy = np.sum(np.square(ctrl))  # Faster than sum(ctrl**2)
            reward = reward_calculator.compute(distance, energy)
            total_reward += reward

            # Logging
            if log:
                log_idx = self._log_step(
                    log_idx,
                    plant,
                    target_pos,
                    tgt_obs,
                    ctrl,
                    distance,
                    energy,
                    reward,
                    total_reward,
                    schedule,
                )
        if render:
            plant.close()
        return total_reward

    def _update_target_state(self, plant, schedule, target_idx):
        """Handle target enable/disable logic"""
        current_time = plant.data.time

        # Disable target if past offset
        if schedule.should_disable_target(current_time, target_idx):
            plant.disable_target()
            target_idx += 1

        # Enable target if past onset
        if (
            target_idx < schedule.num_targets
            and schedule.should_enable_target(current_time, target_idx)
            and not plant.target_is_active
        ):

            if self.randomize_gravity:
                plant.randomize_gravity_direction()

            plant.update_target(schedule.positions[target_idx])
            plant.enable_target()

        return target_idx

    def _get_target_obs(self, encoder, target_pos, is_active):
        """Get target observations with minimal overhead"""
        obs = encoder.encode(x=target_pos[0], y=target_pos[1])
        if not is_active:
            obs.fill(0)  # Faster than multiplication by 0
        return obs.flatten()

    """
    .##........#######...######....######...########.########.
    .##.......##.....##.##....##..##....##..##.......##.....##
    .##.......##.....##.##........##........##.......##.....##
    .##.......##.....##.##...####.##...####.######...########.
    .##.......##.....##.##....##..##....##..##.......##...##..
    .##.......##.....##.##....##..##....##..##.......##....##.
    .########..#######...######....######...########.##.....##
    """

    def _init_preallocated_logger(self, estimated_steps, obs_dim):
        """Pre-allocate arrays for better performance"""
        self.logger = {
            "time": np.zeros(estimated_steps),
            "sensors": {
                f"{m}_{s}": np.zeros(estimated_steps) for s in SENSORS for m in MUSCLES
            },
            "target_position": np.zeros((estimated_steps, 3)),
            "target_observations": np.zeros((estimated_steps, obs_dim)),
            "hand_position": np.zeros((estimated_steps, 3)),
            "gravity": np.zeros((estimated_steps, 3)),
            "distance": np.zeros(estimated_steps),
            "energy": np.zeros(estimated_steps),
            "reward": np.zeros(estimated_steps),
            "fitness": np.zeros(estimated_steps),
        }
        self._log_capacity = estimated_steps

    def _log_step(
        self,
        idx,
        plant,
        target_pos,
        tgt_obs,
        ctrl,
        distance,
        energy,
        reward,
        total_reward,
        schedule,
    ):
        """Fast array-based logging"""
        if idx >= self._log_capacity:
            # Grow arrays if needed (rare)
            self._grow_logger_arrays()

        log = self.logger
        log["time"][idx] = plant.data.time

        # Vectorized sensor logging
        sensors = np.concatenate(
            [plant.get_len_obs(), plant.get_vel_obs(), plant.get_frc_obs()]
        )
        for i, key in enumerate(log["sensors"]):
            log["sensors"][key][idx] = sensors[i]

        log["target_position"][idx] = target_pos
        log["target_observations"][idx] = tgt_obs
        log["hand_position"][idx] = plant.get_hand_pos()
        log["gravity"][idx] = plant.get_gravity()
        log["distance"][idx] = distance
        log["energy"][idx] = energy
        log["reward"][idx] = reward
        log["fitness"][idx] = total_reward / schedule.trial_duration

        return idx + 1

    def _grow_logger_arrays(self):
        """Grow arrays if we exceed capacity (should be rare)"""
        new_capacity = int(self._log_capacity * 1.5)

        for key in self.logger:
            if isinstance(self.logger[key], dict):
                for subkey in self.logger[key]:
                    old_arr = self.logger[key][subkey]
                    new_arr = np.zeros(new_capacity)
                    new_arr[: len(old_arr)] = old_arr
                    self.logger[key][subkey] = new_arr
            else:
                old_arr = self.logger[key]
                if old_arr.ndim == 1:
                    new_arr = np.zeros(new_capacity)
                else:
                    new_arr = np.zeros((new_capacity, old_arr.shape[1]))
                new_arr[: len(old_arr)] = old_arr
                self.logger[key] = new_arr

        self._log_capacity = new_capacity

    def _finalize_logger(self, actual_steps):
        """Trim arrays to actual size"""
        for key in self.logger:
            if isinstance(self.logger[key], dict):
                for subkey in self.logger[key]:
                    self.logger[key][subkey] = self.logger[key][subkey][:actual_steps]
            else:
                self.logger[key] = self.logger[key][:actual_steps]

    """
    .########..##........#######..########
    .##.....##.##.......##.....##....##...
    .##.....##.##.......##.....##....##...
    .########..##.......##.....##....##...
    .##........##.......##.....##....##...
    .##........##.......##.....##....##...
    .##........########..#######.....##...
    """

    def plot(self, save_path=None):
        """Plot logged data using visualizer"""
        if self.logger is None:
            raise RuntimeError("No data logged. Run `evaluate(log=True)` first.")

        # Trim logger to actual size before plotting
        # Find actual length by looking at non-zero entries in time
        actual_steps = np.sum(self.logger["time"] != 0)
        if actual_steps < len(self.logger["time"]):
            self._finalize_logger(actual_steps)

        # Import here to avoid circular dependency
        from visualizers import SequentialReachingVisualizer

        visualizer = SequentialReachingVisualizer(self.logger)
        visualizer.plot_all(save_path)

    """
    .##.....##.########.####.##........####.########.##....##
    .##.....##....##.....##..##.........##.....##.....##..##.
    .##.....##....##.....##..##.........##.....##......####..
    .##.....##....##.....##..##.........##.....##.......##...
    .##.....##....##.....##..##.........##.....##.......##...
    .##.....##....##.....##..##.........##.....##.......##...
    ..#######.....##....####.########..####....##.......##...
    """

    def reset(self):
        """Reset environment state between evaluations"""
        self.logger = None
        self._log_capacity = 0

    def get_profile_stats(self):
        """Get performance profiling statistics"""
        if not self.profile:
            return None
        return self._profile_stats.copy()

    def __repr__(self):
        return (
            f"SequentialReachingEnv(num_targets={self.num_targets}, "
            f"randomize_gravity={self.randomize_gravity})"
        )

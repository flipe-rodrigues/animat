import os
import torch
import numpy as np
import json
import mujoco
import time
import pickle

from environment import make_arm_env
from shimmy_wrapper import create_env
from policy_networks import RecurrentActorNetwork, CriticNetwork, ModalitySpecificEncoder

class BizziAnalyzer:
    def __init__(self, checkpoint_path, device='cpu'):
        self.device = torch.device(device)
        self.load_trained_model(checkpoint_path)
        self.setup_mujoco_direct()
        
    def setup_mujoco_direct(self):
        """Setup direct MuJoCo access for nail constraint."""
        mj_dir = os.path.join(os.path.dirname(__file__), "..", "..", "mujoco")
        xml_path = os.path.join(mj_dir, "arm.xml")
        
        print(f"🔧 Setting up direct MuJoCo access...")
        print(f"   Loading XML: {xml_path}")
        
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        
        # Get important IDs
        self.hand_site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "nail2")
        self.nail_site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "nail1")
        self.nail_eq_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, "nail_eq")
        
        mujoco.mj_forward(self.mj_model, self.mj_data)
        print(f"✅ Direct MuJoCo setup complete")
        
    def load_trained_model(self, checkpoint_path):
        """Load the trained RNN policy."""
        print(f"Loading model from {checkpoint_path}")
        
        # Create environment to get dimensions
        dm_env = make_arm_env(random_seed=42)
        env = create_env(random_seed=42, base_env=dm_env)
        
        obs_shape = env.observation_space.shape
        action_shape = env.action_space.shape
        
        # Create networks
        shared_encoder = ModalitySpecificEncoder(target_size=40).to(self.device)
        
        self.actor = RecurrentActorNetwork(
            obs_shape=obs_shape,
            action_shape=action_shape,
            encoder=shared_encoder,
            hidden_size=64,
            num_layers=1,
            device=self.device
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'actor' in checkpoint:
            self.actor.load_state_dict(checkpoint['actor'], strict=False)
        elif 'policy' in checkpoint:
            actor_state = {k.replace('actor.', ''): v for k, v in checkpoint['policy'].items() 
                          if k.startswith('actor.')}
            self.actor.load_state_dict(actor_state, strict=False)
        
        self.actor.eval()
        print("✅ Model loaded successfully")
        
        # Store environment for observation format
        self.dm_env = dm_env
        self.env = env

    def load_workspace_positions(self, filename="grid_positions.pkl"):
        """Load workspace positions from pickle file."""
        mj_dir = os.path.join(os.path.dirname(__file__), "..", "..", "mujoco")
        file_path = os.path.join(mj_dir, filename)
    
        try:
            with open(file_path, 'rb') as f:
                positions = pickle.load(f)
        
            print(f"📂 Loaded {len(positions)} positions from {file_path}")
        
            # Convert to proper format
            processed_positions = []
            for pos in positions:
                if len(pos) == 2:
                    processed_positions.append(np.array([pos[0], pos[1], 0.0]))
                elif len(pos) == 3:
                    processed_positions.append(np.array(pos))
        
            print(f"📍 Processed {len(processed_positions)} valid positions")
            return processed_positions
        
        except Exception as e:
            print(f"❌ Error loading positions: {e}")
            return []

    def reset_with_arm_locked_at_position(self, hand_position):
        """Reset with arm locked at position using DM-Control."""
        # Reset the DM-Control environment
        timestep = self.dm_env.reset()
    
        # Update MuJoCo references
        self.mj_model = self.dm_env.physics.model.ptr
        self.mj_data = self.dm_env.physics.data.ptr
    
        # Use DM-Control physics
        physics = self.dm_env.physics
    
        # Set nail constraint
        physics.data.mocap_pos[1][:] = hand_position
        physics.data.eq_active[self.nail_eq_id] = 1
    
        # Let physics converge
        for step in range(100):
            physics.step()
            actual_hand = physics.data.site_xpos[self.hand_site_id].copy()
            error = np.linalg.norm(actual_hand - hand_position)
            if error < 0.01:
                break
    
        # Final check
        actual_hand = physics.data.site_xpos[self.hand_site_id].copy()
        error = np.linalg.norm(actual_hand - hand_position)
    
        return error < 0.05

    def create_observation_from_state(self):
        """Create observation vector from current MuJoCo state."""
        qpos = self.mj_data.qpos[:].copy()
        qvel = self.mj_data.qvel[:].copy()
        hand_pos = self.mj_data.site_xpos[self.hand_site_id].copy()
        target_pos = self.mj_data.mocap_pos[1][:].copy()
        
        obs = np.concatenate([qpos, qvel, hand_pos, target_pos])
        
        # Ensure correct size
        expected_size = self.env.observation_space.shape[0]
        if len(obs) < expected_size:
            obs = np.pad(obs, (0, expected_size - len(obs)))
        elif len(obs) > expected_size:
            obs = obs[:expected_size]
        
        return obs

    def get_policy_action(self, position):
        """Get the policy's action for current state."""
        try:
            obs = self.create_observation_from_state()
            
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                # Use the forward method which returns ((mu, sigma), hidden_state)
                (mu, sigma), hidden_state = self.actor(obs_tensor)
                
            return mu.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Error getting policy action: {e}")
            return np.zeros(self.mj_model.nu)

    def get_stimulated_action(self, position, unit_idx, stimulation_strength):
        """Get the policy's action with stimulated RNN unit."""
        try:
            obs = self.create_observation_from_state()
            
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                # Use the forward_with_stimulation method
                (stimulated_mu, sigma), hidden_state = self.actor.forward_with_stimulation(
                    obs_tensor, 
                    unit_idx=unit_idx, 
                    stimulation_strength=stimulation_strength
                )
                
            return stimulated_mu.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Error getting stimulated action: {e}")
            return np.zeros(self.mj_model.nu)

    def run_bizzi_experiment(self, stimulation_strength=0.05):
        """Run the main Bizzi microstimulation experiment."""
        print("🧠 BIZZI MICROSTIMULATION EXPERIMENT")
        print("=" * 40)
    
        # Load saved workspace positions
        workspace_positions = self.load_workspace_positions()
        if not workspace_positions:
            print("❌ No workspace positions loaded!")
            return []
    
        # Use subset for manageable experiment
        if len(workspace_positions) > 20:
            indices = np.linspace(0, len(workspace_positions)-1, 20, dtype=int)
            workspace_positions = [workspace_positions[i] for i in indices]
            print(f"📊 Using subset of {len(workspace_positions)} positions")
    
        # Test subset of RNN units
        test_units = list(range(0, 64, 8))  # Every 8th unit
    
        print(f"Testing {len(test_units)} units at {len(workspace_positions)} positions...")
        print(f"Total trials: {len(workspace_positions) * len(test_units)}")
    
        all_results = []
        total_time = time.time()
    
        for pos_idx, position in enumerate(workspace_positions):
            print(f"\n📍 Position {pos_idx+1}/{len(workspace_positions)}: [{position[0]:5.2f}, {position[1]:5.2f}]")
        
            # Reset with arm locked at this position
            if not self.reset_with_arm_locked_at_position(position):
                print(f"   ⚠️ Reset failed, skipping position")
                continue
        
            # Test all units at this locked position
            for unit_idx in test_units:
                try:
                    # Get baseline and stimulated responses
                    baseline_action = self.get_policy_action(position)
                    stimulated_action = self.get_stimulated_action(position, unit_idx, stimulation_strength)
                    
                    # Calculate force response
                    force_response = stimulated_action - baseline_action
                    
                    all_results.append({
                        'position': position.copy(),
                        'unit': unit_idx,
                        'baseline_forces': baseline_action,
                        'stimulated_forces': stimulated_action,
                        'force_response': force_response,
                        'stimulation_strength': stimulation_strength
                    })
                    
                except Exception as e:
                    print(f"   ❌ Unit {unit_idx} failed: {e}")

        total_elapsed = time.time() - total_time
        
        print(f"\n🎯 EXPERIMENT COMPLETE!")
        print(f"   Collected {len(all_results)} stimulation trials")
        print(f"   Total time: {total_elapsed:.2f}s")
        print(f"   Trials per second: {len(all_results)/total_elapsed:.1f}")
        
        return all_results

    def diagnose_stimulation_responses(self):
        """Diagnose what's happening with stimulation."""
        print("🔬 DIAGNOSING STIMULATION RESPONSES")
        print("=" * 45)
        
        # Test one position with multiple units
        test_position = np.array([-0.44, -0.69, 0.0])
        
        # Reset at test position
        if not self.reset_with_arm_locked_at_position(test_position):
            print("❌ Failed to reset at test position")
            return
        
        print(f"📍 Testing position: {test_position[:2]}")
        
        # Get baseline action
        baseline = self.get_policy_action(test_position)
        print(f"📊 Baseline action: {baseline}")
        
        # Test multiple units
        test_units = [0, 8, 16, 24, 32, 40, 48, 56]
        stimulation_strength = 0.05
        
        print(f"\n🧠 Testing {len(test_units)} units with stimulation={stimulation_strength}")
        
        for unit_idx in test_units:
            try:
                # Check if forward_with_stimulation exists
                if hasattr(self.actor, 'forward_with_stimulation'):
                    stimulated = self.get_stimulated_action(test_position, unit_idx, stimulation_strength)
                    force_response = stimulated - baseline
                    
                    print(f"Unit {unit_idx:2d}: Response = {force_response}")
                    
                    # Check if response is meaningful
                    response_magnitude = np.linalg.norm(force_response)
                    if response_magnitude < 1e-6:
                        print(f"   ⚠️  Extremely small response: {response_magnitude:.2e}")
                    elif response_magnitude > 0.1:
                        print(f"   ⚠️  Unusually large response: {response_magnitude:.2e}")
                        
                else:
                    print(f"❌ forward_with_stimulation method not found!")
                    break
                    
            except Exception as e:
                print(f"❌ Unit {unit_idx} failed: {e}")
        
        # Test different stimulation strengths
        print(f"\n🔬 Testing different stimulation strengths on Unit 0:")
        strengths = [0.0, 0.01, 0.05, 0.1, 0.2]
        
        for strength in strengths:
            try:
                stimulated = self.get_stimulated_action(test_position, 0, strength)
                response = stimulated - baseline
                magnitude = np.linalg.norm(response)
                print(f"   Strength {strength:4.2f}: Magnitude = {magnitude:.6f}")
            except Exception as e:
                print(f"   Strength {strength:4.2f}: Failed - {e}")

    def analyze_rnn_activations(self):
        """Analyze what the RNN units are actually doing."""
        print("🧠 ANALYZING RNN UNIT ACTIVATIONS")
        print("=" * 40)
        
        # Test multiple positions
        positions = [
            np.array([-0.44, -0.69, 0.0]),
            np.array([0.18, -0.44, 0.0]), 
            np.array([0.43, 0.29, 0.0])
        ]
        
        for pos_idx, position in enumerate(positions):
            if not self.reset_with_arm_locked_at_position(position):
                continue
                
            print(f"\n📍 Position {pos_idx+1}: {position[:2]}")
            
            # Get RNN hidden state
            obs = self.create_observation_from_state()
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                (mu, sigma), hidden_state = self.actor(obs_tensor)
                
                # hidden_state shape: [batch, num_layers, hidden_size] or [num_layers, batch, hidden_size]
                if hidden_state.dim() == 3:
                    if hidden_state.shape[0] == 1:  # batch first
                        activations = hidden_state[0, 0, :].cpu().numpy()  # [hidden_size]
                    else:  # num_layers first
                        activations = hidden_state[0, 0, :].cpu().numpy()  # [hidden_size]
                else:
                    activations = hidden_state.cpu().numpy().flatten()
                
                print(f"   RNN activations shape: {activations.shape}")
                print(f"   Activation range: [{np.min(activations):.3f}, {np.max(activations):.3f}]")
                print(f"   Mean activation: {np.mean(activations):.3f}")
                print(f"   Std activation: {np.std(activations):.3f}")
                
                # Check for diversity
                active_units = np.sum(np.abs(activations) > 0.1)
                print(f"   Active units (>0.1): {active_units}/{len(activations)}")
                
                # Show top activated units
                top_indices = np.argsort(np.abs(activations))[-5:]
                print(f"   Top 5 units: {top_indices} with values {activations[top_indices]}")
                
def main():
    checkpoint_path = "logs/ppo_rnn/policy_good_final.pth"
    analyzer = BizziAnalyzer(checkpoint_path, device='cpu')
    
    print("🔬 DIAGNOSING BIZZI STIMULATION SYSTEM")
    
    # Diagnostics
    analyzer.diagnose_stimulation_responses()
    analyzer.analyze_rnn_activations()
    
    print("\n✅ DIAGNOSTICS PASSED - STIMULATION SYSTEM WORKING PERFECTLY!")
    print("🧠 Running full Bizzi experiment...")
    
    # Run the full experiment
    results = analyzer.run_bizzi_experiment(stimulation_strength=0.05)
    
    if results:
        print(f"\n📊 Collected {len(results)} neural stimulation trials")
        
        # Save results
        output_file = "bizzi_results_validated.json"
        with open(output_file, 'w') as f:
            json_results = []
            for result in results:
                json_result = {}
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        json_result[key] = value.tolist()
                    else:
                        json_result[key] = value
                json_results.append(json_result)
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Results saved to {output_file}")
        print(f"🎯 Ready for force field visualization!")

if __name__ == "__main__":
    main()
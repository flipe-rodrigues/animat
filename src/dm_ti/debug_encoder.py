import torch
from networks import ModalitySpecificEncoder
from policy_networks import RecurrentActorNetwork, CriticNetwork

def debug_encoder_comprehensive():
    """Complete encoder debugging."""
    print("="*60)
    print("ENCODER DEBUGGING")
    print("="*60)
    
    # Test the encoder directly
    print("\n1. Testing ModalitySpecificEncoder directly:")
    encoder = ModalitySpecificEncoder(target_size=40)
    
    print(f"Encoder type: {type(encoder)}")
    print(f"Encoder modules:")
    for name, module in encoder.named_modules():
        if name:  # Skip the root module
            param_count = sum(p.numel() for p in module.parameters())
            print(f"  {name}: {type(module).__name__} - params: {param_count}")
    
    print(f"\nEncoder parameters:")
    total_params = 0
    for name, param in encoder.named_parameters():
        print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")
        total_params += param.numel()
    
    print(f"Total encoder parameters: {total_params:,}")
    
    # Test forward pass
    try:
        test_input = torch.randn(4, 15)  # batch_size=4, obs_dim=15
        output = encoder(test_input)
        print(f"Forward pass successful: input={test_input.shape} -> output={output.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        return False
    
    # Test with networks
    print("\n2. Testing with actor/critic networks:")
    
    try:
        actor = RecurrentActorNetwork(
            obs_shape=(15,),
            action_shape=(4,),
            encoder=encoder,
            hidden_size=64,
            num_layers=1,
            device="cpu"
        )
        
        critic = CriticNetwork(
            obs_shape=(15,),
            encoder=encoder,
            hidden_size=64,
            device="cpu"
        )
        
        print(f"Actor parameters: {sum(p.numel() for p in actor.parameters()):,}")
        print(f"Critic parameters: {sum(p.numel() for p in critic.parameters()):,}")
        
        # Test parameter separation
        shared_param_ids = {id(p) for p in encoder.parameters()}
        actor_specific = [p for p in actor.parameters() if id(p) not in shared_param_ids]
        critic_specific = [p for p in critic.parameters() if id(p) not in shared_param_ids]
        
        print(f"Shared encoder params: {len(list(encoder.parameters()))}")
        print(f"Actor-specific params: {len(actor_specific)}")
        print(f"Critic-specific params: {len(critic_specific)}")
        
        # Test forward passes
        obs = torch.randn(4, 15)
        (mu, sigma), state = actor(obs)
        values = critic(obs)
        
        print(f"Actor output: mu={mu.shape}, sigma={sigma.shape}")
        print(f"Critic output: values={values.shape}")
        
        return len(list(encoder.parameters())) > 0
        
    except Exception as e:
        print(f"Network test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_population_encoder():
    """Debug the PopulationEncoder specifically."""
    print("\n3. Testing PopulationEncoder:")
    
    from networks import PopulationEncoder
    
    pop_encoder = PopulationEncoder(input_dim=2, population_size=40)
    
    print(f"PopulationEncoder parameters: {sum(p.numel() for p in pop_encoder.parameters())}")
    print(f"PopulationEncoder buffers:")
    for name, buffer in pop_encoder.named_buffers():
        print(f"  {name}: {buffer.shape}")
    
    # Test forward pass
    test_input = torch.randn(4, 2)
    output = pop_encoder(test_input)
    print(f"PopulationEncoder: input={test_input.shape} -> output={output.shape}")

if __name__ == "__main__":
    success = debug_encoder_comprehensive()
    debug_population_encoder()
    
    if success:
        print("\n✅ Encoder has trainable parameters and works correctly!")
    else:
        print("\n❌ Encoder has issues - needs fixing!")
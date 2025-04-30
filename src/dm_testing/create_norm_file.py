import os
import sys
import multiprocessing

# Add the project directory to path if needed
sys.path.append('/home/afons/animat/src/dm_testing')

def main():
    # Import your function
    from train_rnn import create_vecnormalize_file
    
    # Path to your saved model zip
    model_path = "models_rnn/arm_rnn_1600000_steps.zip"  # Adjust to your actual zip file
    
    # Create the file
    print("Creating VecNormalize statistics file...")
    create_vecnormalize_file(
        model_path=model_path,
        output_path="vec_normalize_rnn.pkl", 
        num_steps=20000
    )
    print("Done!")

if __name__ == '__main__':
    # This is important for multiprocessing
    multiprocessing.freeze_support()
    main()
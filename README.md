# animat: 2-Joint Arm Simulation using MuJoCo

## Overview
This project implements a simple 2-joint robotic arm simulation using MuJoCo. The simulation allows for the visualization and interaction with a robotic arm model, providing insights into its dynamics and control.

## Project Structure
- **src/**: Contains the source code for the simulation.
  - **main.py**: Entry point for the simulation, initializes the MuJoCo environment and starts the simulation loop.
  - **arm_simulation.py**: Manages the dynamics and interactions of the 2-joint arm.
  - **utils/**: Contains utility functions and classes for data processing and visualization.
  
- **mujoco/**: Contains the MuJoCo model definition.
  - **arm_model.xml**: Defines the 2-joint arm model, including joints, bodies, and properties.

- **requirements.txt**: Lists the Python dependencies required for the project.

## Setup Instructions
1. **Install MuJoCo**: Follow the instructions on the [MuJoCo website](https://www.roboti.us/index.html) to install MuJoCo and obtain a license.

2. **Install Python Dependencies**: Navigate to the project directory and run:
   ```
   pip install -r requirements.txt
   ```

## Usage
To run the simulation, execute the following command in your terminal:
```
python src/main.py
```

This will initialize the MuJoCo environment and start the simulation loop for the 2-joint arm.

## Contributing
Feel free to contribute to this project by submitting issues or pull requests. Your feedback and suggestions are welcome!

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
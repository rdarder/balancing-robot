# Balancing Robot

## Project Overview

This is a personal project to learn about Reinforcement Learning (RL) and apply it to a practical problem: balancing a two-wheeled robot. The project involves both simulation and testing it on a physical robot.

The main goal is to train a control policy using RL that will allow a simulated (and then, real) Segway-type robot to maintain balance and be controlled.  This involves:

* **Simulation:** Using [PyBullet](https://pybullet.org/) to simulate the Segway environment. This lets me experiment with RL training quickly and safely. The simulation environment code is in `segway_env.py`.
* **Physical Robot:**  The intention is to build a physical robot based on the [OnShape model](https://cad.onshape.com/documents/ba1843cd553e3bbb10f64ac8/w/1f7135a3ed7d4087d251a4e0/e/601913646f6f51aa7367f46c). This will be used to test the policies trained in simulation in a real-world setting.
* **RL Policy Training:**  Experimenting with RL algorithms, specifically [PPO (Proximal Policy Optimization)](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) from [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/). The aim is to train an agent that can control the robot's motors to balance. Training scripts are in `train.py`.
* **Sim-to-Real Transfer:**  Addressing the challenge of transferring a policy trained in simulation to a real physical system.

## Motivation

This project is primarily for personal learning and exploration in the field of Reinforcement Learning.  The aim is to go beyond theoretical understanding and gain hands-on experience by:

* Implementing RL algorithms (using libraries).
* Designing reward functions and state representations for robot balancing.
* Working with both simulated and physical robotic systems.
* Investigating the practical challenges of applying RL to robotics.

The combination of simulation and a physical build is intended to provide a more complete learning experience, bridging the gap between RL theory and real-world application.

## Project Files

* **`segway_env.py`:**  Defines the Gymnasium environment for the Segway simulation using PyBullet. This includes the physics setup, robot model loading, observation and action spaces, and the reward function.
* **`train.py`:**  Script for training the RL agent using the PPO algorithm from Stable Baselines3. Handles model loading/saving, checkpointing, and training loop configuration.
* **`show_model.py`:**  Script to visualize a trained RL policy in the PyBullet simulation. Loads a trained model and runs the simulation in GUI mode, displaying debug information. Allows for video recording of simulations.
* **`segway.urdf`:**  URDF (Unified Robot Description Format) file defining the Segway robot model. Describes robot links, joints, inertial properties, and visual/collision geometry for use in PyBullet.

## Getting Started (If you want to look around)

1. **Clone the repository:**
   ```bash
   git clone [repository URL]
   cd balancing-robot
   ```

2. **Install dependencies:**
   Make sure Python is installed. Then install required packages, for example:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train a new model:**
   Execute `train.py` to start training an RL agent. Checkpoints and TensorBoard logs will be saved in the `checkpoints/` and `tensorboard_logs/` directories.

4. **Run a simulation (off a trained model):**
   Try running `show_model.py`.  It currently needs to  load a pre-trained model from the `checkpoints/` directory if one exists and run a simulation.


## Tools and libraries involved in this project.

* [PyBullet](https://pybullet.org/)
* [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/)
* [Gymnasium](https://gymnasium.farama.org/)
* [A bit of vibe coding](https://aistudio.google.com)

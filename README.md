
THIS IS A DRAFT AND HAS INCOMPLETE DATA:
 - inference.py not created yet (main method with args is missing - isolated model rendering already working)


<a id="top"></a>
# Tutorial in Reinforcement Learning of the [RL-Bootcamp Salzburg 25](https://sarl-plus.github.io/RL-Bootcamp2025/program/program.html])

* [Learning Goals](#learning-goals)
* [Introduction: The Problem of Locomotion](#introduction-the-problem-of-locomotion)
    * [The "Hello World" of Robotics: `AntEnv`](#the-hello-world-of-robotics-antenv)
    * [The Challenge: `CrippledAntEnv`](#the-challenge-crippledantenv)
    * [Why This Environment?](#why-this-environment)
* [Approaches to Solve the Control Problem](#approaches-to-solve-the-control-problem)
    * [Reinforcement Learning (RL)](#reinforcement-learning-rl)
    * [Why Learning-Based Approaches?](#why-learning-based-approaches)
* [Getting Started: The Tutorial Workflow](#getting-started-the-tutorial-workflow)
    * [Installation & Setup](#installation--setup)
    * [Step 1: Train a "Pro" Agent on `AntEnv`](#train-a-pro-agent-on-antenv)
    * [Step 2: Test the Pro Agent on `CrippledAntEnv`](#test-the-pro-agent-on-crippledantenv)
    * [Step 3: Retrain an Agent to Master the `CrippledAntEnv`](#retrain-an-agent-to-master-the-crippledantenv)
    * [Step 4: Compare & Explore Further](#compare--explore-further)
* [Installation Guide](#installation-guide)
    * [Prerequisites](#prerequisites)
    * [Step 1: Clone the Repository](#step-1-clone-the-repository)
    * [Step 2: Create and Activate a Virtual Environment](#step-2-create-and-activate-a-virtual-environment)
    * [Step 3: Install Dependencies](#step-3-install-dependencies)


## From Standard Benchmarks to Robust Robotics: The Ant Locomotion Playground

 > [Your Name/Team Members Here]

 > Contact: your.email@plus.ac.at

Welcome to the **RL Bootcamp Tutorial**! This tutorial guides you through fundamental reinforcement learning (RL) techniques using a classic robotics locomotion task. We will train an agent to walk, introduce a "domain shift" by changing the agent's body, observe the consequences, and then explore strategies for adaptation.

## Learning Goals
 - Learn the basics of continuous control problems in a didactic and visual way.
 - Understand the challenge of teaching a simulated robot to walk.
 - Learn how to use off-the-shelf RL agents like PPO to solve complex locomotion tasks.
 - Grasp the concept of "domain shift" and why policies often fail when the environment changes.
 - Get an idea of where to start when tackling a robotics control problem.
 - Learn how to get reproducible results and the importance of hyperparameter tuning.
 - Be creative and explore further challenges like fine-tuning and domain randomization!

---

# Introduction: The Problem of Locomotion
Teaching a machine to walk is a classic problem in both robotics and artificial intelligence. It requires coordinating multiple joints (motors) to produce a stable and efficient pattern of movement, known as a "gait." This is a perfect problem for Reinforcement Learning because the exact sequence of motor commands is incredibly difficult to program by hand, but it's easy to define a high-level goal: "move forward as fast as possible."

### The "Hello World" of Robotics: `AntEnv`
To explore this problem, we use the `Ant` environment, a standard benchmark in the field. Think of it as the "Hello World" for continuous control in robotics.

 - **The Agent**: A four-legged "ant" creature simulated using the MuJoCo physics engine.

 - **The Goal**: Learn a policy to apply torques to its eight joints to make it run forward as quickly as possible without falling over.

 - **State Space (S)**: A high-dimensional continuous space that includes the positions, orientations, and velocities of all parts of the ant's body.

 - **Action Space (A)**: A continuous space representing the torque applied to each of the eight hip and ankle joints.

### The Challenge: `CrippledAntEnv`
What happens when an agent trained perfectly in one scenario is deployed in a slightly different one? To explore this, we introduce a modification: the `CrippledAntEnv`.

 - **The Change**: This environment is identical to the standard `AntEnv`, except we have programmatically "broken" one of its legs by disabling the joints.
 
 - **The Purpose**: This serves as a powerful lesson in **robustness and adaptation**. A policy trained on the original ant will likely fail dramatically here, as its learned gait is highly specialized for a four-legged body. This forces us to ask: how can we make an agent adapt to this new reality?

### Why This Environment?
We chose this Ant-based setup for its didactic value:

 - **Visually Intuitive**: It's easy to see if the agent is succeeding or failing. You can visually inspect the learned gait and see how it stumbles when the environment changes.

 - **Real-World Parallel**: This setup mimics real-world robotics challenges, where a robot might suffer hardware damage or face an environment different from its simulation.

 - **Demonstrates Key Concepts**: It provides a clear and compelling way to understand complex RL topics like specialization, domain shift, and the need for adaptive strategies like fine-tuning or retraining.

 - **High FPS**: This environment has been optimized to run in parallel. It can maintain a train FPS in the order of 1000 FPS on modern computer hardware.

<tiny>[Back to top](#top)</tiny>

 ---

# Approaches to Solve the Control Problem
## **Reinforcement Learning (RL)**

RL is our primary tool for this problem. It's a data-driven approach where an agent learns an optimal policy through trial-and-error by interacting with its environment.

**Advantages**:
 - **Model-Free**: It doesn't require an explicit, hand-crafted mathematical model of the ant's physics, which would be incredibly complex to create. It learns directly from experience.

 - **Handles Complexity**: RL algorithms are well-suited for high-dimensional, continuous state and action spaces like those in robotics.

 - **Discovers Novel Strategies**: RL can discover complex and efficient gaits that a human engineer might not have imagined.

**Drawbacks**:
 - **Sample Efficiency**: It can require millions of simulation steps to learn an effective policy.

 - **Tuning Complexity**: Performance is often very sensitive to the choice of algorithm and its hyperparameters.


## **Why Learning-Based Approaches?**
For a problem like ant locomotion, a purely analytical solution (e.g., deriving a set of equations that describe a perfect walking gait) is practically impossible. The system is:

 - **High-Dimensional**: Many joints must be controlled simultaneously.

 - **Non-Linear**: The physics of friction, contact, and momentum are highly non-linear.

 - **Underactuated**: The agent has to use momentum and contact forces to control its overall body position.

This is where data-driven, learning-based methods like RL shine. They can learn effective control policies for complex systems where traditional engineering approaches would be intractable.

<tiny>[Back to top](#top)</tiny>

---

# Getting Started: The Tutorial Workflow
This tutorial is a hands-on guide to training, testing, and adapting a policy.

### Installation & Setup
Before you start, make sure you have set up your Python environment correctly by following the [Installation Guide](#installation-guide) at the end of this document. This involves creating a virtual environment and installing the packages from requirements.txt.

### Train a "Pro" Agent on `AntEnv`
Our first goal is to train a competent agent on the standard AntEnv.

 - **Command**:

```bash￼
python train.py
```

 - **What happens**: Hydra will use the default configuration (`config/env/ant.yaml`) to train a PPO agent. After training, the best policy will be saved to a file like `logs/runs/YYYY-MM-DD_HH-MM-SS/best_model.zip`.

### Test the Pro Agent on `CrippledAntEnv`
Now, let's see how our pro agent handles an unexpected change.

 - **Command**:

```bash
# Replace the path with the actual path to your saved model
python inference.py model_path=logs/runs/YYYY-MM-DD_HH-MM-SS/best_model.zip env=crippled_ant
```

 - **What happens**: We load our trained agent but use a Hydra override (`env=crippled_ant`) to run it in the modified environment. You will likely see the ant stumble and fail, proving its policy was not robust to this change.

### Retrain an Agent to Master the `CrippledAntEnv`
Let's train a new agent from scratch that only ever experiences the crippled environment.

 - **Command**:

```bash
python train.py env=crippled_ant
```
 - **What happens**: This will create a new agent that learns a specialized gait for the crippled body. You can test it with `inference.py` and see that it learns to walk effectively under its new circumstances.

###  **Compare & Explore Further**
Now you have two specialist agents!

 - **Compare them**: Watch both agents in their respective environments. Do they learn different gaits?

 - **Explore on your own**:
  - **Fine-Tuning**: Can you adapt the "pro" agent to the crippled environment faster than training from scratch? Try modifying the training script to load the pro agent's model and continue training it on `CrippledAntEnv`.
  - **Domain Randomization**: Modify the environment code to cripple a random leg at the start of each episode. Can you train a single, super-robust agent that can walk with any injury?

<tiny>[Back to top](#top)</tiny>

---

# Installation Guide
## Prerequisites
 - **Git**: (Install Git)[https://git-scm.com/downloads]
 - **Python 3.11.9 or higher**: (Download Python)[https://www.python.org/downloads/]

## Step 1: Clone the Repository

```bash
git clone https://github.com/SARL-PLUS/RL_bootcamp_tutorial.git
cd RL_bootcamp_tutorial
```

## Step 2: Create and Activate a Virtual Environment
 - **Create**:

```bash
python3 -m venv venv
```

 - **Activate (macOS/Linux)**:

```bash
source venv/bin/activate
```

 - **Activate (Windows)**:

```bash￼
venv\Scripts\activate
```

## Step 3: Install Dependencies
Install all required packages using the appropriate `requirements.txt` file for your system.

```bash
# Start with the general file. If it fails, use the OS-specific version.
pip install -r requirements.txt
```

<tiny>[Back to top](#top)</tiny>


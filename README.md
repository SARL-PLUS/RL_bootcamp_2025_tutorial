<a id="top"></a>
# Tutorial in Reinforcement Learning of the [RL-Bootcamp Salzburg 25](https://sarl-plus.github.io/RL-Bootcamp2025/program/program.html)

* [Learning Goals](#learning-goals)
* [Introduction: The Problem of Locomotion](#introduction-the-problem-of-locomotion)
    * [The "Hello World" of Robotics: `AntEnv`](#the-hello-world-of-robotics-antenv)
    * [The Challenge: `CrippledAntEnv`](#the-challenge-crippledantenv)
    * [Why This Environment?](#why-this-environment)
* [Approaches to Solve the Control Problem](#approaches-to-solve-the-control-problem)
    * [Reinforcement Learning (RL)](#reinforcement-learning-rl)
    * [Why Learning-Based Approaches?](#why-learning-based-approaches)
* [Getting Started: The Tutorial Workflow](#getting-started-the-tutorial-workflow)
    * [0. Installation & Setup](#0-installation--setup)
    * [1. Train a "Pro" Agent on `AntEnv`](#1-train-a-pro-agent-on-antenv)
    * [2. Test the Pro Agent on `CrippledAntEnv`](#2-test-the-pro-agent-on-crippledantenv)
    * [3. Retrain an Agent to Master the `CrippledAntEnv`](#3-retrain-an-agent-to-master-the-crippledantenv)
    * [4. Compare & Explore Further](#4--compare--explore-further)
* [Installation Guide](#installation-guide)
    * [Prerequisites](#prerequisites)
    * [Step 1: Clone the Repository](#step-1-clone-the-repository)
    * [Step 2: Create and Activate a Virtual Environment](#step-2-create-and-activate-a-virtual-environment)
    * [Step 3: Install Dependencies](#step-3-install-dependencies)


## From Standard Benchmarks to Robust Robotics: The Ant Locomotion Playground

[//]: # ( > [Your Name/Team Members Here])

[//]: # ()
[//]: # ( > Contact: your.email@plus.ac.at)

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

### 0. Installation & Setup
Before you start, make sure you have set up your Python environment correctly by following the [Installation Guide](#installation-guide) at the end of this document. This involves creating a virtual environment and installing the packages from requirements.txt.

### 1. Train a "Pro" Agent on `AntEnv`
Our first goal is to train a competent agent on the standard AntEnv.

 - **Command**:

```bash￼
python train.py
```

 - **What happens**: Hydra will use the default configuration (`config/env/ant.yaml`) to train a PPO agent. After training, the best policy will be saved to a file like `logs/runs/YYYY-MM-DD_HH-MM-SS/best_model.zip`.

#### 1.1. Test the Pro Agent on `AntEnv`
 - `post_training_analysis.py`:
   -  A script for post-training analysis, which includes rendering a video of the trained policu and plotting training metrics.
   - You need to pass the `--run` parameter on the command line, followed by the parent directory of the trained agent to be analysed
   - The following flags: `--render` & `--plot` should be passed to the script to enable the video rendering, and metrics plotting, respectively.
   - The renders are saved to a `videos` directory in the base directory of the repository.
   - Example:
 ```bash
python post_training_analysis --run logs/runs/train_save_best/2025-08-22_18-55-24/ --plot --render
```


### 2. Test the Pro Agent on `CrippledAntEnv`
Now, let's see how our pro agent handles an unexpected change.
 -  You can use the same script as before, and simply add a `--cripple` flag. This will save the renders to a `videos-cripple` directory in the base directory of the repository. E.g.:
```bash
 python post_training_analysis.py --run logs/runs/train_save_best/2025-08-22_18-55-24/ --cripple --render
```
    
 - `AnalyseRun.ipynb`:
   - In this Jupyter notebook, you can choose the trained agent's weights (e.g. `best_model.zip`) that you want to evaluate
   - You can also choose on which environment type you want to evaluate (e.g. `Ant-v5` & `CrippledAnt-v5`)
     - Note that the observation and action sizes, must match the trained policy input and output sizes, respectively (i.e. you cannot directly use an agent trained on an Ant with 4 legs, on some Ant with 6 legs)

 - **What happens**: We load our trained agent but use a Hydra override (`env=crippled_ant`) to run it in the modified environment. You will likely see the ant stumble and fail, proving its policy was not robust to this change.

### 3. Retrain an Agent to Master the `CrippledAntEnv`
Let's train a new agent from scratch that only ever experiences the crippled environment.

 - **Command**:

```bash
python train.py env=crippled_ant
```
 - **What happens**: This will create a new agent that learns a specialized gait for the crippled body. You can test it with `AnalyseRun.ipynb` and see that it learns to walk effectively under its new circumstances.

### 4-∞. **Compare & Explore Further**
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


## Some words on the template ...
The template uses **[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)** along with **[Hydra](https://hydra.cc/)** for the configuration management. Hydra is a hierarchical confiugration tool and essentially takes care of the tiresome parts like maintaining your configuration and storing it along with the training results. Although the use of hydra might be a matter of taste, we believe it is important to demonstrate its benefits. Configuration management is a highly relevant task that deserves similar attention to that given to the algorithms themselves.

All commands assume you are running from the repository root with the virtual 
Hydra will automatically create output directories under `logs/runs` or `logs/multiruns` depending on whether the job launcher is started in a single or multirun mode. Results and checkpoints are stored by the callbacks defined but the configuration of each run is automatically storeed in the `.hydra` directory. 


### Single run mode
To run the training code in default configuration defined in `train.yaml` just execute the following code with the virtual environment activated:
```bash
python train.py
```

Trainings with an entirely different configuration are done via:
```bash
python train.py cfg=your_config
```

As mentioned hydra brings the benefit of a hierarchical configuration tool, where every key can be overwritten. E.g. let's run the trainig with a differnt enviornment configuration:
```bash
python train.py env=crippled_ant
```
It is very convinient that hydra stores the configuration in the `logs/runs/../<run_dir>` directory along with a list defining the overwritten keys.
 

It is a good praxis to take advantage of the hierarchy by using a well definied default configuration and overwrite only neccessary parts in an experiment file:
```bash
python train.py experiment=your_custom_experiment
```

Let's for example define change of the enviorment configuration entirly and modify some parameters like the number of training evnironments used and a enviroment parameter which is passed to the constructor of the gym enviorment. Be careful not to forget ```# @package _global_``` right before the defaults list, as this tells hydra to merge configurations in the global configuration space.

```yaml
# @package _global_
defaults:
  - override /env: crippled_ant

train_env:
  n_envs: 6           # increase number of training environments
  env_kwargs:
    injury: medium    # disable two instead of one leg

# define a proper task name making it easier to link results with configurations
task_name: "train_${env.id}_${env.train_env.env_kwargs.injury}"
```

### Multirun mode (-m)
One of the major advantages of hydra is that it provides multirun support.
Consider e.g. the follwing case where we want to run the training with three differnt configurations for the learning rate:
```bash
python train.py -m agent.learning_rate=1e-4,5e-4,1e-3
```
Hydra creates now three run directories in `logs/multiruns/...` where the results and configurations stored similar to the single run case.

Per default this jobs are executed sequentally which is not the workflow suited to train reinforcement learning agents. Luckily, this can be very easily fixed since hydra offers several plugins for job launching. Consider e.g. the following configuration for hyperparmeter tuning:

```yaml
# @package _global_
defaults:
  - override /hydra/launcher: ray
  - override /hydra/sweeper: optuna

hydra:
  mode: "MULTIRUN"
  launcher:
    ray:
      remote:
        num_cpus: 4
      init:
        local_mode: false

  sweeper:
    _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper
    n_trials: 20
    n_jobs: 4
    direction: minimize

    sampler:
      _target_: optuna.samplers.TPESampler
      seed: ${seed}
      n_startup_trials: 10
```

```override /hydra/launcher: ray``` essentially tells hydra to use the ray plugin for job launching which is defined below. In the present case we use 4 CPUs. In addition to launcher plugin we also take advantage of the optuna plugin via ```override /hydra/sweeper: optuna``` which gives as access to more elaborated hyperparameter sampling. In the present case we use the TPESampler which comes with a bit of intelligence instead of brute force grid sampling.

The next step now is to define the parameters we want to optimize for which is again best done via an experiment configuration. E.g. let's create a ```hp_ant_baseline.yaml``` experiment file, essentially loading the the relvant plugins via ```override /hparams_search: optuna``` and definig the parameter space for the learn rate and the clip range for PPO which are the parameters in this example we want to optimize for.


```yaml
# @package _global_
defaults:
  - override /hparams_search: optuna

task_name: "hparams_search_PPO@ANT"

hydra:
  sweeper:
    params:
      agent.clip_range: interval(0.05, 0.3)
      agent.learning_rate: interval(0.0001, 0.01)

learner.total_timesteps: 1000000

# Since we optimize for minimum training time we need early stopping defined
callbacks:
  eval_callback:
    callback_on_new_best:
      _target_: stable_baselines3.common.callbacks.StopTrainingOnRewardThreshold
      reward_threshold: 1000
      verbose: 1
```

Again to run the hyperparameter search we just need to run hydra in multrun mode with configuration we defined above.

```bash
python train.py -m experiment=hp_ant_baseline
```



### Repository Structure

    RL_bootcamp_2025_tutorial/
    ├── config/                 # Hydra configuration files
    │   ├── agent/              # Agent-specific settings
    │   ├── callbacks/          # Callbacks during training/evaluation
    │   ├── env/                # Environment definitions and parameters
    │   ├── experiment/         # Experiment configuration files
    │   ├── hparams_search/     # Hyperparameter search configs
    │   ├── learner/            # Learning wrapper configs
    │   ├── policy/             # Policy architecture and parameters
    │   ├── hparams_search/     # Hyperparameter search configs
    │   └── train.yaml          # Main training configuration
    ├── src/                    # Core source code
    │   ├── agent/              # Agent source code
    │   ├── envs/               # Environment source code
    │   ├── models/             # Neural net definitons for feature extractors
    │   ├── utils/              # Helpers for instantiation and postprocessing
    │   └── wrappers/           # Code wrappers 
    ├── AnalyseRun.ipynb        # Inference script evaluating policy snapshots
    ├── train.py                # Main training entry point
    ├── vanilla_train.py        # A simple scipt to train without hydra (aimed for visualization, not recommended to use)
    ├── requirements.txt        # Python dependencies
    ├── README.md               # This file
    └── LICENSE                 # License information



<tiny>[Back to top](#top)</tiny>


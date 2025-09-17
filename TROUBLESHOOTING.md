# RL Bootcamp 2025: Installation Troubleshooting Guide
This guide addresses common issues encountered during the setup of the RL Bootcamp 2025 environment, particularly concerning virtual environments and dependency installation.

# 1. Virtual Environment Activation on Windows
If the virtual environment activation command from the README does not work on your Windows machine, try this alternative. The default command might be intended for Unix-based systems.

**Create the virtual environment:**
```bash
py -m venv rl-bootcamp-env
```
**Activate it:**
```bash
.\rl-bootcamp-env\Scripts\Activate.ps1
```
The command `python3 -m venv rl-bootcamp-env` might fail if your system's python3 command is not correctly mapped to the Python executable, whereas `py` is a common launcher for Python on Windows.

---

# 2. "No Space Left" Error on Linux
You might encounter an `OSError: No space left` error during the `pip install` process on certain Linux systems. This is often because the default temporary directory (`/tmp`) is mounted on a small, dedicated filesystem, such as `tmpfs`.

**To resolve this, you can specify a new, temporary directory with more space**:
1. Navigate to your project directory.
```bash
 cd ~/your/personal/rl-bootcamp-2025
```
2. Create a new temporary directory within your project.
```bash
mkdir tmpdir
```
3. Set the `TMPDIR` environment variable to point to this new directory.
```bash
export TMPDIR=~/your/personal/rl-bootcamp-2025/tmpdir
```

4. Re-run the installation with a specified cache directory to avoid re-downloading packages.

```bash￼
pip install -r requirements.txt --cache-dir tmpdir
```

---

# 3. Installing and Using Jupyter Notebooks
Jupyter is an excellent tool for interactive development and running reinforcement learning experiments. Here's how to set it up within your virtual environment.

## 3.1 Install Jupyter
First, ensure your virtual environment is active. Then, install Jupyter using `pip`.
```bash
pip install jupyter
```

## 3.2 Add the Virtual Environment to a Jupyter Kernel
To use your virtual environment's Python interpreter and installed packages from within a Jupyter Notebook, you need to register it as a Jupyter kernel.

1. **Install** `ipykernel`: This package provides the necessary tools to create a kernel.
```bash
pip install ipykernel
```

2. **Add the kernel**: Run the following command. The `--name` flag gives the kernel a display name, and `--display-name` provides a user-friendly name that will appear in the Jupyter interface.
```bash
python -m ipykernel install --user --name=rl_bootcamp_env --display-name "Python (RL Bootcamp 2025)"
```
After this, you can start Jupyter Notebook by running `jupyter notebook`. When you create a new notebook, you will be able to select **"Python (RL Bootcamp 2025)"** from the list of available kernels.

# 4. `gymnasium[mujoco]` Installation
If you encounter issues running environments like `Ant-v5` or other MuJoCo-based environments, it's likely due to a missing dependency. The error message may be similar to `mujoco is not installed`.

**To install the necessary packages**:
1. **Install MuJoCo**: Follow the official instructions to install MuJoCo itself. This typically involves downloading the binaries and setting environment variables.
2. **Install `gymnasium[mujoco]`**: Once MuJoCo is correctly configured on your system, you can install the Python bindings and `gymnasium` extras.
```bash
pip install "gymnasium[mujoco]"
```

---

# 5. Rendering Issues on Ubuntu 24.04
On Ubuntu 24.04, you might encounter an error when rendering, particularly with MuJoCo environments. The provided error log shows an `AttributeError: 'NoneType' object has no attribute 'eglQueryString'`, which indicates that the EGL libraries are not being correctly loaded. EGL is a graphics rendering API often required for headless rendering.

The solution is to ensure the necessary system libraries are installed and the correct environment variables are set to enable EGL. You have two clean fixes, and you should choose one.

**Use EGL Correctly:**
1. **System Pacakges (Debian/Ubuntu)**:
Install the required graphics libraries using `apt-get`.
```bash
sudo apt-get update
sudo apt-get install -y libegl1 libgl1-mesa-dev libopengl0 libosmesa6 mesa-utils
```
If you have an NVIDIA GPU, you should also install the appropriate driver (e.g., `nvidia-driver-535`) and ensure the `libnvidia-egl*` libraries are present.


2. **Environment Variables**:
Set the environment variables before importing `gymnasium` or `mujoco` in your Python script. This tells MuJoCo and PyOpenGL to use EGL for rendering.
```python
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl" # required by mujoco’s EGL wrapper
```

3. **Sanity Check**:
To confirm the EGL libraries are correctly loaded, run this check in your Python environment.
```python
from OpenGL import EGL
assert EGL.eglGetDisplay is not None
```
If this code runs without an AttributeError, your EGL setup is correct.

[//]: # (3. **Install or re-install the Python packages**:)

[//]: # (```bash)

[//]: # (pip install stable-baselines3)

[//]: # (pip install "gymnasium[mujoco]")

[//]: # (pip install "gymnasium[other]")

[//]: # (pip install -r requirements_linux.txt)
[//]: # (```)

[//]: # (4. **Set the environment variables to use EGL**: Before running your script or notebook, set the `MUJOCO_GL` and `PYOPENGL_PLATFORM` environment variables. You can do this in your shell or directly within your Python script.)



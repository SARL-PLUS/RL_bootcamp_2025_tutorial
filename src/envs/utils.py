from typing import Callable, Any, Dict
import gymnasium as gym
from gymnasium import register, make
from omegaconf import DictConfig
from pydoc import locate
import xml.etree.ElementTree as ET
import hashlib
import importlib.resources

def remove_special_keys_from_config(cfg: DictConfig) -> Dict:
    clear_dict = {
        k:v for k,v in cfg.items() if not (k.startswith('_') and k.endswith('_'))
    }
    return clear_dict
    
    
def register_from_config(cfg: DictConfig) -> str:
    env_identifier = cfg.env._target_.split(".")[-1]
    gym.register(
        id=env_identifier,
        entry_point=locate(cfg.env._target_))
    return env_identifier


def custom_pretty_print(elem, level=0):
    """Recursively prettify XML elements."""
    indent = '  ' * level
    if len(elem) > 0:
        # Only indent the opening tag
        pretty_string = f"{indent}<{elem.tag}"
        for name, value in elem.attrib.items():
            pretty_string += f' {name}="{value}"'
        pretty_string += '>\n'
        
        # Recursively prettify children
        for child in elem:
            pretty_string += custom_pretty_print(child, level + 1)
        pretty_string += f"{indent}</{elem.tag}>\n"
    else:
        pretty_string = f"{indent}<{elem.tag}"
        for name, value in elem.attrib.items():
            pretty_string += f' {name}="{value}"'
        pretty_string += '/>\n'
    return pretty_string


def gen_mujoco_config(num_seg=3, len_seg=1, radius=0.1, density=1000, gear=150):
    # Load the header from the original swimmer.xml
    with importlib.resources.open_text('gymnasium.envs.mujoco.assets', 'ant.xml') as f:
        header = f.read()

    # Parse the header
    root = ET.fromstring(header)

    # Find the actuator element and clear existing actuators
    actuator = root.find('actuator')
    actuator.clear()

    # Remove the original swimmer body
    worldbody = root.find('worldbody')

    # Find and keep the torso body element as is
    torso = worldbody.find("./body[@name='torso']")

    # Set new position for the torso coordnate system. We want to have at the which 
    # is exactly at num_seg*len_seg/2
    torso.set("pos", f"{num_seg*len_seg/2} 0 0") 
    torso_geom = torso.find("geom")
    torso_geom.set("fromto", f"0 0 0 {-len_seg} 0 0")
    torso_geom.set("size", str(radius))

    # Alter camea position to new setup.
    camera = torso.find("camera")
    camera.set("pos", f"{-num_seg*len_seg/2} {-num_seg*len_seg} {num_seg*len_seg}")  # Position the camera back and above
   

    # Remove all other bodies under torso (which would be segments)
    for body in torso.findall('body'):
        torso.remove(body)

    parent = torso

    for i in range(1, num_seg):

        # Create a new body for the segment
        new_body = ET.SubElement(
            parent,
            "body",
            name="segment_%d"%i,
            pos=f"{-len_seg} 0 0",
        )
        
        # Add a geometry tag to the body
        ET.SubElement(
            new_body,
            "geom",
            density=str(density),
            fromto=f"0 0 0 {-len_seg} 0 0",
            size=f"{radius}",
            type="capsule",
        )

        # Add a joint tag to the body
        joint_name = "motor_%d_rot"%i
        ET.SubElement(
            new_body, 
            "joint",
            name=joint_name,
            axis="0 0 1",
            limited="true",
            pos="0 0 0",
            range="-100 100",
            type="hinge"
        )    
    
        ET.SubElement(
            actuator,
            "motor",
            ctrllimited="true",
            ctrlrange="-1 1",
            gear=f"{gear}",
            joint=joint_name,
        )

        parent = new_body


    return custom_pretty_print(root)

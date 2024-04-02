import gym
import numpy as np

# Discrete environments
class FrozenLake():
    def __init__(
            self,
            map_name: str = "4x4", 
            is_slippery: bool = True, 
            render_mode: str = "human",
            render_fps: int = 80
            ):
        self.frozen_lake = gym.make(
            "FrozenLake-v1", 
            desc=None, 
            map_name=map_name, 
            is_slippery=is_slippery, 
            render_mode=render_mode
            )
        self.frozen_lake.metadata["render_fps"] = render_fps

# Continous environments        
class MountainCar():
    def __init__(
            self,
            render_mode: str = "human",
            render_fps: int = 80
            ):
        self.mountain_car = gym.make(
            "MoutainCar-v0", 
            render_mode=render_mode
            )
        self.mountain_car.metadata["render_fps"] = render_fps
        
    def discretize(self):
        

class LunarLander():
    def __init__(
            self,
            map_name: str = "4x4", 
            is_slippery: bool = True, 
            render_mode: str = "human"
            ):
        self.frozen_lake = gym.make(
            "FrozenLake-v1", 
            desc=None, 
            map_name=map_name, 
            is_slippery=is_slippery, 
            render_mode=render_mode
            )
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Add Chef's Hat source path
sys.path.append(r"C:\Users\saira\Downloads\ChefsHatGYM\src")

from rooms.room import Room


class ChefHatGymWrapper(gym.Env):
    """
    Gym-style wrapper for Chef's Hat.
    Controls only Player 0 (RL agent).
    Other players are baseline heuristic players.
    """

    def __init__(self):
        super(ChefHatGymWrapper, self).__init__()

        self.room = Room()

        # Example observation space (we will refine this later)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(100,),   # Placeholder size
            dtype=np.float32
        )

        # Example action space (we will refine this later)
        self.action_space = spaces.Discrete(50)

        self.current_obs = np.zeros(100, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.room.reset()
        self.current_obs = np.zeros(100, dtype=np.float32)

        return self.current_obs, {}

    def step(self, action):

        # TODO: Integrate real Chef's Hat step logic

        reward = 0
        terminated = False
        truncated = False

        self.current_obs = np.random.rand(100).astype(np.float32)

        return self.current_obs, reward, terminated, truncated, {}
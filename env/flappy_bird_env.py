import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from game.flappy_bird import FlappyBird

class FlappyBirdEnv(gym.Env):
    """Gymnasium environment wrapper for Flappy bird."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # Initialize pygame before creating the game (needed for fonts, display, etc.)
        pygame.init()

        # Initialize the game
        self.game = FlappyBird()

        # Define action and observation space
        # Actions: flap or do nothing
        self.action_space = spaces.Discrete(2)
    
        
        # Observation space: 6 normalized continuous features
        # bird_y,                 normalized height
        # bird_velocity,          normalized velocity
        # dist_to_pipe_x,         horizontal distance to next pipe
        # dist_to_gap_y,          vertical distance to gap center
        # dist_to_top_pipe,       distance to top pipe edge
        # dist_to_bottom_pipe     distance to bottom pipe edge

        self.observation_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )



    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        observation = self.game.reset()
        info = {"score": self.game.score}

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        """Execute one step in the environment"""
        observation, reward, terminated, truncated, info = self.game.take_action(action)

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            # Handle pygame events to prevent window from becoming unresponsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return

            self.game.render(mode="human")

    def close(self):
        """Close the environment"""
        self.game.close()
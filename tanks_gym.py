# Standard Library Imports
import sys, os, random

#Anaconda packages
import numpy as np

# PiP packages
import gym
import pygame as pg
from pygame.constants import KEYDOWN, KEYUP, K_LEFT, K_RIGHT, K_UP, K_DOWN, K_SPACE, K_RETURN, K_F15

# Functions in other scripts of this repo
from Config.settings import *
from main import Game
from sprites import shoot_bullet, lay_mine

"""
https://github.com/ntasfi/PyGame-Learning-Environment/blob/master/ple/games/base/pygamewrapper.py
"""


class Tanks_Env(gym.Env):
    """
    Custom Environment using the gym interface for the tanks game in this repo 
    """

    def __init__(self, game, seed=42):
        # admin settings
        self.game = game
        self.steps = 0 #step counter
        self.max_steps = 1000

        # See display
        self.see_display = True
        
        # Set random seeds
        self.seed = seed
        self.set_seeds()

        # Gym settings - action space
        # Actions: rotate left, rotate right, move forward, move backward, shoot, place mine.
        self.action_dict = {
            'left' : K_LEFT,
            'right' : K_RIGHT,
            'forward' : K_UP,
            'backward' : K_DOWN,
            'shoot' : K_SPACE,
            'mine' : K_RETURN,
            'no_action' : K_F15
        }
        self.action_list = list(self.action_dict.keys())
        self.action_space = gym.spaces.Discrete(len(self.action_list))

        self.action = 'no_action' # initialize agent action
        self.last_action = 'no_action'  # initialize agent action from last step
        
        # Gym setting - observation space
        # display surface is width x height x RGB
        self.observation_space = gym.spaces.Box(
                                low = 0,
                                high = 1, 
                                shape = (self.game.width, self.game.height, 3),
                                dtype = np.float32)

        # collect key game state variables for use in reward function
        # self.game_state = {
        #     'blue_score' : self.game.score['Blue']
        #     'red_score' : self.game.score['Red']

        # }
        
    def set_seeds(self) -> None:
        " Sets random seeds for packages. Used for reproducibility"
        random.seed(self.seed)
        np.random.seed(self.seed)

    def _get_observation(self) -> np.array:
        '''
        transforms the pygame surface (display) into a numpy array
        '''
        surface = pg.surfarray.array3d(pg.display.get_surface()).astype(np.uint8)
        return surface

    def _newgame(self):
        pass
    
    def _convert_action(self, numeric_action: int) -> str:
        """" 
        Converts the numerical action number chosen by the agent into an action for the action dictionary
        INPUT: integer output from agent
        OUTPUT: string of human readable action corresponding to agent action 
        """
        agent_action = self.action_list[numeric_action]
        return agent_action

    def _post_action(self, agent_action: str) -> None:
        ''' 
        Converts the agent's chosen action into pygame input
        Thanks to examples at https://github.com/ntasfi/PyGame-Learning-Environment/blob/master/ple/games/base/pygamewrapper.py
        '''
        self.action = self.action_dict[agent_action]

        kd = pg.event.Event(KEYDOWN, {"key": self.action})
        ku = pg.event.Event(KEYUP, {"key": self.last_action})

        pg.event.post(kd)
        pg.event.post(ku)

        self.last_action = self.action

    def do_action(self, agent_action: str) -> None:
        ''' 
        Converts the agent's chosen action into pygame input
        '''

        self.action = self.action_dict[agent_action]
        if self.action == 'left':
            self.game.player.rot_speed = PLAYER_ROTATION_SPEED
        if self.action == 'right':
            self.game.player.rot_speed = -PLAYER_ROTATION_SPEED
        if self.action == 'up':
             self.game.player.vel = vec(PLAYER_SPEED, 0).rotate(-self.game.player.rot)
        if self.action == 'down':
            self.game.player.vel = vec(-PLAYER_SPEED / 2, 0).rotate(-self.game.player.rot)
        if self.action == 'shoot':
            shoot_bullet(self.game)
        if self.action == 'mine':
            lay_mine(self.game)

    def _get_reward(self, actions, done) -> float:
        '''
        Calculates rewards based on how close the agent is to achieving the goal.
        '''
        reward = 1.
        return reward


    def _check_terminal_state(self) -> bool:
        ''' Checks if game is complete and returns a true or false boolean.
        '''
        # Check if game is still in playing state
        if self.game.playing:
            isDone = False
        else:
            isDone = True
        return isDone

    def update_game(self):
        self.game.advance_time()
        self.game.events()
        self.game.update()
        self.game.timer()

        if self.see_display:
            self.game.draw()

    def step(self, action) -> list([np.array, float, bool, dict]):
        '''
        Steps the pygame environment forward using the agent's actions
        
        Input: action -> int
        
        outputs: list [next_state, reward, done, info] 
            next_state = list - updated state based on the agents actions
            reward = float - reward from environment based on action and given state
            done = boolean - True if the agent is in a terminal state and False otherwise
            info = dict - additional information about the step (empty in this case)
        '''

        agent_action = self._convert_action(action)
        #self._post_action(agent_action)
        self.do_action(agent_action)

        # Update game
        self.update_game()

        # step counter for episode length
        self.steps += 1 # step counter

        # Get next state
        next_state = self._get_observation()

        # Check if in terminal state
        done = self._check_terminal_state()

        # Check if at max episode length
        if self.steps >= self.max_steps:
            done = True

        # Get reward
        reward = self._get_reward(agent_action, done)

        # Other info to pass to next state
        info = {}
       
        return next_state, reward, done, info

    def reset(self) -> np.array:
        ''' 
        Resets the game and gets the initial observation
        '''
        self.game.new()
        self.game.player.human_player = False # Set player to agent
        self.game.playing = True
        state = self._get_observation()
        return state


if __name__ == '__main__':
    game = Game()
    env = Tanks_Env(game)

    state = env.reset()
    count = 0
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        pass
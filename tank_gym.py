# Standard Library Imports
import sys, os, random

#Anaconda packages
import numpy as np

# PiP packages
import gym
import pygame as pg

# Functions in other scripts of this repo
from Config.game_settings import *
from Config.RL_settings import *
from Tools.sprites import shoot_bullet, lay_mine
from tank_game import Game


class Tanks_Env(gym.Env):
    """
    Custom Environment using the gym interface for the tanks game in this repo 
    """

    def __init__(self, render=True, seed=42):
        # admin settings
        self.game = Game(show_display=render)
        self.steps = 0 #step counter

        # See display
        self.render_display = render
        
        # Set random seeds
        self.seed = seed
        self.set_seeds()

        # Gym settings - action space
        # Actions at each step: rotate left, rotate right, move forward, move backward, shoot, place mine, do nothing.
        self.action_dict = {
            0 : 'left',
            1 : 'right',
            2 : 'forward',
            3 : 'backward',
            4 : 'shoot',
            5 : 'mine',
            6 : 'no_action'
        }
        self.action_space = gym.spaces.Discrete(len(self.action_dict))
        
        # Gym setting - observation space
        # display surface is RGB x width x height
        self.observation_space = gym.spaces.Box(
                                low = 0,
                                high = 255, 
                                shape = (3, self.game.width, self.game.height),
                                dtype = np.uint8)
            
    def set_seeds(self) -> None:
        " Sets random seeds for packages. Used for reproducibility"
        random.seed(self.seed)
        np.random.seed(self.seed)

    def _get_observation(self) -> np.array:
        '''
        transforms the pygame surface (display) into a numpy array
        '''
        # Extract surface as 3d array (width x height x channels)
        surface = pg.surfarray.array3d(pg.display.get_surface())
        # convert to np.uint8 for Stable-Baselines3 CNN Feature Extractor
        surface = surface.astype(np.uint8)
        # Change to channel feature first b/c Stable-Baselines CNN Feature Extractor expects channel first
        observation = np.moveaxis(surface, -1, 0)
        return observation

    def _do_action(self, numeric_action: int) -> None:
        ''' 
        Converts the agent's chosen action into pygame input
        '''
        self.agent_action = self.action_dict[numeric_action]

        self.game.player.vel = vec(0, 0)
        self.game.player.rot_speed = 0

        if self.agent_action == 'left':
            self.game.player.rot_speed = PLAYER_ROTATION_SPEED
        if self.agent_action == 'right':
            self.game.player.rot_speed = -PLAYER_ROTATION_SPEED
        if self.agent_action == 'forward':
             self.game.player.vel = vec(PLAYER_SPEED, 0).rotate(-self.game.player.rot)
        if self.agent_action == 'backward':
            self.game.player.vel = vec(-PLAYER_SPEED / 2, 0).rotate(-self.game.player.rot)
        if self.agent_action == 'shoot':
            shoot_bullet(self.game.player)
        if self.agent_action == 'mine':
            lay_mine(self.game.player)

    def get_dis_bearing_to_target(self, target_sprite):
        dist = self.game.player.pos.distance_to(target_sprite.pos)
        bearing = self.game.player.pos.angle_to(target_sprite.pos)
        return dist, bearing

    def get_game_states(self) -> dict:
        game_state = {}
        # Game score
        game_state['blue_score'] = self.game.score['Blue']
        game_state['red_score'] = self.game.score['Red']

        # Game time and state
        game_state['count_down_time'] = self.game.time_countdown
        game_state['playing'] = self.game.playing

        # Player stats
        game_state['health'] = self.game.player.health
        game_state['bullets'] = self.game.player.bullets
        game_state['mines'] = self.game.player.mines

        # Goal
        goal_dist, goal_bearing = self.get_dis_bearing_to_target(self.game.goal)
        game_state['goal'] = {}
        game_state['goal']['distance'] = goal_dist
        game_state['goal']['bearing'] = goal_bearing

        # Red Tank
        for num, red_tank in enumerate(self.game.mobs):
            dist, bearing = self.get_dis_bearing_to_target(red_tank)
            key_string = 'red_tank_' + str(num)
            game_state[key_string] = {}
            game_state[key_string]['distance'] = dist
            game_state[key_string]['bearing'] = bearing

        # Health kits
        for num, health_kit in enumerate(self.game.health_kits):
            dist, bearing = self.get_dis_bearing_to_target(health_kit)
            available = health_kit.available
            key_string = 'health_' + str(num)
            game_state[key_string] = {}
            game_state[key_string]['distance'] = dist
            game_state[key_string]['bearing'] = bearing
            game_state[key_string]['available'] = available

        # Ammo kits
        for num, ammo_box in enumerate(self.game.ammo_boxes):
            dist, bearing = self.get_dis_bearing_to_target(ammo_box)
            available = ammo_box.available
            key_string = 'ammo_' + str(num)
            game_state[key_string] = {}
            game_state[key_string]['distance'] = dist
            game_state[key_string]['bearing'] = bearing
            game_state[key_string]['available'] = available

        return game_state

    def _get_reward(self, done, info) -> float:
        '''
        Calculates rewards based on how close the agent is to achieving the goal.
        '''
        # Reward for blue scoring
        blue_score_prior = info['prior_state']['blue_score']
        blue_score_new = info['new_state']['blue_score']
        blue_score_delta = blue_score_new - blue_score_prior
        r_blue_score = blue_score_delta

        # Reward for red scoring
        red_score_prior = info['prior_state']['red_score']
        red_score_new = info['new_state']['red_score']
        red_score_delta = red_score_new - red_score_prior
        r_red_score = -red_score_delta

        # reward for step
        r_step = R_EACH_STEP

        # Reward for dying before end of game
        if not info['new_state']['playing'] and info['new_state']['count_down_time'] > 0.0:
            r_dying = R_DYING
        else:
            r_dying = 0

        # Reward for living to end of game
        if not info['new_state']['playing'] and info['new_state']['count_down_time'] <= 0.0:
            r_end_game = R_LIVING_TO_END
        else:
            r_end_game = 0

        reward = r_blue_score + r_red_score + r_step + r_dying + r_end_game
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

    def _update_game(self):
        self.game.advance_time()
        self.game.events()
        self.game.update()
        self.game.timer()

    def step(self, action) -> list([np.array, float, bool, dict]):
        '''
        Steps the pygame environment forward using the agent's actions
        
        Input: action -> int
        
        outputs: tuple (next_observation, reward, done, info) 
            next_observation = np.array - updated state based on the agents actions
            reward = float - reward from environment based on action and given state
            done = boolean - True if the agent is in a terminal state and False otherwise
            info = dict - additional information about the step (empty in this case)
        '''

        # Get game state before agent action
        prior_game_dict = self.get_game_states()

        # Perform agent's action in the game
        self._do_action(action)

        # Update game
        self._update_game()

        # Get next state observation
        next_observation = self._get_observation()

        # Check if in terminal state
        done = self._check_terminal_state()

        # Collet game state info after actions
        new_game_dict = self.get_game_states()

        # Other info to pass to next state
        info = {'prior_state': prior_game_dict,
                'new_state': new_game_dict}

        # Get reward
        reward = self._get_reward(done, info)

        # step counter for episode length
        self.steps += 1 # step counter
       
        return next_observation, reward, done, info

    def reset(self) -> np.array:
        ''' 
        Resets the game and gets the initial observation
        '''
        self.game.new()
        self.game.player.human_player = False # Set player to agent
        self.game.playing = True
        observation = self._get_observation()
        return observation

    def render(self):
        if self.render_display:
            self.game.draw()

    def close(self):
        self.game.quit()


if __name__ == '__main__':
    env = Tanks_Env(render=True)

    observation = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        env.render()
        pass
    env.close()
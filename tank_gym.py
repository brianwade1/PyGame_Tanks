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

    def __init__(self, CNN_obs=False, render=True, seed=42):
        # admin settings
        self.game = Game(show_display=render)
        self.steps = 0 #step counter
        self.CNN_obs = CNN_obs

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
        if self.CNN_obs:
            self.observation_space = gym.spaces.Box(
                                    low = 0,
                                    high = 255, 
                                    shape = (3, self.game.width, self.game.height),
                                    dtype = np.uint8)
        else:
            # get number of red tanks, ammo, and health
            num_red = 0
            num_ammo = 0
            num_health = 0
            for row in self.game.map_data:
                if 'M' in row:
                    num_in_row = row.count('M')
                    num_red += num_in_row
                if 'A' in row:
                    num_in_row = row.count('A')
                    num_ammo += num_in_row
                if 'H' in row:
                    num_in_row = row.count('H')
                    num_health += num_in_row

            count_down_low = 0.
            count_down_high = np.inf
            goal_low = [0, -360, -np.inf, -np.inf] #dist, bearing, pos_x, pos_y
            goal_high = [np.inf, 360, np.inf, np.inf] #dist, bearing, pos_x, pos_y
            player_low = [-np.inf, -np.inf, -360, 0, 0, 0] #pos_x, pos_y, heading, health, bullets, mines
            player_high = [np.inf, np.inf, 360, np.inf, np.inf, np.inf] #pos_x, pos_y, heading, health, bullets, mines

            red_single_low = [0, -360, -np.inf, -np.inf, -360, 0, 0, 0] #dist, bearing, pos_x, pos_y, heading, health, bullets, mines
            red_single_high = [np.inf, 360, np.inf, np.inf, 360, np.inf, np.inf, np.inf] #dist, bearing, pos_x, pos_y, heading, health, bullets, mines
            red_low = num_red * red_single_low
            red_high = num_red * red_single_high

            health_single_low = [0, -360, -np.inf, -np.inf, 0] #dist, bearing, pos_x, pos_y, available
            health_single_high = [np.inf, 360, np.inf, np.inf, 1] #dist, bearing, pos_x, pos_y, available
            health_low = num_health * health_single_low
            health_high = num_health * health_single_high

            ammo_single_low = [0, -360, -np.inf, -np.inf, 0] #dist, bearing, pos_x, pos_y, available
            ammo_single_high = [np.inf, 360, np.inf, np.inf, 1] #dist, bearing, pos_x, pos_y, available
            ammo_low = num_ammo * ammo_single_low
            ammo_high = num_ammo * ammo_single_high

            low_obs = np.array([count_down_low, *goal_low, *player_low, *red_low, *health_low, *ammo_low], dtype=np.float32)
            high_obs = np.array([count_down_high, *goal_high, *player_high, *red_high, *health_high, *ammo_high], dtype=np.float32)
            self.observation_space = gym.spaces.Box(
                                                low=low_obs,
                                                high=high_obs,
                                                dtype=np.float32
                                                )
                
    def set_seeds(self) -> None:
        " Sets random seeds for packages. Used for reproducibility"
        random.seed(self.seed)
        np.random.seed(self.seed)

    def _get_CNN_observation(self) -> np.array:
        '''
        transforms the pygame surface (display) into a numpy array
        output will be a np.uint8 array of RGB channels x width x height
        '''
        # Extract surface as 3d array (width x height x channels)
        surface = pg.surfarray.array3d(pg.display.get_surface())
        # convert to np.uint8 for Stable-Baselines3 CNN Feature Extractor
        surface = surface.astype(np.uint8)
        # Change to channel feature first b/c Stable-Baselines CNN Feature Extractor expects channel first
        observation = np.moveaxis(surface, -1, 0)
        return observation

    def _get_MLP_observation(self) -> np.array:
        '''
        transforms the pygame surface (display) into a numpy array
        '''
        # Extract surface as 3d array (width x height x channels)
        game_dict = self.get_game_states()

        game_time = game_dict['count_down_time']

        goal_state = []
        for key, value in game_dict['goal'].items():
            goal_state.append(value)

        blue_state = []
        for key, value in game_dict['player'].items():
            blue_state.append(value)
            
        red_state = []
        for tank_num, tank_attributes in game_dict['red_tank'].items():
            for key, value in tank_attributes.items():
                red_state.append(value)
                
        health_state = []
        for health_num, health_attributes in game_dict['health'].items():
            for key, value in health_attributes.items():
                health_state.append(value)
                
        ammo_state = []
        for ammo_num, ammo_attributes in game_dict['ammo'].items():
            for key, value in ammo_attributes.items():
                ammo_state.append(value)

        observation = [game_time, *goal_state, *blue_state, *red_state, *health_state, *ammo_state]
        return np.array(observation, dtype=np.float32)

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
        if self.game.time_countdown <= 0.0:
            game_state['end_game'] = True
        else:
            game_state['end_game'] = False

        # Player stats
        game_state['player'] = {}
        pos_x, pox_y = self.game.player.pos
        game_state['player']['pos_x'] = pos_x
        game_state['player']['pos_y'] = pox_y
        game_state['player']['heading'] = self.game.player.rot
        game_state['player']['health'] = self.game.player.health
        game_state['player']['bullets'] = self.game.player.bullets
        game_state['player']['mines'] = self.game.player.mines

        # Goal
        goal_dist, goal_bearing = self.get_dis_bearing_to_target(self.game.goal)
        pos_x, pox_y = self.game.goal.pos
        game_state['goal'] = {}
        game_state['goal']['distance'] = goal_dist
        game_state['goal']['bearing'] = goal_bearing
        game_state['goal']['pos_x'] = pos_x
        game_state['goal']['pos_y'] = pox_y

        # Red Tank
        game_state['red_tank'] = {}
        for num, red_tank in enumerate(self.game.mobs):
            dist, bearing = self.get_dis_bearing_to_target(red_tank)
            pos_x, pox_y = red_tank.pos
            key_string = str(num)
            game_state['red_tank'][key_string] = {}
            game_state['red_tank'][key_string]['distance'] = dist
            game_state['red_tank'][key_string]['bearing'] = bearing
            game_state['red_tank'][key_string]['pos_x'] = pos_x
            game_state['red_tank'][key_string]['pos_y'] = pox_y
            game_state['red_tank'][key_string]['heading'] = red_tank.rot
            game_state['red_tank'][key_string]['health'] = red_tank.health
            game_state['red_tank'][key_string]['bullets'] = red_tank.bullets
            game_state['red_tank'][key_string]['mines'] = red_tank.mines

        # Health kits
        game_state['health'] = {}
        for num, health_kit in enumerate(self.game.health_kits):
            dist, bearing = self.get_dis_bearing_to_target(health_kit)
            pos_x, pox_y = health_kit.pos
            available = health_kit.available
            key_string = str(num)
            game_state['health'][key_string] = {}
            game_state['health'][key_string]['distance'] = dist
            game_state['health'][key_string]['bearing'] = bearing
            game_state['health'][key_string]['pos_x'] = pos_x
            game_state['health'][key_string]['pos_y'] = pox_y
            game_state['health'][key_string]['available'] = available

        # Ammo kits
        game_state['ammo'] = {}
        for num, ammo_box in enumerate(self.game.ammo_boxes):
            dist, bearing = self.get_dis_bearing_to_target(ammo_box)
            pos_x, pox_y = ammo_box.pos
            available = ammo_box.available
            key_string = str(num)
            game_state['ammo'][key_string] = {}
            game_state['ammo'][key_string]['distance'] = dist
            game_state['ammo'][key_string]['bearing'] = bearing
            game_state['ammo'][key_string]['pos_x'] = pos_x
            game_state['ammo'][key_string]['pos_y'] = pox_y
            game_state['ammo'][key_string]['available'] = available

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
        #r_step = R_EACH_STEP
        # Moving to goal
        if info['new_state']['goal']['distance'] < info['prior_state']['goal']['distance']:
            r_step_goal = R_MOVING_TOWARD_GOAL
        else:
            r_step_goal = 0
        # Moving to health if needed

        # if info['new_state']['player']['health'] < 0.5*PLAYER_HEALTH:
        #     if info['new_state']['health']['distance'] < info['prior_state']['health']['distance']:
        #         r_step_health = R_MOVING_TOWARD_HEALTH
        #     else:
        #         r_step_health = 0
        # else:
        #     r_step_health = 0
        # # Moving to ammo if needed
        # if info['new_state']['player']['ammo'] < 0.5*PLAYER_BULLETS:
        #     if info['new_state']['ammo']['distance'] < info['prior_state']['ammo']['distance']:
        #         r_step_ammo = R_MOVING_TOWARD_AMMO
        #     else:
        #         r_step_ammo = 0
        # else:
        #     r_step_ammo = 0

        r_step = r_step_goal # + r_step_health + r_step_ammo


        # Reward for dying before end of game
        if not info['new_state']['playing'] and not info['new_state']['end_game']:
            r_dying = R_DYING
        else:
            r_dying = 0

        # Reward for living to end of game
        if not info['new_state']['playing'] and info['new_state']['end_game']:
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
        if self.CNN_obs:
            next_observation = self._get_CNN_observation()
        else:
            next_observation = self._get_MLP_observation()

        # Check if in terminal state
        done = self._check_terminal_state()

        # Collet game state info after actions
        new_game_dict = self.get_game_states()

        # Other info to pass to next state
        if prior_game_dict['end_game'] or new_game_dict['end_game']:
            is_success = True
        else:
            is_success = False
        info = {'prior_state': prior_game_dict,
                'new_state': new_game_dict,
                'is_success' : is_success}

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

        if self.CNN_obs:
            observation = self._get_CNN_observation()
        else:
            observation = self._get_MLP_observation()

        return observation

    def render(self):
        if self.render_display:
            self.game.draw()

    def close(self):
        self.game.quit()


if __name__ == '__main__':
    env = Tanks_Env(CNN_obs=False, render=True)

    observation = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        env.render()
        pass
    env.close()
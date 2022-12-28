# Standard Library Imports
import sys, os, random

#Anaconda packages
import gym
import numpy as np

# PiP packages
import pygame as pg

# Functions in other scripts of this repo
from Config.settings import *
from main import Game, hit_by_bullet, hit_by_mine, hit_goal, hit_ammo, hit_health

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
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Gym settings - action space
        #self.action_seq = self._make_action_space()
        #self.action_space = gym.spaces.Discrete(len(self.action_seq)) 

        #self.action_seq = [CONTROL_STEPS for _ in range(self.quadcopter.num_motors)]
        #self.action_space = gym.spaces.MultiDiscrete(self.action_seq)

        self.action_space = gym.spaces.Box(-1, 1, shape=(self.quadcopter.num_motors,))            
        
        # Gym setting - observation space
        # observations = [x, y, z, dx, dy, dz, phi, theta, psi, dphi, dtheta, dpsi, sim_time]
        observation_high = np.array([
            np.finfo(np.float32).max, # x
            np.finfo(np.float32).max, # y
            np.finfo(np.float32).max, # z
            np.finfo(np.float32).max, # dx
            np.finfo(np.float32).max, # dy
            np.finfo(np.float32).max, # dz
            np.finfo(np.float32).max, # d2x
            np.finfo(np.float32).max, # d2y
            np.finfo(np.float32).max, # d2z
            2*np.pi, # phi
            2*np.pi, # theta
            2*np.pi, # psi
            np.finfo(np.float32).max, # dphi
            np.finfo(np.float32).max, # dtheta
            np.finfo(np.float32).max, # dpsi
            np.finfo(np.float32).max, # d2phi
            np.finfo(np.float32).max, # d2theta
            np.finfo(np.float32).max, # d2psi
            np.finfo(np.float32).max # sim_time
        ], dtype = np.float32)

        observation_low = -observation_high
        #observation_low[-1] = 0. # min sim_time is 0
        self.observation_space = gym.spaces.Box(observation_low, observation_high, dtype=np.float32) 


    def _make_action_space(self) -> np.array:
        '''
        Makes an array of potential discretized settings, between -1 and 1, for each motor based
        on the num_steps_per_motor.  For example, if num_steps_per_motor is 3, then each motor can
        take on the values -1, 0, or 1. This method returns a np.array with all possible combinations
        of settings for each motor.
        '''
        step_size = 2/(CONTROL_STEPS - 1)
        each_motor_setting = np.arange(-1, 1 + step_size, step_size).tolist()

        motor_setting_list = []
        for motor in range(self.quadcopter.num_motors):
            motor_setting_list.append(each_motor_setting)

        action_seq = np.array(np.meshgrid(*motor_setting_list)).T.reshape(-1,self.quadcopter.num_motors)
        return action_seq

    def _convert_discrete_action(self, actions):
        ''' 
        '''
        step_size = 2/(CONTROL_STEPS - 1) 
        each_motor_setting = np.arange(-1, 1 + step_size, step_size).tolist()
        action_set = []
        for action in actions:
            action_set.append(each_motor_setting[action])
        return np.asarray(action_set)


    def _get_reward(self, actions, done) -> float:
        '''
        Calculates rewards based on how close the agent is to achieving the goal.
        '''
        action_delta = max(actions) - min(actions)
        r1 = 1 - abs(np.tanh(np.linalg.norm(self.quadcopter.pos)))
        r2 = 1 - abs(self.quadcopter.angle[2])
        r3 = -0.1 * action_delta
        r4 = -30 * done
        reward = r1 + r2 + r3 + r4
        return reward


    def _check_terimal_state(self) -> bool:
        ''' Checks if agent is in one of three terminal states: exceeding the max displacement from
        the origin, exceeding the max allowable angle in pitch or role, and exceeding the max allowable
        angle in yaw. If any one of the terminal condition is met, it returns true.
        '''
        # Check if in terminal state
        exceeded_max_displacement = np.linalg.norm(self.quadcopter.pos) > self.DisplacementThreshold 
        exceeded_max_angle = max(abs(self.quadcopter.angle[0:2])) > self.AngleThreshold 
        exceeded_max_yaw = abs(self.quadcopter.angle[2]) > self.YawThreshold

        isDone = exceeded_max_displacement or exceeded_max_angle or exceeded_max_yaw
        return bool(isDone)


    def step(self, actions) -> list([np.array, float, bool, dict]):
        '''Use the agent actions to update the state
        Input: actions -> np.array of the four motor actions 
            Each action for each motor is a float between -1 and 1 
            where 1 is max thrust and -1 is min thrust
        
        outpus: list [next_state, reward, done, info] 
            next_state = list - updated state based on the agents actions
            reward = float - reward from environment based on action and given state
            done = boolean - True if the agent is in a terminal state and False otherwise
            info = dict - additional information about the step (empty in this case)
        '''

        #action_set = self.action_seq[actions] # for discrete
        #action_set = self._convert_discrete_action(actions) # for multidiscrete
        action_set = actions # for box space

        # Convert actions to motor speeds
        motor_speeds = self.get_motor_speeds(action_set)
        # Get next state based on speeds
        self.quadcopter.step(motor_speeds)

        # step counter for episode length
        self.steps += 1 # step counter

        # Update states based on time step
        # observations = [x, y, z, dx, dy, dz, phi, theta, psi, dphi, dtheta, dpsi, sim_time]
        next_pos = self.quadcopter.pos # [x, y, z]
        next_vel = self.quadcopter.vel # [dx, dy, dz]
        next_lin_acc = self.quadcopter.lin_acc # [d2x, d2y, d2z]
        next_angle = self.quadcopter.angle # [phi, theta, psi]
        next_ang_vel = self.quadcopter.ang_vel # [dphi, dtheta, dpsi]
        next_ang_acc = self.quadcopter.ang_acc # [d2phi, d2theta, d2psi]
        next_sim_time = self.quadcopter.time # sim time
        next_state = np.array([*next_pos, *next_vel, *next_lin_acc, *next_angle, *next_ang_vel, *next_ang_acc, next_sim_time], dtype = np.float32)

        # Check if in terminal state
        done = self._check_terimal_state()

        # Ger reward
        reward = self._get_reward(action_set, done)

        # Other info to pass to next state
        info = {}

        # Check if at max episode length
        if self.steps >= self.max_steps:
            done = True
       
        return next_state, reward, done, info

    def reset(self) -> np.array:
        ''' Reset to a random initial deviation from the origin (0, 0, 0) and with 
        a random initial angular velocity in pitch, roll, and yaw
        Returns the initial state
        '''
        # Initial position
        spherical_1 = random.random()*2*math.pi
        spherical_2 = random.random()*2*math.pi
        x1 = INITIAL_RADIUS * np.cos(spherical_1) * np.sin(spherical_2)
        y1 = INITIAL_RADIUS * np.sin(spherical_1) * np.sin(spherical_2)
        z1 = INITIAL_RADIUS * np.cos(spherical_2)
        pos_0 = [x1, y1, z1]; # Initial pos relative to target - m

        vel_0 = [0., 0., 0.] # initial velocity [dx; dy; dz] in inertial frame - m/s
        acc_0 = [0., 0., 0.] # initial acceleration [d2x; d2y; d2z] in inertial frame - m/s^2
        angle_0 = [0., 0., 0.] # initial [pitch;roll;yaw] relative to inertial frame -rad
        ang_acc_0 = [0., 0., 0.] #initial angular velocity [phi_dot, theta_dot, psi_dot] - rad/s

        # Add initial random roll, pitch, and yaw rates
        random_set = np.array([random.random(), random.random(), random.random()])
        ang_vel_0 = np.deg2rad(2* INITIAL_DEVIATION * random_set - INITIAL_DEVIATION) #initial angular acceleration [phi_doubledot, theta_doubledot, psi_doubledot] -rad/s^2
        
        sim_time = 0.

        self.steps = 0 # step counter

        # Set internal quadcopter state to initial state
        initial_state = [*pos_0, *vel_0, *acc_0, *angle_0, *ang_vel_0, *ang_acc_0, sim_time]
        self.quadcopter.reset(initial_state)
        # Return initial environment state variables
        state = np.array(initial_state, dtype = np.float32)
        return state


if __name__ == '__main__':
    game = Game()
    env = Tank_Env(game)

    state = env.reset()
    count = 0
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        count +=1
        if count > 15:
            done = True
        pass
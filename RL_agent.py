# standard libraries
import os, shutil, time, math

# Conda imports
import gym
import numpy as np
import matplotlib.pyplot as plt 
import torch as th
import torch.nn as nn

# Pip imports
from stable_baselines3 import A2C, PPO, DQN, SAC
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, StopTrainingOnMaxEpisodes
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

# Other scripts in repo
from Config.game_settings import *
from Config.RL_settings import *
from tank_gym import Tanks_Env


class Agent_Dojo():

    def __init__(self, env_class, env_deterministic, log_dir, model_dir, multi_process):
        self.env_class = env_class
        self.deterministic = env_deterministic
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.multi_process = multi_process

        # Create log and model folders if not exist or clear log if exists
        self.make_and_clear_folders()

        # Check environment
        check_env(self.env_class(render=False))

        # create environment within monitor and/or multi process wrapper
        self.make_env()

    def make_subprocess_env(log_dir_thisone, seed=42):
        """
        Make environments within a SubprocVecEnv call

        :param log_dir_thisone: (str) the save location of the logs
        :param seed: (int) seed value for the env
        """
        def _init():
            sub_env = self.env_class(render=False, seed=seed)
            log_dir_thisone = os.path.join(self.log_dir, str(seed))
            os.makedirs(log_dir_thisone, exist_ok=True)
            env = Monitor(sub_env, log_dir_thisone)
            return env
        stable_baselines3.common.utils.set_random_seed(seed)
        return _init

    def make_env(self):
        if self.multi_process:
            num_CPUs_to_use = math.floor(0.9 * os.cpu_count())
            if num_CPUs_to_use == os.cpu_count():
                num_CPUs_to_use = os.cpu_count() - 1
            if num_CPUs_to_use < 1:
                num_CPUs_to_use == 1
            self.n_envs = num_CPUs_to_use
            self.env = SubprocVecEnv([self.make_subprocess_env(self.log_dir, i) for i in range(n_envs)])
        else:
            self.n_envs = 1
            self.env = DummyVecEnv([lambda: Monitor(self.env_class(render=False), self.log_dir)])

    def make_and_clear_folders(self):
        # Create model and log dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # Clear contents of tmp log folder
        for filename in os.listdir(self.log_dir):
            file_path = os.path.join(self.log_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def observe_agent(self, agent):
        env = self.env_class(render=True)
        obs = env.reset()
        done = False
        while not done:
            action, _states = agent.model.predict(obs, deterministic=self.deterministic)
            obs, reward, done, info = env.step(action)
            env.render()                               

    def plot_results(self, log_folder, title='Episode Reward') -> None:
        """
        plot the results

        :param log_folder: (str) the save location of the results to plot
        :param title: (str) the title of the task to plot
        """
        if not hasatter(self, 'movingAvgWindow'):
            self.movingAvgWindow = 50

        x, y = results_plotter.ts2xy(results_plotter.load_results(log_folder), 'episodes')
        
        # Do moving average
        weights = np.repeat(1.0, movingAvgWindow) / movingAvgWindow
        y = np.convolve(y, weights, 'valid')
        
        # Truncate x
        x = x[len(x) - len(y):]

        fig = plt.figure(title)
        plt.plot(x, y)
        plt.xlabel('Number of Episodes')
        plt.ylabel('Rewards')
        plt.title(title + " Smoothed")
        #plt.show()
        plt.savefig(os.path.join('Images', 'Reward_History.png'), bbox_inches='tight')
        plt.close(fig)

    def post_training_results(self, movingAvgWindow=50):
        self.movingAvgWindow = movingAvgWindow

        if self.multi_process:
            log_dir_0 = os.path.join(self.log_dir, '0')
            self.plot_results(log_dir_0)
        else:
            self.plot_results(self.log_dir)


class CustomCNN(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim) 
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            # nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.Conv2d(n_input_channels, 32, kernel_size=(3,5), stride=(1,2), padding=0),
            nn.ReLU(),
            #nn.Conv2d(32, 64, kernel_size=(3), stride=1, padding=0),
            #nn.ReLU(),
            #nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0),
            #nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class Base_Agent():

    def __init__(self, env, max_episodes=10, verbose=True):
        self.env = env
        self.max_episodes = max_episodes
        self.verbose = verbose
        self.datetime_hash = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

    def create_callbacks(self):
        # Stops training when the model reaches the maximum number of episodes
        callback_max_episodes = StopTrainingOnMaxEpisodes(
                                        max_episodes=self.max_episodes, 
                                        verbose=self.verbose)
        # Create the callback list
        callback_list = CallbackList([callback_max_episodes])
        return callback_list

    def linear_schedule(self, initial_value: float):
        """
        Linear learning rate schedule.

        :param initial_value: Initial learning rate.
        :return: schedule that computes
        current learning rate depending on remaining progress
        """
        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.

            :param progress_remaining:
            :return: current learning rate
            """
            return progress_remaining * initial_value
        return func
    
    def learn(self, total_timesteps):
        callback_list = self.create_callbacks()
        #self.model.learn(total_timesteps=int(1e8), callback=callback_list)
        self.model.learn(total_timesteps=total_timesteps)


class PPO_CNN_Agent(Base_Agent):

    def __init__(self, env, max_episodes, learning_rate=0.001, use_linear_LR_decrease=False, verbose=False):
        super().__init__(env, max_episodes=max_episodes, verbose=verbose)
        self.policy_kwargs = { "features_extractor_class" : CustomCNN }
        # self.policy_kwargs = {
        #             'features_extractor_class' : CustomCNN,
        #             'activation_fn' : th.nn.ReLU,
        #             'net_arch' : [{'pi' : [264, 128], 'vf' : [128, 32]}]
        #             }
        self.clip_range = 0.2
        self.ent_coef = 0.0
        self.n_epochs = 10
        self.initial_learning_rate = learning_rate
        self.policy = 'CnnPolicy' #ActorCriticCnnPolicy
        self.model_type = 'PPO'
        self.feature_extractor = 'CNN'
        self.tensorboard_str = self.model_type + '_' + self.feature_extractor + '_tensorboard'
        
        if use_linear_LR_decrease:
            self.learning_rate = self.linear_schedule(self.initial_learning_rate)
        else:
            self.learning_rate = self.initial_learning_rate

        self.model = PPO(
                    self.policy, 
                    self.env, 
                    #policy_kwargs = self.policy_kwargs, 
                    learning_rate=self.learning_rate,
                    n_epochs=self.n_epochs,
                    clip_range = self.clip_range, 
                    ent_coef = self.ent_coef,
                    verbose = self.verbose)
                    #tensorboard_log = os.path.join(os.path.curdir,self.tensorboard_str,self.datetime_hash)
                    #)


if __name__ == '__main__':
    start_time = time.time()
    env_class = Tanks_Env

    dojo = Agent_Dojo(
                    env_class=env_class,
                    env_deterministic=False, 
                    log_dir=LOG_DIR, 
                    model_dir=MODEL_DIR,
                    multi_process=MULTI_PROCESS)

    #env = env_class(render=True)

    agent = PPO_CNN_Agent(
                    dojo.env, 
                    max_episodes=MAX_EPISODES, 
                    learning_rate=INITIAL_LEARN_RATE, 
                    use_linear_LR_decrease=USE_LR_DECREASE,
                    verbose=VERBOSE)

    #agent = PPO('CnnPolicy', dojo.env, verbose=VERBOSE)

    print('Starting the training!!')
    agent.learn(total_timesteps=TOTAL_TIMESTEPS)

    dojo.post_training_results()
    dojo.observe_agent(agent)

    end_time = time.time()
    elapsed_time_sec = end_time - start_time
    elapsed_time_min = elapsed_time_sec / 60
    print('Program Complete')
    print('Execution time:', elapsed_time_min, 'minutes')

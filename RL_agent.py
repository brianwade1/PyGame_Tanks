# standard libraries
import os, shutil, time, math

# Conda imports
import gym
import numpy as np
import matplotlib.pyplot as plt 
import torch as th
import torch.nn as nn

# Pip imports
import stable_baselines3
from stable_baselines3 import A2C, PPO, DQN, SAC
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, StopTrainingOnMaxEpisodes, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
#Note: ProgressBarCallback requires tdqm and rich - only works when stable-baselines3 is installed from github (not pip or conda)

# Other scripts in repo
from Config.game_settings import *
from Config.RL_settings import *
from tank_gym import Tanks_Env


class Agent_Dojo():

    def __init__(self, env_class, RL_dir, log_dir, model_dir, multi_process, eval_render):
        self.env_class = env_class
        self.RL_dir = RL_dir
        self.log_dir = os.path.join(RL_dir, log_dir)
        self.model_dir = os.path.join(RL_dir, model_dir)
        self.multi_process = multi_process
        self.eval_render = eval_render

        # Create log and model folders if not exist or clear log if exists
        self.make_and_clear_folders()

        # Check environment
        check_env(self.env_class(render=False))

        # create environment within monitor and/or multi process wrapper
        self.make_env()

    def make_subprocess_env(self, log_dir_thisone, seed=42):
        """
        Make environments within a SubprocVecEnv call

        :param log_dir_thisone: (str) the save location of the logs
        :param seed: (int) seed value for the env
        """
        env_class = self.env_class
        def _init():
            sub_env = env_class(render=False, seed=seed)
            log_dir_thisone = os.path.join(self.log_dir, str(seed))
            os.makedirs(log_dir_thisone, exist_ok=True)
            env = Monitor(sub_env, log_dir_thisone)
            return env
        stable_baselines3.common.utils.set_random_seed(seed)
        return _init

    def make_env(self):
        if self.multi_process:
            if N_CPU_TO_USE is None or N_CPU_TO_USE == 0:
                num_CPUs_to_use = math.floor(0.9 * os.cpu_count())
            else:
                num_CPUs_to_use = N_CPU_TO_USE

            if num_CPUs_to_use == os.cpu_count():
                num_CPUs_to_use = os.cpu_count() - 1
            if num_CPUs_to_use < 1:
                num_CPUs_to_use == 1
            self.n_envs = num_CPUs_to_use
            #self.env = SubprocVecEnv([self.make_subprocess_env(self.train_log_dir, i) for i in range(self.n_envs)])
            self.env = make_vec_env(self.env_class, 
                                        n_envs = self.n_envs,
                                        monitor_dir = self.train_log_dir, 
                                        env_kwargs={'render': False})
                                        #vec_env_cls=SubprocVecEnv)
            
            
            self.eval_env = make_vec_env(self.env_class, 
                                        monitor_dir = self.eval_log_dir, 
                                        env_kwargs={'render': self.eval_render, 'seed': 9999999})
                                        #vec_env_cls=SubprocVecEnv)
            #self.eval_env = SubprocVecEnv([Monitor(self.env_class(render=self.eval_render), self.eval_log_dir)])
            #self.eval_env = SubprocVecEnv([self.make_subprocess_env(self.eval_log_dir, 99999 + i) for i in range(2)])
            
        else:
            self.n_envs = 1
            self.env = DummyVecEnv([lambda: Monitor(self.env_class(render=False), self.train_log_dir)])
            self.eval_env = DummyVecEnv([lambda: Monitor(self.env_class(render=self.eval_render), self.eval_log_dir)])

    def make_and_clear_folders(self):
        # Create model and log dir
        self.train_log_dir = os.path.join(self.log_dir, 'train')
        self.eval_log_dir = os.path.join(self.log_dir, 'eval')
        #os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.train_log_dir, exist_ok=True)
        os.makedirs(self.eval_log_dir, exist_ok=True)
        if not os.path.exists(self.model_dir):
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
            action, _states = agent.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()                               

    def plot_results(self, log_folder, dataset) -> None:
        """
        plot the results
        """
        if not hasattr(self, 'movingAvgWindow'):
            self.movingAvgWindow = 50

        x, y = results_plotter.ts2xy(results_plotter.load_results(log_folder), 'episodes')
        
        # if number of evals is less than twice the moving average window then dont do moving average
        if len(y) > 2 * self.movingAvgWindow:
            weights = np.repeat(1.0, self.movingAvgWindow) / self.movingAvgWindow
            y = np.convolve(y, weights, 'valid')
            smoothed = True
        else:
            smoothed = False
        
        # Truncate x
        x = x[len(x) - len(y):]

        title = dataset + ' ' + 'Episode Reward'
        fig_filename = dataset + '_Reward_History.png'
        fig = plt.figure(title)
        plt.plot(x, y)
        plt.xlabel('Number of Episodes')
        plt.ylabel('Rewards')
        if smoothed:
            plt.title(title + " Smoothed")
        else:
            plt.title(title)
        #plt.show()
        plt.savefig(os.path.join('Images', fig_filename), bbox_inches='tight')
        plt.close(fig)

    def post_training_results(self, movingAvgWindow=50):
        self.movingAvgWindow = movingAvgWindow

        self.plot_results(self.train_log_dir, dataset='train')
        self.plot_results(self.eval_log_dir, dataset='eval')

        # if self.multi_process:
        #     train_log_dir_0 = os.path.join(self.train_log_dir, '0')
        #     eval_log_dir_0 = os.path.join(self.eval_log_dir, '0')
        #     self.plot_results(train_log_dir_0, dataset='train')
        #     self.plot_results(eval_log_dir_0, dataset='eval')
        # else:
        #     self.plot_results(self.train_log_dir, dataset='train')
        #     self.plot_results(self.eval_log_dir, dataset='eval')


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

    def __init__(self, dojo, model_name, max_episodes=None, progress_bar=False, max_no_improvement_evals=None, min_evals=5, n_eval_episodes=5, eval_freq=None, eval_render=False, eval_verbose=True):
        self.dojo = dojo
        self.env = dojo.env
        self.model_name = model_name
        self.eval_env = dojo.eval_env
        self.max_episodes = max_episodes
        self.progress_bar = progress_bar
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_render = eval_render
        self.eval_verbose = eval_verbose
        self.datetime_hash = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

    def create_callbacks(self):
        callback_list = []
        # Stops training when the model reaches the maximum number of episodes
        if self.max_episodes is not None:
            callback_max_episodes = StopTrainingOnMaxEpisodes(
                                        max_episodes=self.max_episodes, 
                                        verbose=True)
            callback_list.append(callback_max_episodes)
        # # Create progress bar
        # if self.progress_bar:
        #     progress_bar = ProgressBarCallback()
        #     callback_list.append(progress_bar)

        # stop if no improvement in reward
        if self.max_no_improvement_evals is not None:
            no_improve = StopTrainingOnNoModelImprovement(
                                        max_no_improvement_evals=self.max_no_improvement_evals, 
                                        min_evals=self.min_evals, 
                                        verbose=True)

        # eval env
        if self.eval_freq is not None:
            eval_log_dir = os.path.join(self.dojo.eval_log_dir, self.model_name)
            best_model_dir = os.path.join(self.dojo.model_dir, self.model_name)
            eval_call = EvalCallback(
                                eval_env=self.eval_env,
                                callback_after_eval = no_improve,
                                n_eval_episodes=self.n_eval_episodes,
                                eval_freq=self.eval_freq,
                                log_path=eval_log_dir,
                                best_model_save_path=best_model_dir,
                                deterministic=True,
                                render=self.eval_render,
                                verbose=self.eval_verbose)
            callback_list.append(eval_call)
        
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
        if len(callback_list) > 0:
            self.model.learn(total_timesteps=total_timesteps, callback=callback_list)
        else:
            self.model.learn(total_timesteps=total_timesteps)


class PPO_CNN_Agent(Base_Agent):

    def __init__(self, dojo, model_name, max_episodes, max_no_improvement_evals, progress_bar=True, min_evals=10, n_eval_episodes=5, eval_freq=10000, eval_render=False, learning_rate=0.001, use_linear_LR_decrease=False, train_verbose=False, eval_verbose=True):
        super().__init__(dojo, model_name, max_episodes=max_episodes, progress_bar=progress_bar, max_no_improvement_evals=max_no_improvement_evals, min_evals=min_evals, eval_freq=eval_freq, eval_render=eval_render, eval_verbose=eval_verbose)
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
                    verbose = self.train_verbose)
                    #tensorboard_log = os.path.join(self.dojo.RL_dir,'tensorboard/',self.model_name,self.datetime_hash)
                    #)


class PPO_MLP_Agent(Base_Agent):

    def __init__(self, dojo, model_name, max_episodes, max_no_improvement_evals, progress_bar=True, min_evals=10, n_eval_episodes=5, eval_freq=10000, eval_render=False, learning_rate=0.001, use_linear_LR_decrease=False, train_verbose=False, eval_verbose=True):
        super().__init__(dojo, model_name, max_episodes=max_episodes, progress_bar=progress_bar, max_no_improvement_evals=max_no_improvement_evals, min_evals=min_evals, eval_freq=eval_freq, eval_render=eval_render, eval_verbose=eval_verbose)
        self.policy_kwargs = {
                    'activation_fn' : th.nn.ReLU,
                    'net_arch' : [{'pi' : [1028, 512, 128], 'vf' : [512, 128, 32]}]
                    }
        self.train_verbose = train_verbose
        self.clip_range = 0.2
        self.ent_coef = 0.0
        self.n_epochs = 10
        self.initial_learning_rate = learning_rate
        self.policy = 'MlpPolicy'
        self.model_type = 'PPO'
        self.feature_extractor = 'MLP'
        
        if use_linear_LR_decrease:
            self.learning_rate = self.linear_schedule(self.initial_learning_rate)
        else:
            self.learning_rate = self.initial_learning_rate

        self.model = PPO(
                    self.policy, 
                    self.env, 
                    policy_kwargs = self.policy_kwargs, 
                    learning_rate=self.learning_rate,
                    n_epochs=self.n_epochs,
                    clip_range = self.clip_range, 
                    ent_coef = self.ent_coef,
                    verbose = self.train_verbose,
                    tensorboard_log = os.path.join(self.dojo.RL_dir,'tensorboard/',self.model_name,self.datetime_hash)
                    )


if __name__ == '__main__':
    start_time = time.time()
    env_class = Tanks_Env

    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

    dojo = Agent_Dojo(
                    env_class=env_class,
                    RL_dir=RL_DIR,
                    log_dir=LOG_DIR, 
                    model_dir=MODEL_DIR,
                    multi_process=MULTI_PROCESS,
                    eval_render=EVAL_RENDER)

    if AGENT_OBS_TYPE == 'CNN':
        agent = PPO_CNN_Agent(
                        dojo, 
                        model_name='PPO_CNN',
                        max_episodes=MAX_EPISODES, 
                        min_evals=MIN_EVALS,
                        max_no_improvement_evals=MAX_NO_IMPROVE,
                        progress_bar=PROGRESS_BAR,
                        eval_render=EVAL_RENDER,
                        eval_freq=EVAL_FREQ,
                        n_eval_episodes=N_EVAL_EPISODES,
                        learning_rate=INITIAL_LEARN_RATE, 
                        use_linear_LR_decrease=USE_LR_DECREASE,
                        verbose=VERBOSE)

    elif AGENT_OBS_TYPE == 'MLP':
        agent = PPO_MLP_Agent(
                        dojo,
                        model_name='PPO_MLP',
                        max_episodes=MAX_EPISODES, 
                        max_no_improvement_evals=MAX_NO_IMPROVE,
                        min_evals=MIN_EVALS,
                        progress_bar=PROGRESS_BAR,
                        eval_render=EVAL_RENDER,
                        eval_freq=EVAL_FREQ,
                        n_eval_episodes=N_EVAL_EPISODES,
                        learning_rate=INITIAL_LEARN_RATE, 
                        use_linear_LR_decrease=USE_LR_DECREASE,
                        train_verbose=TRAIN_VERBOSE,
                        eval_verbose=EVAL_VERBOSE)

    else:
        raise Exception("Agent Observation Type Unknown")


    agent_name = 'PPO_' + AGENT_OBS_TYPE
    agent_path = os.path.join('RL', 'Model', agent_name)

    if CONTINUE_TRAINING:
        # Check for agent saved at the end of the last training run
        if os.path.exists(os.path.join(agent_path, f"{agent_name}.zip")):
            agent.model.set_parameters(os.path.join(agent_path, f"{agent_name}.zip"))
        # Check for agent saved during eval callbacks
        elif os.path.exists(os.path.join(agent_path, 'best_model.zip')):
            agent.model.set_parameters(os.path.join(agent_path, 'best_model.zip'))
        # raise exception because agent could not be found
        else:
            raise Exception("Agent does not exist. Cannot continue training. Check path or train a new agent from scratch.")


    print('Starting the training!!')
    agent.learn(total_timesteps=TOTAL_TIMESTEPS)

    agent.model.save(os.path.join(agent_path, agent_name))

    dojo.post_training_results()
    dojo.observe_agent(agent)

    end_time = time.time()
    elapsed_time_sec = end_time - start_time
    min_leftover, sec = divmod(elapsed_time_sec, 60)
    hour, min = divmod(min_leftover, 60) 
    print('Program Complete')
    print(f'Execution time: {hour} : {min} : {sec}')

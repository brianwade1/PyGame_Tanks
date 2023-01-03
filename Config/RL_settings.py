from Config.game_settings import *

# Define agent rewards
R_EACH_STEP = 0.05
R_DYING = -10
R_LIVING_TO_END = 10

# Directories
RL_DIR = 'RL'
LOG_DIR = 'Log/'
MODEL_DIR = 'Model/'

# Results plot settings
MOVING_AVG_WINDOW = 50

# RL Training Settings
MULTI_PROCESS = False
MAX_EPISODES = None
MAX_NO_IMPROVE = 50
MIN_EVALS = 500
INITIAL_LEARN_RATE = 0.001
TRAIN_VERBOSE = False
EVAL_VERBOSE = True
USE_LR_DECREASE = False
TOTAL_TIMESTEPS = 1_500_000
PROGRESS_BAR = True
EVAL_RENDER = False
N_EVAL_EPISODES = 5
EVAL_FREQ = 10_000
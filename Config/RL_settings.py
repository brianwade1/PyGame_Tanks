from Config.game_settings import *

# Define agent rewards
R_EACH_STEP = 0.05
R_DYING = -10
R_LIVING_TO_END = 10

# Directories
LOG_DIR = 'RL_log/'
MODEL_DIR = 'RL_Model/'

# Results plot settings
MOVING_AVG_WINDOW = 50

# RL Training Settings
MULTI_PROCESS = False
MAX_EPISODES = 10
INITIAL_LEARN_RATE = 0.001
VERBOSE = True
USE_LR_DECREASE = False
TOTAL_TIMESTEPS = 500_000
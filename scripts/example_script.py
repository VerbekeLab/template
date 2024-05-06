"""
    __   ___   ___    ____  ____    ____      ____ _____     _____  __ __  ____  
   /  ] /   \ |   \  |    ||    \  /    |    |    / ___/    |     ||  |  ||    \ 
  /  / |     ||    \  |  | |  _  ||   __|     |  (   \_     |   __||  |  ||  _  |
 /  /  |  O  ||  D  | |  | |  |  ||  |  |     |  |\__  |    |  |_  |  |  ||  |  |
/   \_ |     ||     | |  | |  |  ||  |_ |     |  |/  \ |    |   _] |  :  ||  |  |
\     ||     ||     | |  | |  |  ||     |     |  |\    |    |  |   |     ||  |  |
 \____| \___/ |_____||____||__|__||___,_|    |____|\___|    |__|    \__,_||__|__|
                                                                                 
"""

########################################################################
# Let's look at why this setup is useful.                              #
# Again: Scripts and notebooks are for execution, not for development. #
########################################################################

#########################
# 1. Step: Load modules #
#########################

# LOAD MODULES
# Standard library
import os
import sys
import itertools
import warnings

# Third party
from tqdm import tqdm

# NOTE: Your script is not in the root directory. We must hence change the system path
DIR = "/Users/cbr/Code/_boilerplate-code/bp_project-template"
os.chdir(DIR)
sys.path.append(DIR)

# Proprietary
from src.data.ihdp_s_1 import load_data
from src.methods.neural import MLP
from src.utils.metrics import mean_integrated_prediction_error
from src.utils.setup import (
    load_config,
    check_create_csv,
    get_rows,
    add_row,
    add_dict,
)
from src.utils.training import train_val_tuner

#####################
# 2. Step: Settings #
#####################

# SETTINGS
# Directory
# NOTE: I am working with a tracker that keeps track of the experiments that have already been completed. 
# This is useful if you want to run the script in parallel on multiple machines. 
# If you do not want to use this, you can simply remove the tracker and the corresponding code.
RES_FILE = "results.csv"
TRACKER = "tracker.csv"

# SETUP
# Load config
CONFIG = load_config("config/data/ihdp_s_1.yaml")["parameters"]
HYPERPARAMS = load_config("config/methods/mlp.yaml")["parameters"]

# Number of parameter combinations to consider
RANDOM_SEARCH_N = 3

# Save para combinations
COMBINATIONS = list(itertools.product(*CONFIG.values()))

# This will mute divice summaries in PyTorch
import logging
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logger = logging.getLogger("lightning.pytorch.core")
logger.addHandler(logging.FileHandler("core.log"))
warnings.filterwarnings("ignore")

######################
# 3. Step: Execution #
######################

# Initialize the tracker csv file
check_create_csv(TRACKER, CONFIG.keys())

# Iterate over combinations
for combination in tqdm(COMBINATIONS, desc="Iterate over combinations"):
    # 0. Check whether combination is already completed
    completed = get_rows(TRACKER)
    if (combination in completed):
        continue
    else:
        # Add combination to the tracker
        add_row(TRACKER, combination)
        # Save settings as a dictionary
        data_settings = dict(zip(CONFIG.keys(), combination))
    
    # 1. Initialize the results dictionary
    results = {}
    results.update(data_settings)
    
    # 2. Load the data
    data = load_data(**data_settings)
    
    # 3. Train model
    name = "MLP"
    # We need to add the input size for the MLP
    HYPERPARAMS.update({"input_size": [data.x_train.shape[1]]})
    # Train and tune
    model, best_paras = train_val_tuner(
        data=data,
        model=MLP,
        parameters=HYPERPARAMS,
        name=name,
        num_combinations=RANDOM_SEARCH_N,
    )
    
    # 4. Evaluate performance
    mise = mean_integrated_prediction_error(
                x = data.x_test,
                response = data.ground_truth,
                model = model
    )
    
    # 5. Add results to the dictionary
    results.update(
        {
            "MISE " + name: mise,
        }
    )
    
    # 6. Add dictionary to the csv file
    add_dict(RES_FILE, results)
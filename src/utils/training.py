# LOAD MODULES
# Standard library
from typing import Callable, Tuple, Dict, Optional, Union
import random
import itertools

# Third party
import numpy as np
from tqdm import tqdm

# Proprietary
from src.data.utils import ContinuousData

def train_val_tuner(
    data: ContinuousData,
    model: Callable,
    parameters: dict,
    name: str = "method",
    num_combinations: Optional[int] = None,
):
    """
    Performs training and validation tuning on a given model with specified parameters.

    Parameters:
    data (ContinuousData or BinaryData): The dataset to be used for training and validation.
    model (Callable): The machine learning model to be tuned.
    parameters (dict): The parameters for the model.
    name (str, optional): The name of the method. Defaults to "method".
    num_combinations (int, optional): The number of parameter combinations to consider. If None, all combinations are considered. Defaults to None.

    Returns:
    final_model (Callable): The tuned model.
    best_parameters (dict): The best found settings for the model parameters.
    """
    # Seed
    random.seed(42)
    
    # Ini error
    current_best = np.inf
        
    # Save combinations and shuffle
    combinations = list(itertools.product(*parameters.values()))
    random.shuffle(combinations)
    
    # Sample combinations for random search
    if (num_combinations is not None) and (num_combinations < len(combinations)):
        combinations = combinations[:num_combinations]

    # Iterate over all combinations
    for combination in tqdm(combinations, leave=False, desc="Tune " + name):
        # Save settings of current iteration
        settings = dict(zip(parameters.keys(), combination))

        # Set up model
        estimator = model(**settings)

        # Fit model
        estimator.fit(data.x_train, data.y_train, data.d_train)

        # Score
        score = estimator.score(data.x_val, data.y_val, data.d_val)

        # Evaluate if better than current best
        if current_best > score:
            # Set best settings
            best_parameters = settings
            current_best = score

    # Train final model
    final_model = model(**best_parameters)
    final_model.fit(data.x_train, data.y_train, data.d_train)

    return final_model, best_parameters

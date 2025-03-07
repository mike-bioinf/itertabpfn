import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class OptSetup:
    '''
    Dataclass for the finetune setup.
        min_lr (float): The inferior limit for the learning_rate searchable range.
        max_lr (float): The superior limit for the learning_rate searchable range.
        min_bs (int): The inferior limit for the batch_size searchable range.
        max_bs (int): The superior limit for the batch_size searchable range.
    '''
    min_lr: float
    max_lr: float
    min_bs: int
    max_bs: int


@dataclass
class FineTuneSetup:
    '''
    Dataclass for the optimization setup.
        adaptive_rate (float): The rate of increase in patience.
             Set to 0 to disable, or negative to shrink patience during training
        adaptive_offset (int): The initial patience at round 0.
        min_patience (int): The minimum value of patience.
        max_patience (int): The maximum value of patience.
        time_limit (int): Maximum time in seconds after which the fine tuning process is stopped.
        max_steps (int): Maximum number of learning step.
    '''
    adaptive_rate: float = 0.3 
    adaptive_offset: int = 10
    min_patience: int = 10
    max_patience: int = 100
    time_limit: int = 60
    max_steps: int = 10000



def get_minmax_batch_sizes(array: np.ndarray | pd.DataFrame | pd.Series) -> tuple[int, int]:
    '''
    Retrieve the minimum and maximum batch sizes to try in a HPO process based on euristics.
    The function takes in input both 2D and 1D data structure. 
    In the first case the 0 dimension is used to determine the number of samples.
    Parameters:
        array (np.ndarray | pd.DataFrame | pd.Series): array-like structure from which the number of samples is inferred.
    Returns: A binary tuple with the minimum and maximum batch size values.
    '''
    n_samples = array.shape[0]
    
    if n_samples < 100:
        min_bs = round(n_samples/2)
        max_bs = n_samples - 2
    elif n_samples < 200:
        min_bs = round(n_samples/4)
        max_bs = round(n_samples/3)
    else:
        min_bs = round(n_samples/8)
        max_bs = round(n_samples/6)

    return min_bs, max_bs
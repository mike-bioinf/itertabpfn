import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Literal



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
        time_limit (int | Literal["infer"]): Maximum time in seconds after which the fine tuning process is stopped.
            If infer the time limit definition is delayed to the OptFineTuneTabPFN instance. 
        max_steps (int): Maximum number of learning step.
    '''
    adaptive_rate: float = 0.3 
    adaptive_offset: int = 20
    min_patience: int = 20
    max_patience: int = 100
    time_limit: int | Literal["infer"] = 300
    max_steps: int = 10000



def infer_minmax_batch_sizes(array: np.ndarray | pd.DataFrame | pd.Series) -> tuple[int, int]:
    '''
    Retrieve the minimum and maximum batch sizes to try in a HPO process based on euristics.
    The function takes in input both 1D and 2D data structure. 
    In the latter case the 0 dimension is used to determine the number of samples.
    Parameters:
        array (np.ndarray | pd.DataFrame | pd.Series): array-like structure from which the number of samples is inferred.
    Returns: A binary tuple with the minimum and maximum batch size values.
    '''
    n_samples = array.shape[0]
    if n_samples < 100:
        min_bs = round(n_samples/5)
        max_bs = round(n_samples/3)
    elif n_samples < 200:
        min_bs = round(n_samples/9)
        max_bs = round(n_samples/5)
    else:
        min_bs = round(n_samples/18)
        max_bs = round(n_samples/9)
    return min_bs, max_bs



def infer_batch_size(array: np.ndarray | pd.DataFrame | pd.Series) -> int:
    '''
    Retrieve a single batch size value based on euristics.
    The function takes in input both 1D and 2D data structure. 
    In the latter case the 0 dimension is used to determine the number of samples.
    Parameters:
        array (np.ndarray | pd.DataFrame | pd.Series): array-like structure from which the number of samples is inferred.
    Returns: The batch size.
    '''
    n_samples = array.shape[0]
    if n_samples < 100:
        return round(n_samples/3)
    elif n_samples < 200:
        return round(n_samples/6)
    else:
        return round(n_samples/12)



def infer_time_limit(array: np.ndarray | pd.DataFrame | pd.Series) -> float:
    '''
    Infer the time limit based on the number of samples and euristics.
    The idea is to give more time to bigger data and less time to smaller ones that are more likely to overfit.
    The function takes in input both 1D and 2D data structure.
    In the latter case the 0 dimension is used to determine the number of samples.
     Parameters:
        array (np.ndarray | pd.DataFrame | pd.Series): array-like structure from which the number of samples is inferred.
    Returns: The estimated time limit in seconds.
    '''
    n_samples = array.shape[0]
    if n_samples < 100:
        return 300.00
    elif n_samples < 600:
        return 800.00
    else:
        return 1800.00

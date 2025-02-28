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

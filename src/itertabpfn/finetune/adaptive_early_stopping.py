class AdaptiveEarlyStopping:
    '''
    Implements early stopping with adaptive patience.
    Patience is adaptively adjusted across training instead of being a fixed value.
    The class is supposed to only manage the patience and remaining patience computation.
    The train loop must therefore be stopped in the training logic implementation.
    
    ----------
    Parameters:
        adaptive_rate : float, default 0.3
            The rate of increase in patience. 
            Set to 0 to disable, or negative to shrink patience during training.
        adaptive_offset : int, default 10
            The initial patience at round 0.
        min_patience : int, default 10
            The minimum value of patience.
        max_patience : int, default 100
            The maximum value of patience.
    
    ---------
    Attributes:
    best_round : int
        Used to compute patience.
    patience : int
        If no improvement occurs in `patience` rounds or greater, self.early_stop will return True.
        patience is dictated by the following formula:
        patience = min(self.max_patience, (max(self.min_patience, round(self.best_round * self.adaptive_rate + self.adaptive_offset))))
        Effectively, patience = self.best_round * self.adaptive_rate + self.adaptive_offset, bound by min_patience and max_patience.
    '''
    def __init__(self, adaptive_rate: float = 0.3, adaptive_offset: int = 10, min_patience: int = 10, max_patience: int = 100):
        self.adaptive_rate = adaptive_rate
        self.adaptive_offset = adaptive_offset
        self.min_patience = min_patience
        self.max_patience = max_patience
        self.best_round = 0
        self.patience = self.adaptive_offset


    def set_best_round(self, round: int) -> None:
        '''
        Set the input round as the new best round.
        Parameters:
            round (int): New best round.
        Returns: None
        '''
        self.best_round = round

    
    def update_patience(self) -> None:
        '''
        Update the patience based on the best round.
        Returns: None
        '''
        self.patience = min(self.max_patience, max(self.min_patience, round(self.best_round * self.adaptive_rate + self.adaptive_offset)))


    def get_remaining_patience(self, current_round: int) -> int:
        '''
        Computes and returns the remaining patience based on the current vs internal best round.
        Parameters:
            current_round (int): Integer in respect to which the residual patience is computed.
        Returns: The residual patience, aka the number of rounds to still wait.
        '''
        return self.patience - (current_round - self.best_round)
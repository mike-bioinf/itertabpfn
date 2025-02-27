import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal
from copy import deepcopy
import optuna
import torch
import torch.nn.functional as F
from tabpfn.base import load_model_criterion_config
from tabpfn.model.transformer import PerFeatureTransformer
import torch.optim as optim
from torch import GradScaler
from torch.utils.data import TensorDataset, DataLoader
from itertabpfn.finetune.adaptive_early_stopping import AdaptiveEarlyStopping
from itertabpfn.finetune.model_utils import save_model
from sklearn.model_selection import train_test_split
from itertabpfn.finetune.setup import FineTuneSetup, OptSetup
# from schedulefree import AdamWScheduleFree



class OptFineTuneTabpfn:
    '''
    Implements a fine tuning strategy for tabpfn classifiers with AES and hyperparameter optimization.
    The procedure works on a single dataframe.
    The learning rate and batch size are the learnable HPs, with the first sampled exponentially and the second linearly. 
    
    The fine-tuning procedure implements three stopping logic:
        1) early stopping on training time;
        2) AES;
        3) Number of training steps. 
        
    ----------
    Parameters:
        path_base_model (str | Path | Literal["auto"], optional): 
            Path to the base tabpfn model. Defaults to "auto", in which case the model is founded automatically using tabpfn utilities.
        save_path_fine_tuned_model (str | Path): Path where to save the finetuned models.
        X_train (pd.DataFrame): training pandas dataframe.
        y_train (pd.Series): training target series.
        X_val (pd.DataFrame): validation pandas dataframe.
        y_val (pd.Series): validation target series.
        fine_tune_setup (FineTuneSetup): FineTuneSetup instance with the finetune directivies.
        opt_setup (OptSetup): OptSetup instance containing the optimization directivies. 
        softmax_temperature (float, optional): Number between 0 and 1.
            The rate of increase in patience. Set to 0 to disable, or negative to shrink patience during training.
        random_seed (int, optional): seed that control the randomness.
            Setting this seed means estabishiling reproducibile seeds for all the involved random processes. Default to 50.
        device (Literal["auto"], optional): Search automaticaly for the GPU falling otherwise on the CPU.
        
    ----------
    Attributes:

    min_lr (float): The inferior limit for the learning_rate searchable range.
    max_lr (float): The superior limit for the learning_rate searchable range.
    min_bs (int): The inferior limit for the batch_size searchable range.
    max_bs (int): The superior limit for the batch_size searchable range.
    
    adaptive_rate (float): The rate of increase in patience.
    adaptive_offset (int): The initial patience at round 0.
    min_patience (int): The minimum value of patience.
    max_patience (int): The maximum value of patience.
    time_limit (int): Maximum time in seconds after which the fine tuning process is stopped.
    max_steps (int): Maximum number of learning step.
    use_autocast: bool, True if the GPU is available and False otherwise.
        Enable mized precision training (stable only on the GPU).
    
    scaler: instance of the GradScaler class. Apply scaling (and then unscaling) to gradients only in case autocast is True.
    model: tabpfn classifier. Its loading is resolved by a tabpfn base utility.
    criterion: instance of CrossEntropyLoss class since only tabpfn classifiers are used.
    checkpoint_config: object containing metadata of the loaded tabpfn classifier.
    '''
    def __init__(
        self,
        *,
        path_base_model: str | Path | Literal["auto"] = "auto",
        save_path_fine_tuned_model: str | Path,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame, 
        y_val: pd.Series, 
        fine_tune_setup: FineTuneSetup,
        opt_setup: OptSetup,
        softmax_temperature: float = 1,
        random_seed: int = 50, 
        device: Literal["auto"] = "auto"
    ):   
        self._check_data(X_train, y_train, X_val, y_val)
        self.X_train, self.y_train, self.X_val, self.y_val = X_train, y_train, X_val, y_val
        self.n_classes = len(self.y_train.unique())
        self.path_base_model = path_base_model
        self.save_path_fine_tuned_model = Path(save_path_fine_tuned_model) if isinstance(save_path_fine_tuned_model, str) else save_path_fine_tuned_model
        self.min_lr: float = opt_setup.min_lr
        self.max_lr: float = opt_setup.max_lr
        self.min_bs: int = opt_setup.min_bs
        self.max_bs:int = opt_setup.max_bs
        self.adaptive_rate: float = fine_tune_setup.adaptive_rate
        self.adaptive_offset: float = fine_tune_setup.adaptive_offset
        self.min_patience: int = fine_tune_setup.min_patience
        self.max_patience: int = fine_tune_setup.max_patience
        self.time_limit: float = fine_tune_setup.time_limit
        self.max_steps: int = fine_tune_setup.max_steps
        self.softmax_temperature = softmax_temperature
        self.random_seed = random_seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_autocast = True if self.device == "cuda" else False
        # self.scaler: GradScaler = None
        # self.model: PerFeatureTransformer = None
        # self.criterion: torch.nn.CrossEntropyLoss = None
        # self.checkpoint_config = None



    def __call__(self, trial: optuna.Trial) -> float:
        '''
        Method called when the 'OptFineTuneTabpfn' instance is passed in the optuna optimizer.
        The models are currently not saved.
        Returns: The best validation loss.
        '''
        lr = trial.suggest_float("learning_rate", self.min_lr, self.max_lr, log=True)
        batch_size = trial.suggest_int("batch_size", self.min_bs, self.max_bs)
        best_val_loss = self._fine_tune_tabpfn_clf(lr, batch_size, save=False)
        return best_val_loss



    def fine_tune_tabpfn_clf(self, learning_rate: float, batch_size: int, filename: str) -> None:
        '''
        Fine tune tabpfn classifier on a single dataset with custom learning rate and batch size values.
        Saves the finetuned model at the location specified in "__init__".
        Parameters:
            learning_rate (float): Learning rate to use.
            batch_size (int): Batch size to use.
            filename (str): String reporting the filename.
        Returns: None.
        '''
        _ = self._fine_tune_tabpfn_clf(learning_rate, batch_size, save=True, filename=filename)
        return None



    def _fine_tune_tabpfn_clf(self, learning_rate: float, batch_size: int, save: bool, filename: str = None) -> float:
        '''
        Fine tune tabpfn classifier on a single dataset.
        Interal version to use in the "__call__" method. 
        Wants in input the HPs values and the indication to save or not the finetuned model.
        Returns: the best validation loss.
        '''
        if save: 
            filepath = os.path.join(self.save_path_fine_tuned_model, filename)
        
        scaler = GradScaler(device=self.device, growth_interval=100, enabled=self.use_autocast)
        model, criterion, checkpoint_config = self._load_model_criterion_config()
        checkpoint_config = checkpoint_config.__dict__

        aes = AdaptiveEarlyStopping(self.adaptive_rate, self.adaptive_offset, self.min_patience, self.max_patience)
        train_x, train_y, val_x, val_y = self._prepare_data_to_forward(self.X_train, self.y_train, self.X_val, self.y_val)
        
        if self.device == "cuda": 
            model = model.to(device="cuda")
            train_x = train_x.to(device="cuda")
            train_y = train_y.to(device="cuda")
            val_x = val_x.to(device="cuda")
            val_y = val_x.to(device="cuda")

        torch_rng = torch.Generator(self.device).manual_seed(self.random_seed)
        data_loader = self._prepare_data_loader(batch_size, torch_rng)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        # starting time counter
        start_time = time.time()
        
        # setting base-model val loss
        model.eval()
        best_val_loss = self._forward_validation_step(model, criterion, train_x, train_y, val_x, val_y)

        # saving the base model since its the best right now
        if save:
            save_model(model, filepath, checkpoint_config)

        # start training
        total_steps = 0
        effective_steps = 0  
        skipped_steps = 0 

        while True:
            for X_batch, y_batch in data_loader:
                X_batch_train, X_batch_test, y_batch_train, y_batch_test = train_test_split(X_batch, y_batch, test_size=0.3, random_state=self.random_seed)
                train_batch_x, train_batch_y, test_batch_x, test_batch_y= self._prepare_data_to_forward(X_batch_train, y_batch_train, X_batch_test, y_batch_test)

                if self.device == "cuda":
                    train_batch_x = train_batch_x.to(device="cuda")
                    train_batch_y = train_batch_y.to(device="cuda")
                    test_batch_x = test_batch_x.to(device="cuda")
                    test_batch_y = test_batch_y.to(device="cuda")

                total_steps += 1
                model.train()
                
                with torch.autocast(device_type=self.device, enabled=self.use_autocast):
                    loss = self._forward_step(model, criterion, train_batch_x, train_batch_y, test_batch_x, test_batch_y) 
                    
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                ### clipping (TOIMPLEMENT)
                scaler.step(optimizer)
                
                ori_scale = scaler.get_scale()
                scaler.update()
                updated_scale = scaler.get_scale()
                
                if updated_scale < ori_scale:
                    skipped_steps += 1
                    step_with_update = False
                else:
                    effective_steps += 1
                    step_with_update = True
                
                optimizer.zero_grad()
                
                if step_with_update:
                    model.eval()
                    updated_val_loss = self._forward_validation_step(model, criterion, train_x, train_y, val_x, val_y)
                    

                # AES and stopping logic
                if step_with_update and updated_val_loss < best_val_loss:
                    best_val_loss = updated_val_loss
                    aes.set_best_round(effective_steps)
                    aes.update_patience()
                    # overwriting the model with the best one
                    if save: 
                        save_model(model, filepath, checkpoint_config)
                else:
                    residual_patience = aes.get_remaining_patience(effective_steps)
                    if residual_patience <= 0:
                        return best_val_loss
                
                # other stopping logic
                elapsed_time = time.time() - start_time
                
                if elapsed_time >= self.time_limit:
                    return best_val_loss
                
                if total_steps == self.max_steps:
                    return best_val_loss





    def _forward_step(
            self,
            model: PerFeatureTransformer,
            criterion: torch.nn.CrossEntropyLoss,
            train_x: torch.Tensor,   # (n_samples, 1, n_features)
            train_y: torch.Tensor,  # (n_samples, 1, 1)
            test_x: torch.Tensor,   # (n_samples, 1, n_features)
            test_y: torch.Tensor,   # (n_samples, 1, 1)
        ) -> torch.Tensor:
        '''
        Perform the forward pass.
        Needs the context and test tensors.
        Returns: The loss as a tensor of a single scalar.
        '''
        pred_logits = model(train_x=train_x, train_y=train_y, test_x=test_x)
        # not squeezing the first dimension to accomodate case of single test sample
        pred_logits = pred_logits.squeeze(dim=1)[:, :self.n_classes]
        test_y = test_y.squeeze(dim=(1, 2)).long()
        if self.softmax_temperature is not None:
            pred_logits = pred_logits / self.softmax_temperature
        loss = criterion(pred_logits, test_y).mean()
        return loss
    


    def _forward_validation_step(
            self,
            model: PerFeatureTransformer,
            criterion: torch.nn.CrossEntropyLoss,
            train_x: torch.Tensor,
            train_y: torch.Tensor,
            test_x: torch.Tensor,
            test_y: torch.Tensor,
        ) -> float:
        '''
        Perform the forward pass for the validation scenario.
        This means that the forward step is run in no grad mode, and that the returned loss is a scalar.
        Returns: A single scalar loss.
        '''
        with torch.no_grad():
            loss = self._forward_step(model, criterion, train_x, train_y, test_x, test_y)
        return loss.mean().item()



    def _load_model_criterion_config(self) -> tuple:
        '''
        Loads the model spefied in "__init__" using the tabpfn "load_model_criterion_config" utility.
        Returns: The model, the criterion (CrossEntropy torch class instance) and the model configuration (contains metadata).
        '''
        path_model = None if self.path_base_model == "auto" else self.path_base_model
        return load_model_criterion_config(
            model_path=path_model,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=False,
            which="classifier",
            download=True,
            version="v2",
            model_seed=self.random_seed
        )


    @staticmethod
    def _check_data(X_train, y_train, X_val, y_val) -> None:
        '''Check input data types.'''
        for x_data in [X_train, X_val]:
            if not isinstance(x_data, pd.DataFrame):
                raise TypeError("X_train and X_val must be pandas DataFrame objects")
        for y_data in [y_train, y_val]:
            if not isinstance(y_data, pd.Series):
                raise TypeError("y_train and y_val must be pandas Series objects")



    def _prepare_data_loader(self, batch_size: int, generator: torch.Generator) -> DataLoader:
        '''
        Prepares the training data loader.
        Wants in input the batch size and the generator instance.
        Returns: The DataLoader instance.
        '''
        X = torch.tensor(self.X_train.to_numpy(), dtype=torch.float32)
        y = torch.tensor(self.y_train.to_numpy(), dtype=torch.float32)
        return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True, generator=generator, drop_last=True)



    @staticmethod
    def _prepare_data_to_forward(*args: pd.DataFrame | pd.Series | np.ndarray | torch.Tensor) -> list[torch.Tensor]:
        '''
        Prepare the input data (pd.DataFrame, pd.Series, np array or torch tensors) to 3D tensor.
        In detail the dataframe and 2D arrays objects are converted to 3D tensors of shape (shape[0], 1, shape[1]).
        The series and 1D array objects are instead converted to 3D tensors of shape (shape[0], 1, 1).
        The tensors are of float32 type.
        Retuns: A list of the resulting tensor in the same order of the input.
        '''
        tensors = []
        for data in args:
            if isinstance(data, (pd.DataFrame, pd.Series)):
                data = data.to_numpy()
            elif isinstance(data, torch.Tensor):
                data = data.numpy()

            if len(data.shape) == 2: 
                tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
            else:
                tensor = torch.tensor(data, dtype=torch.float32).reshape((-1, 1, 1))
            tensors.append(tensor)
        return tensors
    





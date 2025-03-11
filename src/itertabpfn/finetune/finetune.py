import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal
import optuna
import torch
import torch.optim as optim
from torch import GradScaler
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from tabpfn.base import load_model_criterion_config
from tabpfn.model.transformer import PerFeatureTransformer
from itertabpfn.finetune.adaptive_early_stopping import AdaptiveEarlyStopping
from itertabpfn.finetune.model_utils import save_model
from itertabpfn.finetune.setup import FineTuneSetup, OptSetup, infer_time_limit
from sklearn.model_selection import train_test_split
# from schedulefree import AdamWScheduleFree



class OptFineTuneTabPFN:
    '''
    This class implements a fine-tuning strategy for tabpfn classifiers using AES, with hyperparameter optimization.
    The procedure is designed to work with a single dataframe.
    The optimization process is integrated into the Optuna framework. 
    Specifically, an instance of this class should be passed to the optimize method of an Optuna Study object.
    The learning rate and batch size are treated as tunable hyperparameters. 
    The learning rate is sampled exponentially, while the batch size is sampled linearly.
    
    The fine-tuning procedure implements three stopping logic:
        1) early stopping on training time;
        2) AES;
        3) Number of training steps. 
        
    ----------
    Parameters:
        path_base_model (str | Path | Literal["auto"], optional): 
            Path to the base tabpfn model. Defaults to "auto", in which case the model is founded automatically using tabpfn utilities.
        X_train (pd.DataFrame): training pandas dataframe.
        y_train (pd.Series): training target series.
        X_val (pd.DataFrame): validation pandas dataframe.
        y_val (pd.Series): validation target series.
        fine_tune_setup (FineTuneSetup): FineTuneSetup instance with the finetune directivies.
        opt_setup (OptSetup): OptSetup instance containing the optimization directivies. 
        softmax_temperature (float | None, optional): 
            Number that divides the raw logits in a multiclassification setting if not None. 
            Defaults to 0.9 which is the default used by tabpfn classifiers.  
        random_seed (int, optional): Seed that control the randomness for all the processed involved. Defaults to 50.
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
    
    scaler_setting (dict): Dict with initial scaler settings (are fixed since the scale factor is dinamicaly adapted). 
    
    n_classes (int): Number of classes determined by the number of different labels of the y training set.

    task_type (Literal["binary", "multiclass"]): type of classification problem. 
        Informs the choice of the cross entropy loss "type" to use.

    loss (BCEWithLogitsLoss | CrossEntropyLoss): "type" of cross entropy loss to use.
        BCEWithLogitsLoss class if the classification problem is binary otherwise CrossEntropyLoss class.
    
    use_autocast (bool): True if the GPU is available and False otherwise.
        Enable mixed precision training (stable only on GPU).
    
    current_trial (str): Reports the current trial when __call__ is called. String like "trial1".
    
    trials_reports_aes (list[list]): Reports AES info about the fine tuning process for every trials.
        This are ordered in increasing order (from trial 0 to N). 
        Each list reports the total, effective, skipped steps and stop mechanism info in this order.
    
    single_report_aes (list): Reports info about the fne tuning process done without HP optimization.
        The list reports the total, effective, skipped steps and stop mechanism info in this order.
    
    trials_reports_scaler (dict[str, list[list]]): Reports for every trial a list of info about the scaler.
        Is a dict where every key is a current_trial string.
        Each sublist contains info about a learning step. They are organized from step 1 to total_steps.
    
    single_report_scaler (list[list]): Reports the scaler info during a single non optimized finetune process.
        Each sublist contains info about a learning step. They are organized from step 1 to total_steps.

    single_report_val_loss (list): Reports the validation loss values registered during the finetuning process.
        Only the values for the effective steps are considered.  
    '''
    def __init__(
        self,
        *,
        path_base_model: str | Path | Literal["auto"] = "auto",
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame, 
        y_val: pd.Series, 
        fine_tune_setup: FineTuneSetup,
        opt_setup: OptSetup,
        softmax_temperature: float | None = 0.9,
        random_seed: int = 50, 
        device: Literal["auto"] = "auto"
    ):   
        self._check_data(X_train, y_train, X_val, y_val)
        self.X_train, self.y_train, self.X_val, self.y_val = X_train, y_train, X_val, y_val
        self.path_base_model = path_base_model
        self.min_lr: float = opt_setup.min_lr
        self.max_lr: float = opt_setup.max_lr
        self.min_bs: int = opt_setup.min_bs
        self.max_bs:int = opt_setup.max_bs
        self.adaptive_rate: float = fine_tune_setup.adaptive_rate
        self.adaptive_offset: float = fine_tune_setup.adaptive_offset
        self.min_patience: int = fine_tune_setup.min_patience
        self.max_patience: int = fine_tune_setup.max_patience
        self.time_limit: float = infer_time_limit(X_train) if fine_tune_setup.time_limit == "infer" else fine_tune_setup.time_limit
        self.max_steps: int = fine_tune_setup.max_steps
        self.softmax_temperature = softmax_temperature
        self.random_seed = random_seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_classes = len(self.y_train.unique())
        self.task_type = "binary" if self.n_classes == 2 else "multiclass"
        self.Loss = BCEWithLogitsLoss if self.task_type == "binary" else CrossEntropyLoss
        self.use_autocast = True if self.device == "cuda" else False
        self.save_path_fine_tuned_model: str = None
        
        self.scaler_settings = {
            "init_scale": 2 ** 20,
            "growth_factor": 2, 
            "backoff_factor": 0.5,
            "growth_interval": 100
        }

        self.current_trial: str = None
        self.trials_reports_aes: list[list] = []
        self.trials_reports_scaler: dict[str, list[list]] = {}
        self.single_report_aes: list = []
        self.single_report_scaler: list = []
        self.single_report_val_loss: list = []




    def __call__(self, trial: optuna.Trial) -> float:
        '''
        Method called when the 'OptFineTuneTabpfn' instance is passed in the optuna optimizer.
        The models are currently not saved.
        Returns: The best validation loss.
        '''
        self.current_trial = "trial" + str(trial.number)
        lr = trial.suggest_float("learning_rate", self.min_lr, self.max_lr, log=True)
        batch_size = trial.suggest_int("batch_size", self.min_bs, self.max_bs)
        best_val_loss = self._fine_tune_tabpfn_clf(lr, batch_size, file=None, report="trials", trial=trial)
        return best_val_loss



    def fine_tune_tabpfn_clf(self, learning_rate: float, batch_size: int, file: str | None, return_val_loss: bool = False) -> None | float:
        '''
        Fine tune the tabpfn classifier on the data passed in "__init__" method with specific learning rate and batch size values.
        Parameters:
            learning_rate (float): Learning rate to use.
            batch_size (int): Batch size to use.
            file (str | None): String reporting the filepath (path + filename) to which the model is saved.
                If None the model is not saved.
            return_val_loss (bool, optional): Whether to return the validation loss of the finetuned model.
                Defaults to False.
        Returns: None or the validation loss.
        '''
        val_loss = self._fine_tune_tabpfn_clf(learning_rate, batch_size, file=file, report="single")
        if return_val_loss: return val_loss



    def _fine_tune_tabpfn_clf(
            self, 
            learning_rate: float, 
            batch_size: int, 
            report: Literal["trials", "single"],
            file: str | None = None,
            trial: optuna.Trial | None = None
        ) -> float:
        '''
        Fine tune a tabpfn classifier on a single dataset.
        We load the model and set the optimizer and the scaler in order to secure a "fresh" start for every trial.
        
        Currently the validation loss is computed using the training set as context.
        Should we generate a context specific for the validation from the validation set ???

        Parameters:
            learning rate: learning rate to use.
            batch_size: batch size to use.
            report: Hints in which attributes to store the aes and scaler reports.
            file: The filepath in which the model is saved. If None, the default, the model is not saved.
            trial: Trail object passed in the optimization scenario.
        
        Returns: the best validation loss.
        '''
        model, checkpoint_config, criterion, scaler, aes = self._setup_training()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        
        train_x, train_y, val_x, val_y = self._prepare_data_to_forward(
            self.X_train, self.y_train, self.X_val, self.y_val
        )
        
        if self.device == "cuda":
            model = model.to(device="cuda")
            train_x = train_x.to(device="cuda")
            train_y = train_y.to(device="cuda")
            val_x = val_x.to(device="cuda")
            val_y = val_y.to(device="cuda")

        data_loader = self._create_data_loader(batch_size)

        # start finetune process
        start_time = time.time()
        model.eval()
        best_val_loss = self._forward_validation_step(model, criterion, train_x, train_y, val_x, val_y)

        # saving the base model since its the best right now
        if file is not None:
            save_model(model, file, checkpoint_config)

        # start training
        total_steps = 0
        effective_steps = 0  

        while True:
            for X_batch, y_batch in data_loader:
                total_steps += 1
                train_batch_x, train_batch_y, test_batch_x, test_batch_y = self._process_training_batch(X_batch, y_batch)
                
                model.train()
                with torch.autocast(device_type=self.device, enabled=self.use_autocast):
                    loss = self._forward_step(model, criterion, train_batch_x, train_batch_y, test_batch_x, test_batch_y) 
                
                # scaling, clipping and updating
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                _ = clip_grad_norm_(model.parameters(), max_norm=1, error_if_nonfinite=False)
                scaler.step(optimizer)
                optimizer.zero_grad()
                
                ori_scale = scaler.get_scale()
                scaler.update()
                updated_scale = scaler.get_scale()
                
                if self.use_autocast:
                    self._construct_scaler_report(
                        updated_scale,
                        scaler.get_growth_factor(), 
                        scaler.get_backoff_factor(), 
                        scaler.get_growth_interval(), 
                        report
                    )
                
                if updated_scale < ori_scale:
                    step_with_update = False
                else:
                    effective_steps += 1
                    step_with_update = True
                
                if step_with_update:
                    model.eval()
                    updated_val_loss = self._forward_validation_step(model, criterion, train_x, train_y, val_x, val_y)
                    if report == "trials":
                        trial.report(updated_val_loss, effective_steps)
                    else:
                        self.single_report_val_loss.append(best_val_loss)

                # stopping logics
                if step_with_update and updated_val_loss < best_val_loss:
                    best_val_loss = updated_val_loss
                    aes.set_best_round(effective_steps)
                    aes.update_patience()
                    # overwriting the model with the best one
                    if file is not None: 
                        save_model(model, file, checkpoint_config)
                else:
                    residual_patience = aes.get_remaining_patience(effective_steps)
                    if residual_patience <= 0:
                        self._construct_finetune_report(total_steps, effective_steps, "patience termination", report)
                        return best_val_loss
                
                elapsed_time = time.time() - start_time
                
                if elapsed_time >= self.time_limit:
                    self._construct_finetune_report(total_steps, effective_steps, "time termination", report)
                    return best_val_loss
                
                if total_steps == self.max_steps:
                    self._construct_finetune_report(total_steps, effective_steps, "steps termination", report)
                    return best_val_loss



    def _setup_training(self):
        '''
        Setup training components: model, scaler and early stopping.
        Loads the model spefied in "__init__" using the tabpfn "load_model_criterion_config" utility.
        '''
        path_model = None if self.path_base_model == "auto" else self.path_base_model
        
        model, _, checkpoint_config = load_model_criterion_config(
            model_path=path_model,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=False,
            which="classifier",
            download=True,
            version="v2",
            model_seed=self.random_seed
        )

        checkpoint_config = checkpoint_config.__dict__
        criterion = self.Loss(reduction="none")
        
        scaler = GradScaler(
            device=self.device,
            init_scale=self.scaler_settings["init_scale"],
            growth_factor=self.scaler_settings["growth_factor"],
            backoff_factor=self.scaler_settings["backoff_factor"],
            growth_interval=self.scaler_settings["growth_interval"],
            enabled=self.use_autocast
        )
        
        aes = AdaptiveEarlyStopping(
            self.adaptive_rate, 
            self.adaptive_offset, 
            self.min_patience, 
            self.max_patience
        )

        return model, checkpoint_config, criterion, scaler, aes



    def _create_data_loader(self, batch_size: int) -> DataLoader:
        '''
        Create and return the data loader for training.
        Drops partial batches in order to not "consume" (eventually) in the aes procedure
        training steps when the context window is reduced.
        '''
        X = torch.tensor(self.X_train.to_numpy(), dtype=torch.float32)
        y = torch.tensor(self.y_train.to_numpy(), dtype=torch.float32)
        loader_rng = torch.Generator("cpu").manual_seed(self.random_seed)
        return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True, generator=loader_rng, drop_last=True)



    def _forward_step(
            self,
            model: PerFeatureTransformer,
            criterion: BCEWithLogitsLoss | CrossEntropyLoss,
            train_x: torch.Tensor,   # (n_samples, 1, n_features)
            train_y: torch.Tensor,  # (n_samples, 1, 1)
            test_x: torch.Tensor,   # (n_samples, 1, n_features)
            test_y: torch.Tensor,   # (n_samples, 1, 1)
        ) -> torch.Tensor:
        '''
        Perform the forward pass.
        Needs the crtierion, context and test tensors.
        Returns: The loss as a tensor of a single scalar.
        '''
        pred_logits = model(train_x=train_x, train_y=train_y, test_x=test_x)
        # squeezing dimension 1 since we work always with a single batch
        pred_logits = pred_logits.squeeze(dim=1)

        if self.task_type == "multiclass":
            test_y = test_y.flatten().long()
            pred_logits = pred_logits[:, :self.n_classes]
            if self.softmax_temperature is not None:
                pred_logits = pred_logits / self.softmax_temperature
        else:
            # select positive class logits only
            pred_logits = pred_logits[:, 1]
            test_y = test_y.squeeze(dim=1).float()
                
        loss = criterion(pred_logits, test_y).mean()
        return loss
    


    def _forward_validation_step(
            self,
            model: PerFeatureTransformer,
            criterion: BCEWithLogitsLoss | CrossEntropyLoss,
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



    def _process_training_batch(self, X_batch: torch.Tensor, y_batch: torch.Tensor) -> tuple[torch.Tensor]:
        '''
        Process a training batch and prepare it for the forward pass.
        Returns: The x and y train and test batch sets.
        '''
        y_unique, y_counts = y_batch.unique(return_counts=True)
        stratify = y_batch if y_unique.shape[0] > 1 and 1 not in y_counts else None
        
        train_batch_x, test_batch_x, train_batch_y, test_batch_y = self._prepare_data_to_forward(
            *train_test_split(
                X_batch, 
                y_batch, 
                test_size=0.3, 
                random_state=self.random_seed, 
                stratify=stratify
            )
        )

        if self.device == "cuda":
            train_batch_x = train_batch_x.to(device="cuda")
            train_batch_y = train_batch_y.to(device="cuda")
            test_batch_x = test_batch_x.to(device="cuda")
            test_batch_y = test_batch_y.to(device="cuda")
            
        return train_batch_x, train_batch_y, test_batch_x, test_batch_y



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



    @staticmethod
    def _check_data(X_train, y_train, X_val, y_val) -> None:
        '''Check input data types.'''
        for x_data in [X_train, X_val]:
            if not isinstance(x_data, pd.DataFrame):
                raise TypeError("X_train and X_val must be pandas DataFrame objects")
        for y_data in [y_train, y_val]:
            if not isinstance(y_data, pd.Series):
                raise TypeError("y_train and y_val must be pandas Series objects")




    def _construct_finetune_report(
            self,
            total_steps: int,
            effective_steps: int, 
            stop_mechanisms: str, 
            where: Literal["trials", "single"]
        ) -> None:
        '''
        Construct a report about the finetuning process with the input info. 
        This is stored in the attribute "trials_reports_aes" if where is equal to "trials" otherwise in "single_report_aes".
        Returns: None
        '''
        info = [total_steps, effective_steps, stop_mechanisms]
        if where == "trials":
            self.trials_reports_aes.append(info)
        else:
            self.single_report_aes = info



    def _construct_scaler_report(
            self,
            scale_factor: float,
            growth_factor: float,
            backoff_factor: int,
            growth_interval: int,
            where:  Literal["trials", "single"]
    ) -> None:
        '''
        Construct a report with the input info during the finetuning process.
        The info are stored in the attribute hinted by 'where'.
        Returns: None
        '''
        info = [scale_factor, growth_factor, backoff_factor, growth_interval]
        if where == "trials":
            if self.current_trial not in self.trials_reports_scaler.keys():
                self.trials_reports_scaler[self.current_trial] = [info]
            else:
                self.trials_reports_scaler[self.current_trial].append(info)
        else:
            self.single_report_scaler.append(info)



    def get_df_aes_report(self, what: Literal["single_finetune", "opt_finetune"]) -> pd.DataFrame | None:
        '''
        Get the Adaptive early stopping report in a pandas Dataframe.
        
        Parameters:
        what (Literal["single_finetune", "opt_finetune"]): Specifies the type of report to retrieve.
            Since the reports are stored in separate attributes, one must choose between:
            - "single_finetune": Returns the report for the single fine-tuning process (without optimization).
            - "opt_finetune": Returns the report for the fine-tuning process with optimization.

        Returns: The dataframe or None if no aes information is available.
        '''
        data = self.single_report_aes if what == "single_finetune" else self.trials_reports_aes
        if data:
            report = self._organize_df_aes_report(data)
            skipped_steps = report["total_steps"] - report["effective_steps"]
            report.insert(loc=2, column="skipped_steps", value=skipped_steps)
            return report


    
    def get_df_scaler_report(self, what: Literal["single_finetune", "opt_finetune"]) ->  pd.DataFrame | dict[str, pd.DataFrame] | None:
        '''
        Organizes and retrieves the scaler report/s as Pandas DataFrames.
        
        Parameters:
            what (Literal["single_finetune", "opt_finetune"]): Specifies the type of report to retrieve.
                Since the reports are stored in separate attributes, one must choose between:
                - "single_finetune": Returns the report for the single fine-tuning process (without optimization).
                - "opt_finetune": Returns the report for the fine-tuning process with optimization.
        
        Returns: None if the specified report is unavailable. 
        A single DataFrame if what="single_finetune".
        A dictionary of DataFrames if what="opt_finetune".
        '''
        report = None
        
        if what == "single_finetune":
            if self.single_report_scaler:
                report = self._organize_df_scaler_report(self.single_report_scaler)
        else:
            if self.trials_reports_scaler:
                report = {}
                for k, v in self.trials_reports_scaler.items():
                    report[k] = self._organize_df_scaler_report(v)

        return report



    @staticmethod
    def _organize_df_aes_report(list_report: list | list[list]) -> pd.DataFrame:
        '''
        Helps to organize a single aes dataframe report takin the list/s with aes info.
        '''
        index = None
        if isinstance(list_report[0], list):
            index = index=["trial" + str(i) for i in range(len(list_report))]
        else:
            list_report = [list_report]
        return pd.DataFrame(list_report, index=index, columns=["total_steps", "effective_steps", "stop_mechanisms"])



    @staticmethod
    def _organize_df_scaler_report(list_report: list[list]) -> pd.DataFrame:
        '''
        Helps to generate a single scaler dataframe report taking the list with the step lists info. 
        '''
        index = ["step" + str(i) for i in range(0, len(list_report))]
        return pd.DataFrame(data=list_report, index=index, columns=["scale_factor", "growth_factor", "backoff_factor", "growth_interval"])
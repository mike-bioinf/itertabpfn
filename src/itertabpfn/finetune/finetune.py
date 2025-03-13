import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal, Generator
from sklearn.model_selection import train_test_split, StratifiedKFold
import optuna
import torch
import torch.optim as optim
from torch import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from tabpfn.base import load_model_criterion_config
from tabpfn.model.transformer import PerFeatureTransformer
from itertabpfn.finetune.adaptive_early_stopping import AdaptiveEarlyStopping
from itertabpfn.finetune.model_utils import save_model
from itertabpfn.finetune.setup import FineTuneSetup, OptSetup, infer_time_limit
from itertabpfn.finetune.report import *
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
        2) adaptive early stopping;
        3) number of training steps. 
        
    ----------
    Parameters:
        path_base_model (str | Path | Literal["auto"], optional): 
            Path to the base tabpfn model. Defaults to "auto", in which case the model is founded automatically using tabpfn utilities.
        X_train (pd.DataFrame): training pandas dataframe.
        y_train (pd.Series): training target series.
        X_val (pd.DataFrame): validation pandas dataframe.
        y_val (pd.Series): validation target series.
        fine_tune_setup (FineTuneSetup): FineTuneSetup instance with the finetune directivies.
        opt_setup (OptSetup | None): OptSetup instance containing the optimization directivies, 
            or None if no optimization is involved.
        softmax_temperature (float | None, optional): 
            Number that divides the raw logits in a multiclassification setting if not None. 
            Defaults to 0.9 which is the default used by tabpfn classifiers.  
        random_seed (int, optional): Seed that control the randomness for all the processed involved. Defaults to 50.
        device (Literal["auto"], optional): Search automaticaly for the GPU falling otherwise on the CPU.
        
    ----------
    Attributes:

    opts (OptSetup | None): data instance with the following attributes:
        min_lr (float): The inferior limit for the learning_rate searchable range.
        max_lr (float): The superior limit for the learning_rate searchable range.
        min_bs (int): The inferior limit for the batch_size searchable range.
        max_bs (int): The superior limit for the batch_size searchable range.

    fts (FineTuneSetup): data instance with the following attributes:   
        adaptive_rate (float): The rate of increase in patience.
        adaptive_offset (int): The initial patience at round 0.
        min_patience (int): The minimum value of patience.
        max_patience (int): The maximum value of patience.
        time_limit (int): Maximum time in seconds after which the fine tuning process is stopped.
        max_steps (int): Maximum number of learning step.
    
    scaler_setting (dict): Dict with initial scaler settings (are fixed since the scale factor is dinamically adapted). 
    
    n_classes (int): Number of classes determined by the number of different labels of the y training set.

    task_type (Literal["binary", "multiclass"]): type of classification problem. 
        Informs the choice of the cross entropy loss "type" to use.

    loss (BCEWithLogitsLoss | CrossEntropyLoss): "type" of cross entropy loss to use.
        BCEWithLogitsLoss class if the classification problem is binary otherwise CrossEntropyLoss class.
    
    use_autocast (bool): True if the GPU is available and False otherwise. Enable mixed precision training (stable only on GPU).

    logger (logging.Logger): logger instance used to manage logging in the single finetune scenario.
    
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
        opt_setup: OptSetup | None,
        softmax_temperature: float | None = 0.9,
        random_seed: int = 50, 
        device: Literal["auto"] = "auto"
    ):   
        self._check_data(X_train, y_train, X_val, y_val)
        self.X_train, self.y_train, self.X_val, self.y_val = X_train, y_train, X_val, y_val
        self.path_base_model = path_base_model
        
        fine_tune_setup.time_limit = infer_time_limit(X_train) if fine_tune_setup.time_limit == "infer" else fine_tune_setup.time_limit
        self.opts = opt_setup
        self.fts = fine_tune_setup
        
        self.softmax_temperature = softmax_temperature
        self.random_seed = random_seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_classes = len(self.y_train.unique())
        self.task_type = "binary" if self.n_classes == 2 else "multiclass"
        self.Loss = BCEWithLogitsLoss if self.task_type == "binary" else CrossEntropyLoss
        self.use_autocast = True if self.device == "cuda" else False
        self.logger: logging.Logger = None
        
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
        The Trial object is automatically passed in input. 
        The resulting models are currently not saved.
        Returns: The best validation loss.
        '''
        self.current_trial = "trial" + str(trial.number)
        lr = trial.suggest_float("learning_rate", self.opts.min_lr, self.opts.max_lr, log=True)
        batch_size = trial.suggest_int("batch_size", self.opts.min_bs, self.opts.max_bs)
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
        create_logger(self)
        print(f" ============== Finetuning with learning rate of {learning_rate} and batch size of {batch_size}  ================ \n")
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

        index_loader = self._create_index_loader(batch_size)

        # start finetune process
        start_time = time.time()
        model.eval()
        init_val_loss = self._forward_validation_step(model, criterion, train_x, train_y, val_x, val_y)
        best_val_loss = init_val_loss

        # saving the base model since its the best right now
        if file is not None:
            save_model(model, file, checkpoint_config)

        # start training
        total_steps = 0
        effective_steps = 0  

        for _ , idx_batch in index_loader:
            total_steps += 1
            
            X_batch = self.X_train.iloc[idx_batch, :]
            y_batch = self.y_train.iloc[idx_batch]
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
                construct_scaler_report(
                    opt_obj=self,
                    scale_factor=updated_scale,
                    growth_factor=scaler.get_growth_factor(), 
                    backoff_factor=scaler.get_backoff_factor(), 
                    growth_interval=scaler.get_growth_interval(), 
                    where=report
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

            # logging
            total_time_spent = time.time() - start_time
            time_remaining = max(0, self.fts.time_limit - total_time_spent)

            self.logger.debug(
                f"""
                Step {total_steps}/{self.fts.max_steps}
                Total Time Spent: '{total_time_spent}'
                Time remaining: '{time_remaining}'
                Initial Validation Loss: '{init_val_loss}'
                Best Validation Loss: '{best_val_loss}'
                """.replace("\n", "\t")
            )

            # stopping logics
            if step_with_update and updated_val_loss < best_val_loss:
                best_val_loss = updated_val_loss
                aes.set_best_round(effective_steps)
                aes.update_patience()
                # overwriting the model with the new best one
                if file is not None: 
                    save_model(model, file, checkpoint_config)
            else:
                residual_patience = aes.get_remaining_patience(effective_steps)
                if residual_patience <= 0:
                    construct_aes_report(self, total_steps, effective_steps, "patience termination", report)
                    return best_val_loss
            
            elapsed_time = time.time() - start_time
            
            if elapsed_time >= self.fts.time_limit:
                construct_aes_report(self, total_steps, effective_steps, "time termination", report)
                return best_val_loss
            
            if total_steps == self.fts.max_steps:
                construct_aes_report(self, total_steps, effective_steps, "steps termination", report)
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
            self.fts.adaptive_rate, 
            self.fts.adaptive_offset, 
            self.fts.min_patience, 
            self.fts.max_patience
        )

        return model, checkpoint_config, criterion, scaler, aes



    def _create_index_loader(self, batch_size: int) -> Generator[tuple[int, int], None, None]:
        '''
        Create the "index data loader" for the training process.
        
        This loader is NOT a DataLoader object but instead a Generator that yields an endless 
        stream of train/test stratified split indexes organized in a binary tuple.
        
        The designed use for such iterators is to discard the "train" information and take only the "test" idx from which obtain the batch.
        In this way we can obtain stratified batch of the desired dimensions.
        
        Note: to generate batch of the desired dimension we discard some samples. 
        This is functionaly equivalent to discard partial batches using a standard torch data loader.
        Such behaviour is justified in the finetune framework since it allows to not "eventually consume" 
        steps when the context window is reduced in the AES-controlled-training.
        '''
        rng_split = np.random.default_rng(self.random_seed)
        rng_selection = np.random.default_rng(self.random_seed)

        n_samples = self.X_train.shape[0]
        n_splits = int(n_samples // batch_size)
        rest = n_samples % batch_size 

        while True:
            # remove samples to obtain the desired batches dimension
            # the selection is random and therefore should approximately respect the distribution of classes
            if rest:
                idx_to_keep = rng_selection.choice(n_samples, size=n_samples-rest, replace=False, shuffle=True)
                X_sel = self.X_train.iloc[idx_to_keep, :]
                y_sel = self.y_train.iloc[idx_to_keep]
            else:
                X_sel = self.X_train
                y_sel = self.y_train

            yield from StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=rng_split.integers(0, np.iinfo(np.int32).max)
                ).split(X_sel, y_sel)



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



    def _process_training_batch(self, X_batch: pd.DataFrame, y_batch: pd.Series) -> tuple[torch.Tensor]:
        '''
        Process a training batch and prepare it for the forward pass.
        Returns: The x and y train and test batch sets.
        '''
        y_unique, y_counts = np.unique(y_batch.to_numpy(), return_counts=True)
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



    def get_df_report(self, stats: Literal["aes", "scaler", "val_loss"], what_process: Literal["single_finetune", "opt_finetune"]) -> pd.DataFrame | None:
        '''
        Retrieve and organize the internal finetune-related collected info as pandas DataFrames.
        Is not possible to retrieve with this function the validation loss info for the different trials in the optimized route.
        To do this one must inspect the optuna Study object which store this info. 
        In detail see the "trials" attribute which store the Trial objects and "intermediate_values" attribute for each one of them.

        Parameters:
            stats (Literal["aes", "scaler", "val_loss]): 
                Which info to retrieve, the ones related to the AES procedure, to the gradient scaler or to the validation loss.
            what_process (Literal["single_finetune", "opt_finetune"]): Of which process to retrive the info: 
                - "single_finetune" for the info collected about the finetune strategy with fixed HPs.
                - "opt_finetune" for the info collected about the finetune strategy with learned HPs.
        
        Returns: The DataFrame or None if the target information is not available.
        '''
        if stats == "val_loss" and what_process == "opt_finetune":
            raise ValueError("To retrieve info about the validation losses for the different trials inspect the optuna Study object.")
        
        if stats == "aes":
            return get_df_aes_report(self, what_process)
        elif stats == "scaler":
            return get_df_scaler_report(self, what_process)
        else:
            return get_df_val_loss_report(self)

import logging
import sys
from typing import Literal
import pandas as pd
# from itertabpfn.finetune import OptFineTuneTabPFN     ### circular import problem



def construct_aes_report(opt_obj, total_steps: int, effective_steps: int, stop_mechanisms: str, where: Literal["trials", "single"]) -> None:
    '''Construct the aes report for the OptFineTuneTabPFN object.'''
    info = [total_steps, effective_steps, stop_mechanisms]
    if where == "trials":
        opt_obj.trials_reports_aes.append(info)
    else:
        opt_obj.single_report_aes = info



def construct_scaler_report(opt_obj, scale_factor: float, growth_factor: float, backoff_factor: int, growth_interval: int, where: Literal["trials", "single"]) -> None:
    '''Construct the scaler report for the OptFineTuneTabPFN object.'''
    info = [scale_factor, growth_factor, backoff_factor, growth_interval]
    if where == "trials":
        if opt_obj.current_trial not in opt_obj.trials_reports_scaler.keys():
            opt_obj.trials_reports_scaler[opt_obj.current_trial] = [info]
        else:
            opt_obj.trials_reports_scaler[opt_obj.current_trial].append(info)
    else:
        opt_obj.single_report_scaler.append(info)



def get_df_aes_report(opt_obj, what: Literal["single_finetune", "opt_finetune"]) -> pd.DataFrame | None:
    '''Get the Adaptive early stopping report in a pandas Dataframe.'''
    data = opt_obj.single_report_aes if what == "single_finetune" else opt_obj.trials_reports_aes
    if data:
        report = organize_df_aes_report(data)
        skipped_steps = report["total_steps"] - report["effective_steps"]
        report.insert(loc=2, column="skipped_steps", value=skipped_steps)
        return report



def get_df_scaler_report(opt_obj, what: Literal["single_finetune", "opt_finetune"]) ->  pd.DataFrame | None:
    '''Organizes and retrieves the scaler report/s into a Pandas DataFrame.'''
    report = None
    
    if what == "single_finetune":
        if opt_obj.single_report_scaler:
            report = organize_df_scaler_report(opt_obj.single_report_scaler)
    else:
        if opt_obj.trials_reports_scaler:
            report = pd.concat([organize_df_scaler_report(info, trial_name) for trial_name, info in opt_obj.trials_reports_scaler.items()], axis=0)

    return report



def get_df_val_loss_report(opt_obj) -> pd.DataFrame | None:
    '''Organizes and retrieves the validation loss info into a Pandas DataFrame.'''
    if opt_obj.single_report_val_loss:
        index = ["step" + str(i) for i in range(0, len(opt_obj.single_report_val_loss))]
        return pd.DataFrame(opt_obj.single_report_val_loss, index=index, columns=["validation_loss"])



def organize_df_aes_report(list_report: list | list[list]) -> pd.DataFrame:
    '''Organizes the aes info into a single DataFrame.'''
    index = None
    if isinstance(list_report[0], list):
        index = index=["trial" + str(i) for i in range(len(list_report))]
    else:
        list_report = [list_report]
    return pd.DataFrame(list_report, index=index, columns=["total_steps", "effective_steps", "stop_mechanisms"])



def organize_df_scaler_report(list_report: list[list], trial: str = None) -> pd.DataFrame:
    '''Organize the scaler info into a single DataFrame.'''
    index = ["step" + str(i) for i in range(0, len(list_report))]
    if trial: 
        index = pd.MultiIndex.from_product([[trial], index], names=["trial", "step"])
    return pd.DataFrame(data=list_report, index=index, columns=["scale_factor", "growth_factor", "backoff_factor", "growth_interval"])



def create_logger(opt_obj) -> None:
    '''Create and set in the logger attribute of the opt_obj the logger instance.'''
    logger = logging.getLogger("finetune")
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt="{asctime} \t {message}", datefmt="%Y-%m-%d %H:%M", style="{")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    opt_obj.logger =logger

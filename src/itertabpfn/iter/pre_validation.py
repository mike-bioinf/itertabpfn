import pandas as pd
from pandas.api.types import is_integer_dtype
from warnings import warn


def check_target_cols(datasets: dict[str, tuple[pd.DataFrame, str]]) -> None:
    'Check datasets argument in type and presence of target column'
    for k, v in datasets.items():
        df, target = v
        if not target in df.columns:
            raise ValueError(f"'{target}' column is not found in dataframe '{k}'.")


def cast_target_cols(datasets: dict[str, tuple[pd.DataFrame, str]]) -> dict[str, tuple[pd.DataFrame, str]]:
    'Cast the target column to integer type.'
    for k, (df, target) in datasets.items():
        target_column = df[target]
        if not is_integer_dtype(target_column):
            warn(f"The '{target}' target column of '{k}' is casted IN PLACE to integer type (int64).")
            df[target] = target_column.astype(int)
    return datasets
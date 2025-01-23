import pandas as pd


def check_target_cols(datasets: dict[str, tuple[pd.DataFrame, str]]) -> None:
    'Check datasets arguments in type and presence of target column'
    for k, v in datasets.items():
        df, target = v
        if not target in df.columns:
            raise ValueError(f"'{target}' column is not found in dataframe '{k}'.")

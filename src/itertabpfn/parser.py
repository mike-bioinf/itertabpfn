import numpy as np
import pandas as pd
import re


def parse_pred_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Parse specific columns of the predicition dataframe returned by the IterTabPFNClassifier.
    In detail the following columns are parsed from strings to numpy arrays: 
    "number_classes", "array_test_label", "array_pred_label", "array_pred_proba".
    In particular the "number_classes" values are converted to binary tuples of numpy arrays.
    The function expects all columns to be present. If one of them is missing an error is raised.
    Parameters:
        df: prediction dataframe returned by the IterTabPFNClassifier class.
    Returns: The parsed DataFrame.
    '''
    columns_to_parse = ["number_classes", "array_test_label", "array_pred_label", "array_pred_proba"]
    
    for column in columns_to_parse:
        if column not in df.columns:
            raise ValueError(f"'{column}' column is missing in df.")
    
    for column in columns_to_parse[:3]:
        converter = convert_number_classes if column == "number_classes" else convert_string_to_array
        df[column] = df[column].apply(converter)

    df["array_pred_proba"] = df.apply(lambda row: convert_string_to_biarray(row["array_pred_proba"], row["number_classes"][0].size), axis=1)
    return df



def convert_string_to_array(string: str, dtype = np.int64, **nkwargs) -> np.ndarray:
    '''
    Convert the string in input to a 1D numpy array.
    Parameters:
        string (str): String in input to convert.
        dtype (str | data-type, optional): datatype of the resulting numpy array. 
            Can be a string or numpy datatype. Defaults to 'np.int64'.
        nkwargs: Allow to uniform its use with other convert_* functions accepting different parameters.
    Returns: The numpy array.
    '''
    return np.fromstring(re.sub(r"[\[\]\n]", "", string), sep=" ", dtype=dtype)



def convert_string_to_biarray(string: str, ncols: int, dtype = np.float64, **nkwargs) -> np.ndarray:
    '''
    Convert the string in input to a 2D array. The elements of the string are added in row-wise order.
    The number of rows is inferred from the number of columns and elements.
    Parameters:
        string (str): String in input to convert.
        dtype (str | data-type, optional): datatype of the resulting numpy array. 
            Can be a string or numpy datatype. Defaults to 'np.float64'.
        ncols (int): number of columns of the resulting array.
        nkwargs: Allow to uniform its use with other convert_* functions accepting different parameters.
    Returns: A 2D numpy array.
    '''
    array = np.fromstring(re.sub(r"[\[\]\n]", "", string), sep=" ", dtype=dtype)
    array_reshaped = array.reshape((-1, ncols))
    return array_reshaped



def convert_number_classes(string: str, **nkwargs) -> tuple[np.ndarray]:
    '''
    Convert the number_class string value to a tuple of numpy 1D integer arrays.
    Parameters:
        string (str): String in input to convert.
        nkwargs: Allow to uniform its use with other convert_* functions accepting different parameters.
    Returns: A tuple of 2 numpy arrays.
    '''
    array_numbers = np.fromstring(re.sub(r"[\(\)\[\]array]", "", string), sep = ",", dtype=np.int64)
    array_size = array_numbers.size
    half_size = int(array_size/2)
    array_classes = array_numbers[:half_size]
    array_count_classes = array_numbers[half_size:array_size]
    return (array_classes, array_count_classes)
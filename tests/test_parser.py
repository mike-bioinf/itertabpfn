import pandas as pd
from pathlib import Path
from itertabpfn.parser import parse_pred_dataframe
import numpy as np


def import_pred_dataframe():
    path = Path(__file__).parent / "data/pred_dataframe.txt"
    return pd.read_table(path)    


def import_parsed_pred_dataframe():
    df = import_pred_dataframe()
    return parse_pred_dataframe(df)


def test_number_classes_is_parsed():
    'Test that the "number_classes" column is parsed correctly.'
    parsed_df = import_parsed_pred_dataframe()
    item = parsed_df["number_classes"][0]
    assert isinstance(item, tuple), "Parsed number_classes values should be tuples"
    assert len(item) == 2, "Parsed number_classes values should have exactly two elements"
    assert isinstance(item[0], np.ndarray), "Parsed number_classes values should have a numpy array as first element"
    assert isinstance(item[1], np.ndarray), "Parsed number_classes values should have a numpy array as second element"


def test_array_test_label_is_parsed():
    'Test that the "array_test_label" column is parsed correctly.'
    parsed_df = import_parsed_pred_dataframe()
    item = parsed_df["array_test_label"][0]
    assert len(item.shape) == 1, "Parsed array_test_label values should be 1 dimensional"
    assert item.size, "The parsed value should have length 30"


def test_array_pred_label_is_parsed():
    'Test that the "array_pred_label" column is parsed correctly.'
    parsed_df = import_parsed_pred_dataframe()
    item = parsed_df["array_pred_label"][0]
    assert len(item.shape) == 1, "Parsed array_pred_label values should be 1 dimensional"
    assert item.size, "The parsed value should have length 30"


def test_array_pred_proba_is_parsed():
    'Test that the "array_pred_proba" column is parsed correctly.'
    parsed_df = import_parsed_pred_dataframe()
    item = parsed_df["array_pred_proba"][0]
    assert len(item.shape) == 2, "Parsed array_pred_proba values should be 2 dimensional"
    assert item.shape[1] == 2, "The parsed array should have 2 columns"
    assert item.size == 60, "The parsed value should have length 60"
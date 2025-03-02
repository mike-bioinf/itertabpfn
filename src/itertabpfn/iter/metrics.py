import pandas as pd
import numpy as np
from typing import Literal
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support



def compute_metrics(row: pd.Series, adapt_to_test_labels: bool = True) -> pd.Series:
    '''
    Function to pass in input to the apply method called on the prediction dataframe with axis=1.
    It computes a series of metrics: recall, precision, accuracy, f1 and auc for each row of the prediction dataframe.
    For the "multiclass" rows the function computes recall, precision, f1 and auc for each class in a "one vs rest" approach.
    In this case the resulting order follow the encoded numerical order of classes, assuming the class at that position
    as the positive one. So the first metric refer to the case in which the positive class is 0, then 1, 2, ... .
    Note: in multiclass setting the multiple values computed for each metric are organized in numpy arrays.
        
    Parameters:
        row (pd.Series): row passed by the "apply" method.
        adapt_to_test_labels (bool, optional): In a multiclass setting, determines whether to manually 
            compute binary AUCs instead of relying on the multiclass capabilities of the roc_auc_score function. 
            This is necessary when certain labels are missing in the test set, thus avoiding the error
            "Number of classes in y_true not equal to the number of columns in 'y_score'".
            ATTENTION: a numerical correspondence between the integer encoding of the classes and 
            their positions in the probability array is required. 
            Therefore the class encoding scheme must:
            - Start from zero.
            - Be sequential, without skipping any integers.

    Returns: An indexed Series of recall, precision, f1, accuracy and AUC value/s.
    '''
    y_true, y_pred, y_pred_proba, class_setting = row["array_test_label"], row["array_pred_label"], row["array_pred_proba"], row["class_setting"]
    prf_average = "binary" if class_setting == "binary" else None
    y_pred_proba = y_pred_proba[:, 1] if class_setting == "binary" else y_pred_proba
    precision, recall, f1, _  = precision_recall_fscore_support(y_true, y_pred, average=prf_average, zero_division=np.nan)
    accuracy = accuracy_score(y_true, y_pred)
    # "raise" is the default value for the binary case in sklearn implementation
    auc_multi_class = "raise" if class_setting == "binary" else "ovr"

    if class_setting == "multiclass" and adapt_to_test_labels:
        test_classes = row["number_classes"][0]
        auc = _compute_binary_aucs(test_classes, y_true, y_pred_proba)
    else:
        auc = roc_auc_score(y_true, y_pred_proba, average=None, multi_class=auc_multi_class)

    return pd.Series({"recall": recall, "precision": precision, "f1": f1, "accuracy": accuracy, "auc": auc})
    


def _compute_binary_aucs(integer_classes: np.ndarray, y_true:np.ndarray, array_proba:np.ndarray) -> np.ndarray:
    '''
    Enables to compute auc scores when some classes are not present in the test set.
    In this situation the "multiclass" version of roc_auc_score raise an error since 
    the number of columns in array_proba is not equal to the number of unique labels in y_true.
    To resolve this issue, the multiclass case is resolved in binarized cases,
    needing only the predicted probabilities of the actual positive class.
    ATTENTION: a corresponde between the integer encoded classes and the index of the columns
    in the predicted array is needed and taken for granted.

    Parameters:
        integer_classes (np.ndarray): Array reporting the present integer-encoded classes.
        y_true (np.ndarray): 1D array with the true integer-encoded classes.
        array_proba (np.ndarray): 2D array with the predicted probabilties on the columns.
            The association column-class is done on the intger encoding the class and the 
            integer index of the column.

    Returns: A numpy array of AUCs for the classes reported in integer_classes in the same order. 
    '''
    aucs = []
    for i in integer_classes:
        y_pred_proba = array_proba[:, i]
        y_true_bin = np.where(y_true == i, 1, 0)
        aucs.append(roc_auc_score(y_true_bin, y_pred_proba))
    return np.array(aucs)



def compute_multiclass_average_metrics(row: pd.Series, average: Literal["micro", "macro", "weighted"], auc_comparisons: Literal["ovo", "ovr"]) -> pd.Series:
    '''
    Function to pass in input to the apply method called on the prediction dataframe with axis=1.
    It computes "averaged" values for recall, precision, f1 and auc metrics in a multiclass setting.
    In case of binary rows a Series of np.nan is returned to conserve the rows correspondence between 
    the original and the new dataframe (the one returned by apply).

    Parameters:
        row (pd.Series): row passed by the "apply" method.
        average (Literal["micro", "macro", "weighted"]):
            Averaging method to use. 
            - "micro": the metrics are computed in a single step using extented formulas to accomodate statistics that come for multiple confusion matrices.
            - "macro": the performance metrics are computed for each class and then avaraged.
            - "weighted": the metrics are computed for each class similar in macro-averaging, but then 
                they are scaled by the ratio of number of testing instance of that class over the total number of testing instances.
                These scaled values are then summed together to give the averaged metric.
        auc_comparisons (Literal["ovo", "ovr"]):
            Define the comparison system to use for AUC calculation (ONLY).
            - "ovo" (one vs one): each class as the positive one is compared against all the other single negative ones.
                Compatible only with "macro" averaging method.
            - "ovr" (one vs rest): each class as the positive one is compared against a single negative class made of all the other ones.  
    
    Returns: An indexed Series with the averaged values of recall, precision, f1 and auc.
    '''
    if average == "micro" and auc_comparisons == "ovo":
        raise ValueError("'ovo' strategy is possible only with 'macro' averaging.")
    
    y_true, y_pred, y_pred_prob, class_setting = row["array_test_label"], row["array_pred_label"], row["array_pred_proba"], row["class_setting"]

    if class_setting == "binary":
        dict_row = {
            f"{average}_recall": np.nan,
            f"{average}_precision": np.nan,
            f"{average}_f1": np.nan,
            f"{average}_auc": np.nan
        }
    else:
        recall, precision, f1, _  = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=np.nan)
        dict_row = {
            f"{average}_recall": recall, 
            f"{average}_precision": precision,
            f"{average}_f1": f1,
            f"{average}_auc": roc_auc_score(y_true, y_pred_prob, average=average, multi_class=auc_comparisons)
        }

    return(pd.Series(dict_row))
        
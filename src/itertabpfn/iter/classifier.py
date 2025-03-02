import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, field_validator, validate_call, ConfigDict, PositiveFloat, PositiveInt
from typing import Literal, Generator
from tabpfn import TabPFNClassifier
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from sklearn.model_selection import train_test_split
from itertabpfn.iter.pre_validation import check_target_cols, cast_target_cols



class IterTabPFNClassifier(BaseModel):
    '''
    Class to repeat the inference process wuth a TabPFNClassifier on multiple datasets with different configurations.
    "simple_register" and "ensemble_register" attributes are dictionaries used to track already processed 
    configurations for each dataset for "simple" and "ensemble" classifier types.
    Each dictionary key correspond to a dataset and each entry contains a dict of two lists:
    context sizes: a list of context size values utilized with "context_size" key.
    random states: a list of random state values utilized with "random_state" key.
    The corresponding positions in these two lists represent combinations of values (aka configurations) 
    that have already been processed.

    Parameters:
        datasets (dict[str: tuple(pd.DataFrame, str)]): 
            Dictionary of binary tuples consisting in a pandas Dataframe and a string specifying the name of the target column. 
            The dict keys must be strings identifying the datasets.
        context_sizes (set[float]): 
            A set of floats representing the context sizes precentages to be used during inference. 
            If multiple values are provided, all combinations with the random states values are tried.
            This must always be a set, even if it contains only a single value.
        random_states (set[int]): 
            Set of integers indicating the seeds to use. 
            If multiple values are provided, all combinations with the context size values are tried.
            Must be always a set even if one value is passed.
    '''
    datasets: dict[str, tuple[pd.DataFrame, str]]
    context_sizes: set[PositiveFloat] = Field(min_length=1)
    random_states: set[PositiveInt] = Field(min_length=1)
    simple_clf: TabPFNClassifier = None
    ensemble_clf: AutoTabPFNClassifier = None
    simple_clf_config: dict = None
    ensemble_clf_config: dict = None
    simple_register: dict[str, dict[str, list]] = {}
    ensemble_register:  dict[str, dict[str, list]] = {}
    df_columns_names: list[str] = [
        "dataset", "context_size", "random_state", "type_classifier", "class_setting", "number_classes",
        "array_test_label", "array_pred_label", "array_pred_proba"
    ]
    pred_dataframe: pd.DataFrame = pd.DataFrame(
        columns=[
            "dataset", "context_size", "random_state", "type_classifier", "class_setting", "number_classes",
            "array_test_label", "array_pred_label", "array_pred_proba"
        ]
    )


    class Config:
        'Configuration class used by pydantic'
        arbitrary_types_allowed = True


    @field_validator("datasets")
    @classmethod
    def _manage_target_cols(cls, datasets):
        check_target_cols(datasets)
        datasets = cast_target_cols(datasets)
        return datasets


    def __repr__(self):
        dataset_names = list(self.datasets.keys())
        return(f"dataset_names: {dataset_names}; context_sizes = {self.context_sizes}; random_states = {self.random_states}")



    def set_simple_classifier(self, **kwargs) -> None:
        '''
        Set a simple classifier. Overwrite the pre-existing simple classifier.
        Parameters: 
            Take all parameters taken by TabPFNClassifier class by keywords only.
        Returns: None
        '''
        self.simple_clf = TabPFNClassifier(**kwargs)
        self.simple_clf_config = kwargs



    def set_ensemble_classifier(self, **kwargs) -> None:
        '''
        Set the instructions to build ensemble classifiers using GES tecnique on a random portfolio of pre-available tabpfn models.
        The ensemble classifiers will be build later on every dataset at fit/inference time (very time consuming especially without gpu).
        Calling this method overwrite/cancel the pre-existent fitted ensemble classifier.
        Parameters: 
            **kwargs: 
                Take all parameters taken by AutoTabPFNClassifier class by keywords only.
        Returns: None
        '''
        self.ensemble_clf = AutoTabPFNClassifier(**kwargs)
        self.ensemble_clf_config = kwargs



    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def add_datasets(self, other_datasets: dict[str, tuple[pd.DataFrame, str]]) -> None:
        '''
        Add datasets to the instance.
        Parameters: 
            other_datasets (dict[str: tuple(pd.DataFrame, str)]): 
                A dictionary of datasets added to the initializating ones. No duplicates in keys are allowed.
        Returns: None
        '''
        check_target_cols(other_datasets)
        other_keys = other_datasets.keys()
        keys = self.datasets.keys()

        for key in keys:
            if key in other_keys:
                raise ValueError(f"{key} is an already registered dataset key. Remove the duplicates or change the key.")

        self.datasets.update(other_datasets)
    
    

    @validate_call
    def modify_random_states(self, random_states: set[PositiveInt], mode = Literal["add", "remove", "replace"]) -> None:
        '''
        Modify random states attribute.
        Parameters:
            random_states (set[PositiveInt]):
                Set of integers indicating the seeds to use.   
            mode (Literal["add", "remove", "replace"]): 
                Specify what type of modification to apport with the new values:
                "add" = the new values are added to the existing ones (Union operation)
                "remove" = the new values are subtracted from the existing ones (Difference operation)
                "replace" = the new values replace the old ones. 
        Returns: None
        '''
        self.random_states = self._apply_modify(random_states, mode, "random_states")



    @validate_call
    def modify_context_sizes(self, context_sizes: set[PositiveFloat], mode = Literal["add", "remove", "replace"]):
        '''Modify context sizes attribute'''
        self.context_sizes = self._apply_modify(context_sizes, mode, "context_sizes")



    def _apply_modify(self, new_values: set, mode: Literal["add", "remove", "replace"], property: Literal["context_sizes", "random_states"]):
        '''Helper to control the type fo modification done to context_sizes and random_states attributes'''
        old_values = self.context_sizes if property == "context_sizes" else self.random_states
        if mode == "add":
            final_values = old_values | new_values
        elif mode == "remove":
            final_values = old_values - new_values
        else:
            final_values = new_values
        return final_values



    @validate_call
    def predict(self, types: list[Literal["simple", "ensemble"]]) -> None:
        '''
        Perform predictions with the setted classifiers on all unprocessed configurations.
        A configuration is a unique combination of (dataset, context_size, random_state).
        The original fit tabpfn method is solely used to pass the input data since ICL does not involve real training.
        However for ensemble methods a learning phase on the training set (SHOULD be the one passed as context --> TO CHECK)
        is done in order to learn the ensemble. 
        Either the case in this implementation the traditional fit and predict methods are merged into this single one. 
        The predictions consists in both "raw" labels and probability estimates. In addition also the true labels
        and the number of samples for each class, starting from the 0 encoded class going up, are reported.
        All this info is inserted into a single dataframe (see attribute "pred_dataframe").
        Parameters: 
            types (list[Literal["simple", "ensemble"]]): List with at least one of the two possible strings "simple" and "ensemble". 
                It must be always a list.
        Returns: None       
        '''
        self._validate_classifiers_presence(types)
        pred_rows = self._generate_predictions(types)
        self._update_prediction_dataframe(pred_rows)
        


    def _update_prediction_dataframe(self, new_rows: list[list]) -> None:
        '''
        Update and arrange the pred_dataframe attribute with the new rows.
        Returns: None.
        '''
        if new_rows: 
            pred_df = pd.DataFrame(new_rows, columns=self.df_columns_names)
            self.pred_dataframe = pd.concat([self.pred_dataframe, pred_df], ignore_index=True)
            self.pred_dataframe.sort_values(by=["type_classifier", "dataset", "context_size", "random_state"], axis=0, inplace=True)



    def _validate_classifiers_presence(self, types: list[Literal["simple", "ensemble"]]) -> None:
        '''
        Performs check on the existence of classifier used in the predict method.
        Returns: None
        '''
        if "simple" in types and self.simple_clf is None:
            raise ValueError("The simple classifier has to be initialized.")
        if "ensemble" in types and self.ensemble_clf is None:
            raise ValueError("The ensemble classifier has to be initialized.")



    def _generate_predictions(self, types: list[Literal["simple", "ensemble"]]) -> list[list]:
        '''
        Generate predictions for all unprocessed configurations.
        The result obtained for each configuration is a list that "act" as dataframe row.
        Returns: A list of sublist with config and predictive info in a specific order.
        '''
        predictions = []
        for config in self._get_configurations():
            types_to_process = self._get_types_to_process(config, types)
            if not types_to_process: 
                continue
            for type_to_process in types_to_process:
                predictions.append(self._process_configuration(config, type_to_process))
        return predictions



    def _get_configurations(self) -> Generator[dict, None, None]:
        '''
        Generate all possible configurations from the current setting.
        Returns: A generator object yielding a configuration dict.
        '''
        for name, (data, target) in self.datasets.items():
            for context_size in self.context_sizes:
                for random_state in self.random_states:
                    yield {
                        "name": name,
                        "data": data,
                        "target": target,
                        "context_size": context_size,
                        "random_state": random_state
                    }



    def _get_types_to_process(self, config: dict, types: list[Literal["simple", "ensemble"]]) -> list[str]:
        '''
        Get the clf types to use for the current configuration.
        Returns: A list with two possible optional values "simple" and "ensemble" strings, indicating for which type the config should be processed.
        '''
        types_to_process = []
        name, context_size, random_state = config["name"], config["context_size"], config["random_state"]
        if "simple" in types and not self._is_already_predicted(name, context_size, random_state, "simple"):
            types_to_process.append("simple")
        if "ensemble" in types and not self._is_already_predicted(name, context_size, random_state, "ensemble"):
            types_to_process.append("ensemble")
        return types_to_process



    def _is_already_predicted(self, dataset_name: str, context_size: float, random_state: int, type: Literal["simple", "ensemble"]) -> bool:
        '''
        Check whether the particular configurations specified throught input args has already been processed.
        Returns: A boolean.
        '''
        register = self.simple_register if type == "simple" else self.ensemble_register

        if dataset_name not in register.keys():
            return False
        else:
            dataset_register = register[dataset_name]
            return any([True for cs, rs in zip(dataset_register["context_size"], dataset_register["random_state"]) if cs == context_size and rs == random_state]) 



    def _process_configuration(self, config: dict, type: Literal["simple", "ensemble"]) -> list:
        '''
        Process a specific configurations with the classifier of the selected type.
        Manage the update of registers, the predective inference and dataframe row construction (returned in output).
        Returns: list with the configuration and predictive performance details in a specific order. 
        '''
        data, target, name, context_size, random_state = config["data"], config["target"], config["name"], config["context_size"], config["random_state"]
        X_train, X_test, y_train, y_test = self._split_dataset(data, target, context_size, random_state)
        self._update_register(name, context_size, random_state, type)
        y_pred, y_pred_prob = self._predict(X_train, X_test, y_train, type)
        class_setting = "binary" if y_pred_prob.shape[1] == 2 else "multiclass"
        number_classes = np.unique(y_test, return_counts=True)
        return [name, context_size, random_state, type, class_setting, number_classes, y_test, y_pred, y_pred_prob] 



    @staticmethod
    def _split_dataset(data, target, context_size, random_state) -> list[np.ndarray]:
        '''
        Wrapper of sklearn train_test_split function.
        Returns: tuple of four elements in the following order X_train, X_test, y_train and y_test.
        '''
        X = data.drop(target, axis=1).to_numpy()
        y = data.loc[:, target].to_numpy()
        return train_test_split(X, y, train_size=context_size, random_state=random_state, stratify=y)



    def _update_register(self, dataset_name, context_size, random_state, type: Literal["simple", "ensemble"]) -> None:
        '''
        Updates the register with the input configurations.
        Returns: None
        '''
        register = self.simple_register if type == "simple" else self.ensemble_register

        if dataset_name not in register.keys():
            register[dataset_name] = {"context_size": [context_size], "random_state": [random_state]}
        else:
            register[dataset_name]["context_size"].append(context_size)
            register[dataset_name]["random_state"].append(random_state)
    


    def _predict(self, X_train, X_test, y_train, type: Literal["simple", "ensemble"]) -> tuple[np.ndarray, np.ndarray]:
        '''
        Predict y labels and the probabilities of the class encoded as 1.
        Returns: Binary tuple of 1D numpy arrays of predicated labels and probabilities.
        '''
        clf = self.simple_clf if type == "simple" else self.ensemble_clf
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        pred_prob = clf.predict_proba(X_test)
        return (pred, pred_prob)
        
        

    def save_pred_dataframe(self, file: str, sep: str = "\t") -> None:
        '''
        Save the pred_dataframe attribute into a txt file.
        Parameters:
            file (str): filename path.
            sep (str): string used as sep. Defaults to "\t".
        Returns: None
        '''
        self.pred_dataframe.to_csv(file, sep=sep, index=False)
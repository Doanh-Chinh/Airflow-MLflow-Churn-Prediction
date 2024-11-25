from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd

from steps.config import TrainerConfig, PreprocessConfig
from utils.data_classes import PreprocessingData
import logging
LOGGER = logging.getLogger(__name__)

class PreprocessStep:
    """Preprocessing based on Exploratory Data Analysis done in `notebooks/0_exploratory_data_analysis.ipynb`
    
    Args:
        inference_mode (bool): Training or inference mode.
        preprocessing_data (PreprocessingData): PreprocessingStep output paths."""

    def __init__(
        self,
        inference_mode: bool,
        preprocessing_data: PreprocessingData
    ) -> None:
        self.inference_mode = inference_mode
        self.preprocessing_data = preprocessing_data

    def __call__(self, data_path: Path, stratify_mode=True) -> None:
        """Data is preprocessed then, regarding if inference=True or False:
            * False: Split data into train and test.
            * True: Data preprocessed then returned simply.
        
        Args:
            data_path (Path): Input
            stratify_mode (bool):
                * True: sklearn.model_selection.train_test_split will be called.
                * False: pandas.DataFrame.sample will be called.
        """

        preprocessed_df = pd.read_csv(data_path)
        preprocessed_df = self._preprocess(preprocessed_df)


        if not self.inference_mode: # in the future, should use train_test_split from sklearn with more robust stratified splits and multiple configurations
            if stratify_mode:
                LOGGER.info("sklearn.model_selection.train_test_split will be called.")
                # Split X_input and y_target
                X = preprocessed_df.drop(PreprocessConfig.target, axis=1)
                y = preprocessed_df[PreprocessConfig.target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, \
                    train_size=TrainerConfig.train_size, random_state=TrainerConfig.random_state, stratify=y)
                X_train[PreprocessConfig.target] = y_train # for compatibility code structure
                X_test[PreprocessConfig.target] = y_test  
                X_train.to_parquet(self.preprocessing_data.train_path, index=False)
                X_test.to_parquet(self.preprocessing_data.test_path, index=False)
            else:
                LOGGER.info("pandas.DataFrame.sample will be called.")
                train_df = preprocessed_df.sample(
                    frac=TrainerConfig.train_size, random_state=TrainerConfig.random_state
                )
                test_df = preprocessed_df.drop(train_df.index)
                train_df.to_parquet(self.preprocessing_data.train_path, index=False)
                test_df.to_parquet(self.preprocessing_data.test_path, index=False)

        if self.inference_mode:
            preprocessed_df.to_parquet(self.preprocessing_data.batch_path, index=False)

    @staticmethod
    def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
        """Preprocessing."""
        # drop unused columns based on dropped_col_names in config.py
        df.drop(labels=PreprocessConfig.dropped_col_names, axis=1, inplace=True, errors="ignore") # If ‘ignore’, suppress error and only existing labels are dropped.
        return df

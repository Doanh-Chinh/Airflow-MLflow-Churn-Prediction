import logging
from typing import Tuple, Optional
import joblib
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
import pandas as pd
import numpy as np

from utils.data_classes import FeaturesEncoder, FeaturesEngineeringData
from steps.config import FeatureEngineeringConfig


LOGGER = logging.getLogger(__name__)


class FeatureEngineeringStep:
    """Feature engineering: transform features for model training and inference.
    
    Args:
        inference_mode (bool): Whether the step is used in the training or inference pipeline. 
        feature_engineering_data (FeaturesEngineeringData): Paths relative to the FeatureEngineeringStep
    """

    def __init__(
        self, 
        inference_mode: bool, 
        feature_engineering_data: FeaturesEngineeringData
    ) -> None:
        self.inference_mode = inference_mode
        self.feature_engineering_data = feature_engineering_data

    def __call__(
        self,
        train_path: Optional[Path] = None,
        test_path: Optional[Path] = None,
        batch_path: Optional[Path] = None,
    ) -> None:
        """
        Input data paths depending on whether it's training (train, test) or inference (batch)

        Args:
            train_path (Optional[Path], optional): Input train path. Defaults to None.
            test_path (Optional[Path], optional): Input test path. Defaults to None.
            batch_path (Optional[Path], optional): input batch path. Defaults to None.
        """
        if not self.inference_mode:
            train_df = pd.read_parquet(train_path)
            test_df = pd.read_parquet(test_path)
            self.fit_transform(
                df=train_df, 
                output_path=self.feature_engineering_data.train_path
            )
            self.transform(
                df=test_df,
                output_path=self.feature_engineering_data.test_path
            )

        if self.inference_mode:
            batch_df = pd.read_parquet(batch_path)
            self.transform(
                batch_df, 
                output_path=self.feature_engineering_data.batch_path
            )

    def fit_transform(
            self, 
            df: pd.DataFrame, 
            output_path: Path
        ) -> None:
        """Fit encoders on data and store the encoder into the features store
        The processed data is then stored.

        Args:
            df (pd.DataFrame): Data to train encoders and to transform.
            output_path (Path): Data path after encoding.
        """
        LOGGER.info("Start features engineering 'fit_transform'.")
        features_encoder = self._init_features_encoder()
        base_df, ordinal_df, onehot_df, target_col = self._get_dfs(
            df=df, 
            features_encoder=features_encoder
        )

        ### Fit_transform Encoder
        ordinal_encoded_data = features_encoder.ordinal_encoder.fit_transform(ordinal_df)
        onehot_encoded_data = features_encoder.onehot_encoder.fit_transform(onehot_df).toarray()
        label_encoded_data = features_encoder.lb_encoder.fit_transform(target_col)
        LOGGER.info(f"label_encoded_data: {label_encoded_data.shape}")
        label_encoded_data = self.convert_minority_class_to_positive_class(y=label_encoded_data) # "Attrited Customer" is minority class labeled as 0 by LabelEncoder, we care this class so convert it to 1 (positive class)

        LOGGER.info(f"Ordinal categories: {features_encoder.ordinal_encoder.categories_}")
        LOGGER.info(f"Onehot categories: {features_encoder.onehot_encoder.categories_}")
        LOGGER.info(f"Label classes: {features_encoder.lb_encoder.classes_}")
        LOGGER.info(f"onehot_encoded_data: {onehot_encoded_data}")
        LOGGER.info(f"created_onehot_features: {features_encoder.created_onehot_features}")

        ### Assign to base_df
        base_df[features_encoder.ordinal_features] = ordinal_encoded_data
        # Process onehot columns
        base_df[features_encoder.created_onehot_features] = onehot_encoded_data
        # Don't forget to add the target
        base_df[features_encoder.target] = label_encoded_data

        ### Fit_resample Resampler (for class imbalance)
        LOGGER.info(f"base_ordinal_onehot_columns: {features_encoder.base_ordinal_onehot_columns}")
        X_train_enc = base_df[features_encoder.base_ordinal_onehot_columns].copy()  # all features input
        y_train_enc = base_df[features_encoder.target].copy() # target variable
        X_res_enc, y_res_enc = features_encoder.resampler.fit_resample(X_train_enc, y_train_enc) # fit resample on train set only
        X_y_res_enc = pd.concat([X_res_enc, y_res_enc], axis=1)
        LOGGER.info(f'Resampled dataset shape {Counter(y_res_enc)}')
        LOGGER.info(f"X_res_enc.shape {X_res_enc.shape}")
        # Initial new base_df with resample data
        columns = base_df.columns
        LOGGER.info(f"base_df.columns {columns}")
        base_df = pd.DataFrame(data=X_y_res_enc, columns=columns)
        LOGGER.info(f"base_df.shape 1 {base_df.shape}")


        LOGGER.info("Complete resample step.")
        ### Fit_transform Scaler
        scaled_base_ordinal_onehot_data = features_encoder.scaler.fit_transform(base_df[features_encoder.base_ordinal_onehot_columns]) # scale on features input only
        base_df[features_encoder.base_ordinal_onehot_columns] = scaled_base_ordinal_onehot_data

        ### Save base_df
        LOGGER.info(f"base_df.shape 2 {base_df.shape}")
        base_df.to_parquet(path=output_path)
        features_encoder.to_joblib(path=self.feature_engineering_data.encoders_path)
        LOGGER.info(
            f"Features and encoders successfully saved respectively to {str(output_path)} and {str(self.feature_engineering_data.encoders_path)}"
        )

    def transform(
            self, 
            df: pd.DataFrame, 
            output_path: Path
        ) -> None:
        """Transform data based on trained encoders.

        Args:
            df (pd.DataFrame): Data to transform.
            output_path (Path): Transformed data path.
        """
        LOGGER.info("Start features engineering 'transform'.")
        features_encoder = self._load_features_encoder()
        base_df, ordinal_df, onehot_df, target_col = self._get_dfs(
            df, features_encoder=features_encoder
        )
        # Apply transform by the pre-fitted features_encoder
        ### Transform Encoder
        ordinal_encoded_data = features_encoder.ordinal_encoder.transform(ordinal_df)
        onehot_encoded_data = features_encoder.onehot_encoder.transform(onehot_df).toarray()        

        ### Assign to base_df
        base_df[features_encoder.ordinal_features] = ordinal_encoded_data
        # Process onehot columns
        base_df[features_encoder.created_onehot_features] = onehot_encoded_data
        
        # In test set and batch set, we don't apply resample, keep original shape after Preprocess Step

        ### Transform Scaler
        scaled_base_ordinal_onehot_data = features_encoder.scaler.transform(base_df[features_encoder.base_ordinal_onehot_columns])
        base_df[features_encoder.base_ordinal_onehot_columns] = scaled_base_ordinal_onehot_data

        if target_col is not None:
            # Training
            label_encoded_data = features_encoder.lb_encoder.transform(target_col)
            label_encoded_data = self.convert_minority_class_to_positive_class(y=label_encoded_data)
            base_df[features_encoder.target] = label_encoded_data

        base_df.to_parquet(path=output_path)
        LOGGER.info(f"Features successfully saved to {str(output_path)}")

    def _init_features_encoder(self) -> FeaturesEncoder:
        """Init encoders for fit_transform()

        Return:
            features_encoder (FeaturesEncoder): Encoders artifact
        """
        # Need more process here
        # Resampler
        resampler = SMOTE(random_state=42)
        # Encoder
        ordinal_encoder = OrdinalEncoder(
            categories=FeatureEngineeringConfig.create_expected_ordinal_cates_order(),
            handle_unknown="use_encoded_value", 
            unknown_value=-1 
        ) # if unknown values occur (not in predefined categories), they are replaced by -1
        onehot_encoder = OneHotEncoder(
            categories=FeatureEngineeringConfig.create_expected_onehot_categories_order(),
            handle_unknown="ignore"
        ) # if unknown values occur (not in predefined categories), they are ignore
        lb_encoder = LabelEncoder()
        # Scaler
        scaler = StandardScaler()     
        # Onehot columns generated OneHotEncoder
        created_onehot_features = np.concatenate(FeatureEngineeringConfig.create_expected_onehot_names_order()).tolist()
        # X columns excluding y (target variable)
        base_ordinal_onehot_columns = FeatureEngineeringConfig.base_features + FeatureEngineeringConfig.ordinal_features + created_onehot_features
        return FeaturesEncoder(
            resampler = resampler,
            ordinal_encoder=ordinal_encoder,
            onehot_encoder=onehot_encoder,
            lb_encoder=lb_encoder,
            scaler=scaler,
            base_features=FeatureEngineeringConfig.base_features,
            ordinal_features=FeatureEngineeringConfig.ordinal_features,
            onehot_features=FeatureEngineeringConfig.onehot_features,
            created_onehot_features=created_onehot_features,
            base_ordinal_onehot_columns=base_ordinal_onehot_columns,
            target=FeatureEngineeringConfig.target,

        )

    def _load_features_encoder(self) -> FeaturesEncoder:
        """Load encoders artifact

        Returns:
            FeaturesEncoder: Encoders artifact
        """
        features_encoder = joblib.load(self.feature_engineering_data.encoders_path)
        return features_encoder

    def _get_dfs(
        self, 
        df: pd.DataFrame, 
        features_encoder: FeaturesEncoder
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
        """Extract the relevant columns based on features for respectively: 
        no transformation - ordinal categories - onehot categories"""
        base_df = df[features_encoder.base_features].copy()
        ordinal_df = df[features_encoder.ordinal_features].copy()
        onehot_df = df[features_encoder.onehot_features].copy()
        if not self.inference_mode:
            target_col = df[features_encoder.target].copy()
            return base_df, ordinal_df, onehot_df, target_col
        elif self.inference_mode:
            return base_df, ordinal_df, onehot_df, None

    def convert_minority_class_to_positive_class (self, y: np.ndarray) -> np.ndarray:
        "Covert minority class to 1 (positive class), if need."
        # Since metrits such as f1-score, precision, recall in metrics package extracted from prediction of positive class
        # and we care much about how the models performing with minority class which is "Attrition" class in this case
        if not isinstance(y, np.ndarray):
            raise ValueError("`y` must be a target variable after encoded by LabelEncoder!")

        # Count the number of each class from the target variable
        class_counter = Counter(y)
        LOGGER.info(f"Original class counter: {class_counter}")
        # Label 0 for majority, 1 for minority class
        number_of_class_0 = class_counter[0] # class O
        number_of_class_1 = class_counter[1] # class 1
        if number_of_class_0 < number_of_class_1:
            y = np.where(y == 0, 1, 0) # if the current value is 0 then replace by 1, otherwise replace by 0
            # Check result counter again
            LOGGER.info(f"Check result counter again...")
            class_counter = Counter(y)
            LOGGER.info(f"Final class counter: {class_counter}")
        else:
            LOGGER.info("Minority class is already label as 1 (positive class)")
        return y
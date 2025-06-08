# data/preprocessors.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
import logging
from typing import List, Optional, Self

logger = logging.getLogger(__name__)

class TabnetPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            numerical_cols : Optional[List[str]] = None,
            categorical_cols : Optional[List[str]] = None,
            ordinal_unknown_value: int = -1
                ):
        """
        Custom preprocessor for tabular data.
        Requires explicit specification of numerical and categorical columns.
        Handles categorical NaNs by filling, then encodes categoricals and scales numericals.
        """
        self.numerical_cols = numerical_cols if numerical_cols is not None else []
        self.categorical_cols = categorical_cols if categorical_cols is not None else []
        self.ordinal_unknown_value = ordinal_unknown_value

        # Transformers to be fitted
        self.ordinal_encoder_ = None
        self.numerical_scaler_ = None


        # Information learned during fit
        self.fitted_numerical_cols_ = []
        self.fitted_categorical_cols_ = []
        self.final_feature_names_ = []  
        self.cat_dims_ = [] #needed (maybe, see later) for tabnet, represents number of unique category for each encoded feature
        #for example -> encoded fature: 'city', unique categories: ['Rome', 'Milan'], cat_dims_[0] = 2
        self.cat_idxs_ = [] #for embedding in tabnet, represents the columns that contains categorical features 
        #we need to know indexes (position of the columns) of categorical features.

    def fit(self, X: pd.DataFrame, y:Optional[pd.Series]=None, X_val: Optional[pd.DataFrame]=None) -> Self:
        """
        Fit the preprocessor on the training features X.
        y is ignored (required for Scikit-learn compatibility).
        """
        logger.info("Starting fitting of TabularPreprocessor...")
        X_fit = X.copy()

        # 1. Identify actual columns present and of correct type from those specified
        self.fitted_numerical_cols_ = [
            col for col in self.numerical_cols
            if col in X_fit.columns and pd.api.types.is_numeric_dtype(X_fit[col])
        ]

        self.fitted_categorical_cols_ = [
            col for col in self.categorical_cols
            if col in X_fit.columns
        ]
    
        # 2. Categorical Column Preprocessing
        if self.fitted_categorical_cols_:
            if X_val is not None:
                logger.info("Considering validation data to build a complete categorical vocabulary.")
                # Sometimes a category is found in X_val but not found in X_train
                combined_categorical_data = pd.concat(
                    [X_fit[self.fitted_categorical_cols_], X_val[self.fitted_categorical_cols_]],
                    axis=0
                ).astype(str)  #Converted to string for safety
            else:
                combined_categorical_data = X_fit[self.fitted_categorical_cols_].astype(str)
            
            self.ordinal_encoder_ = OrdinalEncoder(
                handle_unknown='error',
                dtype=int
            )
            self.ordinal_encoder_.fit(combined_categorical_data)
            logger.info(f"Fitted OrdinalEncoder for: {self.fitted_categorical_cols_}")
            self.cat_dims_ = [len(cats) for cats in self.ordinal_encoder_.categories_]
        else:
            logger.info("No valid categorical columns found or specified for encoding.")
            self.cat_dims_ = []

        # 3. Numerical Column Preprocessing
        if self.fitted_numerical_cols_:
            # This class assumes numerical NaNs/Infs have been handled beforehand.
            # A warning is issued if they are still present.
            if X_fit[self.fitted_numerical_cols_].isnull().any().any():
                logger.warning(f"NaN values found in numerical columns {self.fitted_numerical_cols_} during scaler fit. "
                               f"It's recommended to handle these prior to preprocessing.")
            
            self.numerical_scaler_ = StandardScaler()
            self.numerical_scaler_.fit(X_fit[self.fitted_numerical_cols_])
            logger.info(f"Fitted StandardScaler for: {self.fitted_numerical_cols_}")
        else:
            logger.info("No valid numerical columns found or specified for scaling.")

        # 4. Determine final feature names and categorical indices for TabNet
        # the order will be numerical first, then categorical.
       
        self.final_feature_names_ = self.fitted_numerical_cols_ + self.fitted_categorical_cols_

        if self.fitted_categorical_cols_:
            num_features_count = len(self.fitted_numerical_cols_)
            self.cat_idxs_ = [
                i + num_features_count #indxs start from last numerical feature column
                for i in range(len(self.fitted_categorical_cols_))
            ]

        logger.info(f"Final feature names determined: {self.final_feature_names_}")
        logger.info(f"Categorical indices determined (cat_idxs_): {self.cat_idxs_}")
        logger.info("Fitting of TabularPreprocessor complete.")
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned transformations to the data X.
        """
        if not self.final_feature_names_: # basic check if fit was called
            logger.error("Preprocessor has not been fitted. Call fit() before transform().")
            raise RuntimeError("Preprocessor has not been fitted. Call fit() before transform().")

        logger.info("Starting data transformation with TabularPreprocessor...")
        X_copy = X.copy()
        X_transformed = pd.DataFrame(index=X_copy.index)

        # Numerical Transformations
        if self.numerical_scaler_ and self.fitted_numerical_cols_:
            cols_to_transform = [col for col in self.fitted_numerical_cols_ if col in X_copy.columns]
            if cols_to_transform:
                X_transformed[cols_to_transform] = self.numerical_scaler_.transform(X_copy[cols_to_transform])


        # Categorical Transformations
        if self.ordinal_encoder_ and self.fitted_categorical_cols_:
            cols_to_transform = [col for col in self.fitted_categorical_cols_ if col in X_copy.columns]
            if cols_to_transform:
                data_to_encode = X_copy[cols_to_transform].astype(str)
                X_transformed[cols_to_transform] = self.ordinal_encoder_.transform(data_to_encode)

        final_df = X_transformed.reindex(columns=self.final_feature_names_)

        missing_cols_in_input = set(self.final_feature_names_) - set(X.columns)
        if missing_cols_in_input:
            logger.warning(f"Input data was missing columns: {missing_cols_in_input}. They will be NaNs in the output.")

        logger.info("Data transformation complete.")
        return final_df
    

    def get_tabnet_params(self):
            """
            Returns the cat_dims and cat_idxs parameters needed for TabNet,
            which were calculated and stored during the fit method.
            """
            if not hasattr(self, 'cat_dims_') or not hasattr(self, 'cat_idxs_'):
                logger.error("The preprocessor has not been fitted. Call fit() before get_tabnet_params().")
                raise RuntimeError("The preprocessor has not been fitted. Call fit() before get_tabnet_params().")

            logger.info("Retrieving cat_dims and cat_idxs from the fitted preprocessor.")
            
            return self.cat_dims_, self.cat_idxs_

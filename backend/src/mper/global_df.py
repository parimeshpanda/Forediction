import pandas as pd
from typing import Optional


class GlobalDataframe:
    # Class-level attributes for storing different DataFrames
    original_df: pd.DataFrame | None = None
    drop_duplicates_df: pd.DataFrame | None = None  ## drop_duplicates
    typecasted_df: pd.DataFrame | None = None  ## type_cast_columns
    grouped_dates_df: pd.DataFrame | None = None  ## grouping_same_dates
    missing_values_detection_df: pd.DataFrame | None = None  ## missing_values_treatment
    missing_values_treated_df: pd.DataFrame | None = None  ## missing_values_treatment
    train_df: pd.DataFrame | None = None  ## train_test_validation_split_df
    test_df: pd.DataFrame | None = None  ## train_test_validation_split_df
    validation_df: pd.DataFrame | None = None  ## train_test_validation_split_df
    outlier_treated_df: pd.DataFrame | None = None  ## outlier_detection_bound
    adf_df: pd.DataFrame | None = None  ## stationarity_ADF
    kpss_df: pd.DataFrame | None = None  ## stationarity_KPSS
    forecasted_outlier_df: pd.DataFrame | None = None  ## outlier_treatment_bound_forecast
    final_forecast_df: pd.DataFrame | None = None  ## forecast_time_series_steps
    current_df: pd.DataFrame | None = None

    # Singleton instance
    _instance: Optional['GlobalDataframe'] = None

    def __new__(cls, *args, **kwargs):
        # Check if an instance already exists
        if cls._instance is None:
            # If no instance exists, create one
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance


global_df1 = GlobalDataframe()
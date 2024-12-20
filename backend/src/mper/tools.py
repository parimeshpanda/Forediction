from langchain.tools import tool
import time
import pandas as pd
import constants
from scipy.stats import shapiro
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import kpss
from sklearn.impute import KNNImputer
import sys
import io
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import base64
from scipy.fft import fft, ifft, fftfreq
from langchain_core.messages import HumanMessage
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import traceback
import warnings
# from fbprophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from itertools import product
from sklearn.model_selection import GridSearchCV
from io import BytesIO

from src.mper.agent_template import AgentTemplate
from src.mper.global_df import global_df1

global_data: pd.DataFrame | None = None


@tool
def data_insights(csv_path):
    """
    Describe CSV Data for a forecasting task. Used to describe the Dataset, it takes path to the csv file as an input. This input is a string.
    """
    # try:
    #     data = pd.read_csv(csv_path)
    #     data.to_csv(constants.INTERMEDIATE_DF_PATH + "intermediate.csv")
    # except:
    #     return "Invalid Path"
    try:
        data = pd.read_csv(csv_path)
        os.makedirs(constants.INTERMEDIATE_DF_PATH, exist_ok=True)
        os.makedirs(constants.GRAPH_FLAG_PATH, exist_ok=True)
        data.to_csv(constants.INTERMEDIATE_DF_PATH + "intermediate.csv", index=False)
        global_df1.original_df = data
        global_df1.current_df = data
    except FileNotFoundError:
        return "Invalid Path"
    except pd.errors.EmptyDataError:
        return "The file is empty"
    except Exception as e:
        return f"An error occurred: {e}"

    max_retries = constants.MAX_RETRIES
    for attempt in range(max_retries):
        try:

            info = {}
            info["original_data_shape"] = data.shape
            info["non_null"] = data.count()
            info["unique_values"] = data.apply(lambda srs: len(srs.unique()))
            info["row_duplicates"] = data.duplicated().sum()
            info["dtypes"] = str(data.dtypes)
            data = data.drop_duplicates()
            info["data_shape_after_dropping_duplicates"] = data.shape

            # print("Data Insights df: ", data)
            data.to_csv(constants.INTERMEDIATE_DF_PATH + "intermediate.csv", index=False)
            global_df1.duplicate_dropped_df = data
            global_df1.current_df = data
            return str(info)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(1)  # Wait a bit before retrying
            else:
                print("All attempts failed.")
                return str(e)



@tool
def time_interval_missing(date_column, ts_id_column, test_size: float):
    """
    Extracts key information about time interval from a time series DataFrame and identifies missing points by generating
    the full range of possible dates (based on inferred frequency) and finding the missing dates.

    Parameters:
    - date_column (str): The name of the column that contains the datetime values.
    - ts_id_column (str): The name of the column that contains the time series ID values.
    - test_size (int): The proportion of the data to include in the testing set. The test size should be equal to the forecast horizon given by the user provided there are sufficient data points. Forecast horizon should be less then
    """
    max_retries = constants.MAX_RETRIES
    for attempt in range(max_retries):
        try:
            try:
                # df = pd.read_csv(constants.INTERMEDIATE_DF_PATH + "intermediate.csv")
                if global_df1.grouped_dates_df is None:
                    df = global_df1.current_df
                else:
                    df = global_df1.grouped_dates_df
            except Exception as e:
                print("error",e)

            if (7 + (2 * test_size)) > len(df):
                sys.exit("Not having sufficient Data points to forecast")

            df[date_column] = pd.to_datetime(df[date_column])

            df = df.sort_values(by=date_column)

            start_date = df[date_column].min()
            end_date = df[date_column].max()

            time_diff = df[date_column].diff().dropna().mode()[0]

            # Determine frequency based on the time difference between the dates
            if time_diff <= pd.Timedelta(days=1):
                frequency = "Daily"
                expected_date_range = pd.date_range(start=start_date, end=end_date, freq="D")
            elif time_diff <= pd.Timedelta(days=7):
                frequency = "Weekly"
                most_frequent_weekday = df[date_column].dt.weekday.mode()[
                    0
                ]  # Monday=0, Sunday=6
                expected_date_range = pd.date_range(
                    start=start_date,
                    end=end_date,
                    freq=f'W-{["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"][most_frequent_weekday]}',
                )
            elif time_diff <= pd.Timedelta(days=14):  # Check for biweekly (14 days)
                frequency = "Biweekly"
                expected_date_range = pd.date_range(start=start_date, end=end_date, freq="14D")
            elif (
                time_diff <= pd.Timedelta(days=30)
                or time_diff <= pd.Timedelta(days=31)
                or time_diff <= pd.Timedelta(days=28)
                or time_diff <= pd.Timedelta(days=29)
            ):
                frequency = "Monthly"
                if start_date.day == 1:
                    expected_date_range = pd.date_range(
                        start=start_date, end=end_date, freq="MS"
                    )
                else:
                    expected_date_range = pd.date_range(
                        start=start_date, end=end_date, freq="ME"
                    )
            elif time_diff <= pd.Timedelta(days=365):
                frequency = "Yearly"
                expected_date_range = pd.date_range(start=start_date, end=end_date, freq="YS")
            elif time_diff <= pd.Timedelta(days=366):
                frequency = "Yearly"
                expected_date_range = pd.date_range(start=start_date, end=end_date, freq="A")
            else:
                frequency = "Unknown"
                expected_date_range = pd.DatetimeIndex([])

            # Remove duplicates and find the missing dates
            actual_dates = df[date_column].drop_duplicates()
            missing_dates = expected_date_range.difference(actual_dates)

            df.set_index(date_column, inplace=True)
            df = df.reindex(expected_date_range)
            df.columns.name = date_column
            # print(df.columns.name)
            # df[ts_id_column] = df[ts_id_column].fillna(method="ffill", inplace=True)
            df[ts_id_column] = df[ts_id_column].ffill()
            df.reset_index(inplace=True)
            # print("Time Interval df: ", df)
            # print(df.columns)
            df.rename(columns={df.columns[0]: date_column}, inplace=True)

            global_df1.missing_values_detection_df = df
            global_df1.current_df = df
            # df.to_csv(constants.INTERMEDIATE_DF_PATH + "intermediate.csv", index=False)

            return str(
                {
                    "Start Date": start_date,
                    "End Date": end_date,
                    "Frequency": frequency,
                    "Number of Data Points": len(df),
                    "Missing Points": len(missing_dates),
                    # "Missing Dates": missing_dates.tolist()
                }
            )

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(1)  # Wait a bit before retrying
            else:
                print("All attempts failed.")
                return str(e)



@tool
def train_test_validation_split_df(date_column:str, column_name: str, test_size: int):
    """
    Splits the DataFrame into training and testing sets for time series forecasting.

    Parameters:
    - date_column (str): The name of the column that contains the datetime values.
    - column_name (str): The name of the column that contains the target values to forecast.
    - test_size (int): The number of data points for test and validation set. The test size should be equal to the forecast horizon given by the user (provided there are sufficient data points). Forecast horizon should be less than the total number of data points.
    """

    # df = pd.read_csv(constants.INTERMEDIATE_DF_PATH + "intermediate.csv")
    # df[column_name] = pd.to_numeric(df[column_name], errors='coerce')

    # # train_size = int(0.8 * len(df))

    # train_size = len(df) - (test_size) * 2
    # train_df = df.iloc[:train_size]
    # temp_df = df.iloc[train_size:]
    # test_df = temp_df[:len(temp_df)//2]
    # validation_df = temp_df[len(temp_df)//2:]
    # train_df.to_csv(constants.INTERMEDIATE_DF_PATH + "train.csv")
    # test_df.to_csv(constants.INTERMEDIATE_DF_PATH + "test.csv")
    # validation_df.to_csv(constants.INTERMEDIATE_DF_PATH + "validation.csv")
    max_retries = constants.MAX_RETRIES
    for attempt in range(max_retries):
        try:

            intermediate_path = f"{constants.INTERMEDIATE_DF_PATH}intermediate.csv"
            train_path = f"{constants.INTERMEDIATE_DF_PATH}train.csv"
            test_path = f"{constants.INTERMEDIATE_DF_PATH}test.csv"
            validation_path = f"{constants.INTERMEDIATE_DF_PATH}validation.csv"

            try:
                # df = pd.read_csv(intermediate_path)
                if global_df1.missing_values_treated_df is None:
                    df = global_df1.current_df
                else:
                    df = global_df1.missing_values_treated_df
            except Exception as e:
                print("error", e)
            # Convert column to numeric, coercing errors
            df[column_name] = pd.to_numeric(df[column_name], errors="coerce")

            # Calculate indices for splitting
            train_end = len(df) - 2 * test_size
            test_end = train_end + test_size

            # Split data
            train_df = df.iloc[:train_end]
            test_df = df.iloc[train_end:test_end]
            validation_df = df.iloc[test_end:]

            # Save datasets to CSV files
            # train_df.to_csv(train_path)
            # test_df.to_csv(test_path)
            # validation_df.to_csv(validation_path)


            # fig = go.Figure()
            # fig.add_trace(go.Scatter(x=train_df[date_column], y=train_df[column_name], mode='lines', name='Training Data', line=dict(color='blue')))
            # fig.add_trace(go.Scatter(x=test_df[date_column], y=test_df[column_name], mode='lines', name='Testing Data', line=dict(color='red')))
            # fig.add_trace(go.Scatter(x=validation_df[date_column], y=validation_df[column_name], mode='lines', name='Validation Data', line=dict(color='green')))
            # fig.update_layout(
            #     title="Time Series Forecasting Split",
            #     xaxis_title=date_column,
            #     yaxis_title=column_name,
            #     legend_title="Legend",
            # )
            # fig_json = json.loads(fig.to_json())

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=train_df[date_column],
                y=train_df[column_name],
                mode='lines',
                name='Training Data',
                line=dict(color='blue')
            ))

            fig.add_trace(go.Scatter(
                x=test_df[date_column],
                y=test_df[column_name],
                mode='lines',
                name='Testing Data',
                line=dict(color='red')
            ))

            fig.add_trace(go.Scatter(
                x=validation_df[date_column],
                y=validation_df[column_name],
                mode='lines',
                name='Validation Data',
                line=dict(color='green')
            ))

            fig.update_layout(
                title="Time/Test/Validation Split",
                xaxis_title=date_column,
                yaxis_title=column_name,
                legend_title="Legend",
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot area
                font=dict(color='white'),  # White font color for dark theme
                xaxis=dict(
                    gridcolor='gray',  # Grid color for better visibility
                    linecolor='white',  # Axis line color
                    zerolinecolor='gray'  # Zero line color
                ),
                yaxis=dict(
                    gridcolor='gray',  # Grid color for better visibility
                    linecolor='white',  # Axis line color
                    zerolinecolor='gray'  # Zero line color
                ),
                legend=dict(
                    bgcolor='rgba(0,0,0,0)',  # Transparent legend background
                    bordercolor='white'  # Legend border color
                ),
                updatemenus=[
                    {
                        "buttons": [
                            {
                                "args": [{"xaxis.gridcolor": 'gray', "yaxis.gridcolor": 'gray'}],
                                "label": "Show Grid",
                                "method": "relayout"
                            },
                            {
                                "args": [{"xaxis.gridcolor": 'rgba(0,0,0,0)', "yaxis.gridcolor": 'rgba(0,0,0,0)'}],
                                "label": "Hide Grid",
                                "method": "relayout"
                            }
                        ],
                        "direction": "down",
                        "showactive": True,
                        "x": 0.1,
                        "xanchor": "left",
                        "y": 1.1,
                        "yanchor": "top"
                    }
                ]
            )

            fig_json = json.loads(fig.to_json())

            with open(constants.GRAPH_FLAG_PATH + "train_test_split.txt", "w") as f:
                f.write(json.dumps(fig_json))
                f.close()

            if not os.path.exists(constants.GRAPH_FLAG_PATH + "graph_flag.txt"):
                open(constants.GRAPH_FLAG_PATH + "graph_flag.txt", 'w').close()

            with open(constants.GRAPH_FLAG_PATH + "graph_flag.txt", "r+") as f:
                lines = f.readlines()
                if not lines:
                    lines.append("1\n")
                    lines.append("train_test_split.txt\n")
                else:
                    flag = True
                    while flag:
                        f.seek(0)
                        lines = f.readlines()
                        if int(lines[0]) == 1:
                            time.sleep(2)
                            continue
                        first_line = 1
                        lines[0] = f"{first_line}\n"
                        lines.append("train_test_split.txt\n")
                        flag = False

                f.seek(0)
                f.writelines(lines)
                f.close()

            global_df1.train_df = train_df
            global_df1.test_df = test_df
            global_df1.validation_df = validation_df
            global_df1.current_df = train_df

            # train_df.to_csv(train_path, index=False)
            # test_df.to_csv(test_path, index=False)
            # validation_df.to_csv(validation_path, index=False)

            return "DataFrame has been split into training, testing  and validation sets."

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(1)  # Wait a bit before retrying
            else:
                print("All attempts failed.")
                return str(e)




@tool
def grouping_same_dates(
    date_column: str, target_column: str, grouping_function: str, ts_id_column: str
) -> str:
    """
    Groups a data frame by a specified date column and time series identifier column, applying a specified aggregation
    function (either 'mean' or 'sum'. The specified aggregation depends on the use-case. Example: if the target value to forecast is temperature then mean would make more sense becuase adding up temperature does not make sense, but if the target value is sales then sum makes more sense) to the target column, and saves the result to a CSV file.

    Parameters:
    date_column : str
        The name of the column containing the date values to group by.
    target_column : str
        The name of the column on which the aggregation function will be applied.
    grouping_function : str
        The aggregation function to apply to the target column. Must be either 'mean' or 'sum'.
    ts_id_column : str
        The name of the time series identifier column, used to group data along with the date column.
    """
    max_retries = constants.MAX_RETRIES
    for attempt in range(max_retries):
        try:
            try:
                # df = pd.read_csv(constants.INTERMEDIATE_DF_PATH + "intermediate.csv")
                if global_df1.typecasted_df is None:
                    df = global_df1.current_df
                else:
                    df = global_df1.typecasted_df
            except Exception as e:
                print("error", e)
            ### OPTIMIZE: repeating code, please rewrite.
            if grouping_function.lower().strip() == "sum":
                grouped_df = df.groupby([date_column, ts_id_column])[target_column].sum()
                grouped_df = pd.DataFrame(grouped_df).reset_index()
                # print("Groupby df: ", grouped_df)
                # grouped_df.to_csv(
                #     constants.INTERMEDIATE_DF_PATH + "intermediate.csv", index=False
                # )
                global_df1.grouped_dates_df = grouped_df
                global_df1.current_df = grouped_df
                return (
                    "data frame was grouped by "
                    + str(date_column)
                    + ", with the "
                    + str(target_column)
                    + " column being summed. The shape of the grouped data frame is "
                    + str(grouped_df.shape)
                )

            elif grouping_function.lower().strip() == "mean":
                grouped_df = df.groupby([date_column, ts_id_column])[target_column].mean()
                grouped_df = pd.DataFrame(grouped_df).reset_index()
                # print("Groupby df: ", grouped_df)
                # grouped_df.to_csv(
                #     constants.INTERMEDIATE_DF_PATH + "intermediate.csv", index=False
                # )
                global_df1.grouped_dates_df = grouped_df
                global_df1.current_df = grouped_df
                return (
                    "data frame was grouped by "
                    + str(date_column)
                    + ", with the "
                    + str(target_column)
                    + " column being averaged. The shape of the grouped data frame is "
                    + str(grouped_df.shape)
                )
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(1)  # Wait a bit before retrying
            else:
                print("All attempts failed.")
                return str(e)


@tool
def normalisation_insights(column_name):
    """
    perform shapiro-wilk test on the datset and provide insights regarding the normality of the data, it takes the target column's name (column used for forecasting) for shapiro-wilk test as input.
    """
    # print(column_name)
    max_retries = constants.MAX_RETRIES
    for attempt in range(max_retries):
        try:
            csv_path = constants.INTERMEDIATE_DF_PATH + "train.csv"
            try:
                # data = pd.read_csv(csv_path)
                if global_df1.train_df is None:
                    data = global_df1.current_df
                else:
                    data = global_df1.train_df
            except Exception as e:
                print("error", e)

            noramlisation_info = shapiro(data[column_name])
            # print(noramlisation_info)
            info_string = f"\n shapiro-wilk test insights: {noramlisation_info}"

            return info_string
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(1)  # Wait a bit before retrying
            else:
                print("All attempts failed.")
                return str(e)



@tool
def type_cast_columns(date_column, target_column, ts_id_column):
    """
    Type-cast the columns in the DataFrame to the appropriate data types for time series analysis.
    Parameters:
    - date_column (str): The name of the column that contains the datetime values.
    - target_column (str): The name of the column that contains the target values to forecast.
    - ts_id_column (str): The name of the column that contains the time series ID values.
    Returns:
    - str: A message indicating that the columns have been type-casted successfully.
    """
    max_retries = constants.MAX_RETRIES
    for attempt in range(max_retries):
        try:
            try:
                # df = pd.read_csv(constants.INTERMEDIATE_DF_PATH + "intermediate.csv")
                if global_df1.drop_duplicates_df is None:
                    df = global_df1.current_df
                else:
                    df = global_df1.drop_duplicates_df
            except Exception as e:
                print("error", e)

            df[date_column] = pd.to_datetime(df[date_column])
            df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
            df[ts_id_column] = df[ts_id_column].astype(
                str
            )  ### NOTE: If this column does not exist, handle this...

            # fig = px.line(df, x=date_column, y=target_column)
            # fig.update_layout(
            #     title="Original Time Series",
            #     xaxis_title=date_column,
            #     yaxis_title=target_column,
            # )
            # fig_json = json.loads(fig.to_json())


            fig = px.line(df, x=date_column, y=target_column)

            fig.update_layout(
                title="Original Time Series",
                xaxis_title=date_column,
                yaxis_title=target_column,
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot area
                font=dict(color='white'),  # White font color for dark theme
                xaxis=dict(
                    gridcolor='gray',  # Grid color for better visibility
                    linecolor='white',  # Axis line color
                    zerolinecolor='gray'  # Zero line color
                ),
                yaxis=dict(
                    gridcolor='gray',  # Grid color for better visibility
                    linecolor='white',  # Axis line color
                    zerolinecolor='gray'  # Zero line color
                ),
                legend=dict(
                    bgcolor='rgba(0,0,0,0)',  # Transparent legend background
                    bordercolor='white'  # Legend border color
                ),
                updatemenus=[{
                    "buttons": [
                        {
                            "args": [{"xaxis.gridcolor": 'gray', "yaxis.gridcolor": 'gray'}],
                            "label": "Show Grid",
                            "method": "relayout"
                        },
                        {
                            "args": [{"xaxis.gridcolor": 'rgba(0,0,0,0)', "yaxis.gridcolor": 'rgba(0,0,0,0)'}],
                            "label": "Hide Grid",
                            "method": "relayout"
                        }
                    ],
                    "direction": "down",
                    "showactive": True,
                    "x": 1,
                    "xanchor": "right",
                    "y": 1.1,
                    "yanchor": "top"
                }]
            )
            fig_json = json.loads(fig.to_json())

            with open(constants.GRAPH_FLAG_PATH + "timeseries_original.txt", "w") as f:
                f.write(json.dumps(fig_json))
                f.close()

            if not os.path.exists(constants.GRAPH_FLAG_PATH + "graph_flag.txt"):
                open(constants.GRAPH_FLAG_PATH + "graph_flag.txt", 'w').close()

            with open(constants.GRAPH_FLAG_PATH + "graph_flag.txt", "r+") as f:
                lines = f.readlines()
                if not lines:
                    lines.append("1\n")
                    lines.append("timeseries_original.txt\n")
                else:
                    flag = True
                    while flag:
                        f.seek(0)
                        lines = f.readlines()
                        if int(lines[0]) == 1:
                            time.sleep(2)
                            continue
                        first_line = 1
                        lines[0] = f"{first_line}\n"
                        lines.append("timeseries_original.txt\n")
                        flag = False
                f.seek(0)
                f.writelines(lines)
                f.close()

            # df.to_csv(constants.INTERMEDIATE_DF_PATH + "intermediate.csv", index=False)
            global_df1.typecasted_df = df
            global_df1.current_df = df

            return "Columns have been type-casted successfully."

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(1)  # Wait a bit before retrying
            else:
                print("All attempts failed.")
                return str(e)


# @tool
# def outlier_detection_modified_z(column_name, threshold=3.5):
#     """
#     Detect outliers using the modified Z-score method for each numeric column in the DataFrame.
#     Apply this method if data is normally distributed.
#     Returns a DataFrame with only the rows containing outliers.

#     Parameters:
#     - it takes target column name (column used for forecasting) for Outlier Detection as input.
#     - threshold (float): The threshold for detecting outliers. Data points with a modified Z-score
#                          greater than this value will be considered outliers.

#     Returns:
#     - outliers_df (str): String dataFrame containing only the rows with outliers.
#     """

#     csv_path = constants.INTERMEDIATE_DF_PATH + "intermediate.csv"
#     df = pd.read_csv(csv_path)

#     data = df[column_name].dropna()

#     median = np.median(data)
#     mad = np.median(np.abs(data - median))

#     modified_z_score = 0.6745 * (data - median) / mad

#     outlier_indices = data.index[modified_z_score.abs() > threshold]

#     return str(outlier_indices)


# @tool
# def outlier_detection_iqr(column_name):
#     """
#     Detect outliers in the given DataFrame using the Interquartile Range (IQR) method.
#     Apply this method if data is not normally distributed.
#     The IQR method identifies outliers by calculating the first (Q1) and third (Q3) quartiles
#     of the data for each numeric column, and then calculating the IQR (Q3 - Q1). Any data point
#     that lies below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR is considered an outlier.

#     Parameters:
#     - it takes target column name (column used for forecasting) for Outlier Detection as input.

#     Returns:
#     - str: A string representation of the rows in the DataFrame that are outliers.
#       It returns all rows that contain outliers based on the IQR method for each numeric column.
#       If no outliers are found, it will return an empty DataFrame as a string.
#     """
#     outliers = pd.DataFrame()
#     csv_path = constants.INTERMEDIATE_DF_PATH + "intermediate.csv"
#     df = pd.read_csv(csv_path)
#     Q1 = df[column_name].quantile(0.25)
#     Q3 = df[column_name].quantile(0.75)

#     IQR = Q3 - Q1

#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     outliers = pd.concat(
#         [
#             outliers,
#             df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)],
#         ]
#     )

#     return str(outliers)


@tool
def outlier_treatment_bound(date_column, column_name, method="iqr", threshold=3):
    """
    Treat outliers using the specified outlier detection method: IQR or Z-score. The outlier is capped at a certain value above the 75th percentile value or floored at a factor below the 25th percentile value.

    Parameters:
    - date_column (str): The name of the column that contains the datetime values.
    - column_name (str): it takes target column name (column used for forecasting) for Outlier Treatment as input.
    - method (str): The method to use for outlier detection. Options are:
      - "iqr" : Use Interquartile Range (IQR) method.
      - "zscore" : Use Z-score method.
    - threshold (float): For the Z-score method, the threshold above which data points are considered outliers.
      Default is 3. For the IQR method, this parameter is ignored.
    """
    max_retries = constants.MAX_RETRIES
    for attempt in range(max_retries):
        try:

            csv_path = constants.INTERMEDIATE_DF_PATH + "train.csv"

            try:
                # df = pd.read_csv(csv_path)
                if global_df1.train_df is None:
                    df = global_df1.current_df
                else:
                    df = global_df1.train_df
            except Exception as e:
                print("error", e)

            modified_df = df
            temp_df = modified_df.copy(deep=True)

            if method == "iqr":
                Q1 = df[column_name].quantile(0.25)
                Q3 = df[column_name].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                modified_df[column_name] = df[column_name].clip(
                    lower=lower_bound, upper=upper_bound
                )

                # fig = go.Figure()

                # fig.add_trace(go.Scatter(x=temp_df[date_column], y=temp_df[column_name], mode='lines', name='Original Data', line=dict(color='gray', dash='dot')))
                # fig.add_trace(go.Scatter(x=modified_df[date_column], y=modified_df[column_name], mode='lines', name='Modified Data', line=dict(color='blue')))
                # fig.update_layout(
                #     title=f"Outlier Detection using {method}: Original vs Modified Data",
                #     xaxis_title=date_column,
                #     yaxis_title=column_name,
                #     legend_title="Legend",
                # )
                # fig_json = json.loads(fig.to_json())

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    # x=temp_df[date_column],
                    # y=temp_df[column_name],
                    x=global_df1.typecasted_df[date_column],
                    y=global_df1.typecasted_df[column_name],
                    mode='lines',
                    name='Original',
                    line=dict(color='grey')
                ))

                if global_df1.missing_values_treated_df is None:
                    missing_values_plot_df = global_df1.missing_values_detection_df
                else:
                    missing_values_plot_df = global_df1.missing_values_treated_df

                fig.add_trace(go.Scatter(
                    x=missing_values_plot_df[date_column],
                    y=missing_values_plot_df[column_name],
                    mode='lines',
                    name='Missing Value Treated',
                    line=dict(color='green')
                ))

                fig.add_trace(go.Scatter(
                    x=modified_df[date_column],
                    y=modified_df[column_name],
                    mode='lines',
                    name='Missing Value & Outlier Treated',
                    line=dict(color='blue')
                ))

                fig.update_layout(
                    title=f"Preprocessed - Adjusted Post Missing Value & Outlier Treatmend",
                    xaxis_title=date_column,
                    yaxis_title=column_name,
                    legend_title="Legend",
                    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                    plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot area
                    font=dict(color='white'),  # White font color for dark theme
                    xaxis=dict(
                        gridcolor='gray',  # Grid color for better visibility
                        linecolor='white',  # Axis line color
                        zerolinecolor='gray'  # Zero line color
                    ),
                    yaxis=dict(
                        gridcolor='gray',  # Grid color for better visibility
                        linecolor='white',  # Axis line color
                        zerolinecolor='gray'  # Zero line color
                    ),
                    legend=dict(
                        bgcolor='rgba(0,0,0,0)',  # Transparent legend background
                        bordercolor='white'  # Legend border color
                    ),
                    updatemenus=[
                        {
                            "buttons": [
                                {
                                    "args": [{"xaxis.gridcolor": 'gray', "yaxis.gridcolor": 'gray'}],
                                    "label": "Show Grid",
                                    "method": "relayout"
                                },
                                {
                                    "args": [{"xaxis.gridcolor": 'rgba(0,0,0,0)', "yaxis.gridcolor": 'rgba(0,0,0,0)'}],
                                    "label": "Hide Grid",
                                    "method": "relayout"
                                }
                            ],
                            "direction": "down",
                            "showactive": True,
                            "x": 1,
                            "xanchor": "right",
                            "y": 1.1,
                            "yanchor": "top"
                        }
                    ]
                )

                fig_json = json.loads(fig.to_json())

                with open(constants.GRAPH_FLAG_PATH + "outliers.txt", "w") as f:
                    f.write(json.dumps(fig_json))
                    f.close()

                if not os.path.exists(constants.GRAPH_FLAG_PATH + "graph_flag.txt"):
                    open(constants.GRAPH_FLAG_PATH + "graph_flag.txt", 'w').close()

                with open(constants.GRAPH_FLAG_PATH + "graph_flag.txt", "r+") as f:
                    lines = f.readlines()
                    if not lines:
                        lines.append("1\n")
                        lines.append("outliers.txt\n")
                    else:
                        flag = True
                        while flag:
                            f.seek(0)
                            lines = f.readlines()
                            if int(lines[0]) == 1:
                                time.sleep(2)
                                continue
                            first_line = 1
                            lines[0] = f"{first_line}\n"
                            lines.append("outliers.txt\n")
                            flag=False

                    f.seek(0)
                    f.writelines(lines)
                    f.close()

                global_df1.outlier_treated_df = modified_df
                global_df1.current_df = modified_df
                # modified_df.to_csv(constants.INTERMEDIATE_DF_PATH + "train.csv", index=False)
                return "Outlier treated using IQR method"

            elif method == "zscore":
                mean = df[column_name].mean()
                std = df[column_name].std()
                z_scores = (df[column_name] - mean) / std

                modified_df[column_name] = df[column_name].where(
                    np.abs(z_scores) <= threshold, mean
                )

                # fig = go.Figure()

                # fig.add_trace(go.Scatter(x=temp_df[date_column], y=temp_df[column_name], mode='lines', name='Original Data', line=dict(color='gray', dash='dot')))
                # fig.add_trace(go.Scatter(x=modified_df[date_column], y=modified_df[column_name], mode='lines', name='Modified Data', line=dict(color='blue')))
                # fig.update_layout(
                #     title=f"Outlier Detection using {method}: Original vs Modified Data",
                #     xaxis_title=date_column,
                #     yaxis_title=column_name,
                #     legend_title="Legend",
                # )
                # fig_json = json.loads(fig.to_json())


                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    # x=temp_df[date_column],
                    # y=temp_df[column_name],
                    x=global_df1.typecasted_df[date_column],
                    y=global_df1.typecasted_df[column_name],
                    mode='lines',
                    name='Original',
                    line=dict(color='grey')
                ))

                if global_df1.missing_values_treated_df is None:
                    missing_values_plot_df = global_df1.missing_values_detection_df
                else:
                    missing_values_plot_df = global_df1.missing_values_treated_df

                fig.add_trace(go.Scatter(
                    x=missing_values_plot_df[date_column],
                    y=missing_values_plot_df[column_name],
                    mode='lines',
                    name='Missing Value Treated',
                    line=dict(color='green')
                ))

                fig.add_trace(go.Scatter(
                    x=modified_df[date_column],
                    y=modified_df[column_name],
                    mode='lines',
                    name='Missing Value & Outlier Treated',
                    line=dict(color='blue')
                ))

                fig.update_layout(
                    title=f"Preprocessed - Adjusted Post Missing Value & Outlier Treatmend",
                    xaxis_title=date_column,
                    yaxis_title=column_name,
                    legend_title="Legend",
                    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                    plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot area
                    font=dict(color='white'),  # White font color for dark theme
                    xaxis=dict(
                        gridcolor='gray',  # Grid color for better visibility
                        linecolor='white',  # Axis line color
                        zerolinecolor='gray'  # Zero line color
                    ),
                    yaxis=dict(
                        gridcolor='gray',  # Grid color for better visibility
                        linecolor='white',  # Axis line color
                        zerolinecolor='gray'  # Zero line color
                    ),
                    legend=dict(
                        bgcolor='rgba(0,0,0,0)',  # Transparent legend background
                        bordercolor='white'  # Legend border color
                    ),
                    updatemenus=[
                        {
                            "buttons": [
                                {
                                    "args": [{"xaxis.gridcolor": 'gray', "yaxis.gridcolor": 'gray'}],
                                    "label": "Show Grid",
                                    "method": "relayout"
                                },
                                {
                                    "args": [{"xaxis.gridcolor": 'rgba(0,0,0,0)', "yaxis.gridcolor": 'rgba(0,0,0,0)'}],
                                    "label": "Hide Grid",
                                    "method": "relayout"
                                }
                            ],
                            "direction": "down",
                            "showactive": True,
                            "x": 1,
                            "xanchor": "right",
                            "y": 1.1,
                            "yanchor": "top"
                        }
                    ]
                )

                fig_json = json.loads(fig.to_json())


                with open(constants.GRAPH_FLAG_PATH + "outliers.txt", "w") as f:
                    f.write(json.dumps(fig_json))
                    f.close()

                if not os.path.exists(constants.GRAPH_FLAG_PATH + "graph_flag.txt"):
                    open(constants.GRAPH_FLAG_PATH + "graph_flag.txt", 'w').close()

                with open(constants.GRAPH_FLAG_PATH + "graph_flag.txt", "r+") as f:
                    lines = f.readlines()
                    if not lines:
                        lines.append("1\n")
                        lines.append("outliers.txt\n")
                    else:
                        flag = True
                        while flag:
                            f.seek(0)
                            lines = f.readlines()
                            if int(lines[0]) == 1:
                                time.sleep(2)
                                continue
                            first_line = 1
                            lines[0] = f"{first_line}\n"
                            lines.append("outliers.txt\n")
                            flag=False

                    f.seek(0)
                    f.writelines(lines)
                    f.close()

                global_df1.outlier_treated_df = modified_df
                global_df1.current_df = modified_df
                # modified_df.to_csv(constants.INTERMEDIATE_DF_PATH + "train.csv", index=False)
                return "Outlier treated using Z-score method"

            else:
                raise ValueError("Method must be either 'iqr' or 'zscore'")

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(1)  # Wait a bit before retrying
            else:
                print("All attempts failed.")
                return str(e)



@tool
def outlier_treatment_mean(date_column, column_name, method="iqr", threshold=3):
    """
    Treat outliers using the specified outlier detection method (IQR or Z-score), and replace the outliers with the mean of the respective column/series.

    Parameters:
    - date_column (str): The name of the column that contains the datetime values.
    - column_name (str): it takes target column name (column used for forecasting) for Outlier Treatment as input.
    - method (str): The method to use for outlier detection. Options are:
      - "iqr" : Use Interquartile Range (IQR) method.
      - "zscore" : Use Z-score method.
    - threshold (float): For the Z-score method, the threshold above which data points are considered outliers.
      Default is 3. For the IQR method, this parameter is ignored.

    """
    max_retries = constants.MAX_RETRIES
    for attempt in range(max_retries):
        try:
            csv_path = constants.INTERMEDIATE_DF_PATH + "train.csv"

            try:
                # df = pd.read_csv(csv_path)
                if global_df1.train_df is None:
                    df = global_df1.current_df
                else:
                    df = global_df1.train_df
            except Exception as e:
                print("error", e)

            modified_df = df
            temp_df = modified_df.copy(deep=True)

            if method == "iqr":
                Q1 = df[column_name].quantile(0.25)
                Q3 = df[column_name].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                for column_name in df.select_dtypes(include=["number"]).columns:
                    column_mean = df[column_name].mean()
                    modified_df[column_name] = df[column_name].apply(
                        lambda x: column_mean if (x < lower_bound or x > upper_bound) else x
                    )

                # fig = go.Figure()

                # fig.add_trace(go.Scatter(x=temp_df[date_column], y=temp_df[column_name], mode='lines', name='Original Data', line=dict(color='gray', dash='dot')))
                # fig.add_trace(go.Scatter(x=modified_df[date_column], y=modified_df[column_name], mode='lines', name='Modified Data', line=dict(color='blue')))
                # fig.update_layout(
                #     title=f"Outlier Detection using {method}: Original vs Modified Data",
                #     xaxis_title=date_column,
                #     yaxis_title=column_name,
                #     legend_title="Legend",
                # )
                # fig_json = json.loads(fig.to_json())


                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    # x=temp_df[date_column],
                    # y=temp_df[column_name],
                    x=global_df1.typecasted_df[date_column],
                    y=global_df1.typecasted_df[column_name],
                    mode='lines',
                    name='Original',
                    line=dict(color='grey')
                ))

                if global_df1.missing_values_treated_df is None:
                    missing_values_plot_df = global_df1.missing_values_detection_df
                else:
                    missing_values_plot_df = global_df1.missing_values_treated_df

                fig.add_trace(go.Scatter(
                    x=missing_values_plot_df[date_column],
                    y=missing_values_plot_df[column_name],
                    mode='lines',
                    name='Missing Value Treated',
                    line=dict(color='green')
                ))

                fig.add_trace(go.Scatter(
                    x=modified_df[date_column],
                    y=modified_df[column_name],
                    mode='lines',
                    name='Missing Value & Outlier Treated',
                    line=dict(color='blue')
                ))

                fig.update_layout(
                    title=f"Preprocessed - Adjusted Post Missing Value & Outlier Treatmend",
                    xaxis_title=date_column,
                    yaxis_title=column_name,
                    legend_title="Legend",
                    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                    plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot area
                    font=dict(color='white'),  # White font color for dark theme
                    xaxis=dict(
                        gridcolor='gray',  # Grid color for better visibility
                        linecolor='white',  # Axis line color
                        zerolinecolor='gray'  # Zero line color
                    ),
                    yaxis=dict(
                        gridcolor='gray',  # Grid color for better visibility
                        linecolor='white',  # Axis line color
                        zerolinecolor='gray'  # Zero line color
                    ),
                    legend=dict(
                        bgcolor='rgba(0,0,0,0)',  # Transparent legend background
                        bordercolor='white'  # Legend border color
                    ),
                    updatemenus=[
                        {
                            "buttons": [
                                {
                                    "args": [{"xaxis.gridcolor": 'gray', "yaxis.gridcolor": 'gray'}],
                                    "label": "Show Grid",
                                    "method": "relayout"
                                },
                                {
                                    "args": [{"xaxis.gridcolor": 'rgba(0,0,0,0)', "yaxis.gridcolor": 'rgba(0,0,0,0)'}],
                                    "label": "Hide Grid",
                                    "method": "relayout"
                                }
                            ],
                            "direction": "down",
                            "showactive": True,
                            "x": 1,
                            "xanchor": "right",
                            "y": 1.1,
                            "yanchor": "top"
                        }
                    ]
                )

                fig_json = json.loads(fig.to_json())

                with open(constants.GRAPH_FLAG_PATH + "outliers.txt", "w") as f:
                    f.write(json.dumps(fig_json))
                    f.close()

                if not os.path.exists(constants.GRAPH_FLAG_PATH + "graph_flag.txt"):
                    open(constants.GRAPH_FLAG_PATH + "graph_flag.txt", 'w').close()

                with open(constants.GRAPH_FLAG_PATH + "graph_flag.txt", "r+") as f:
                    lines = f.readlines()
                    if not lines:
                        lines.append("1\n")
                        lines.append("outliers.txt\n")
                    else:
                        flag = True
                        while flag:
                            f.seek(0)
                            lines = f.readlines()
                            if int(lines[0]) == 1:
                                time.sleep(2)
                                continue
                            first_line = 1
                            lines[0] = f"{first_line}\n"
                            lines.append("outliers.txt\n")
                            flag=False

                    f.seek(0)
                    f.writelines(lines)
                    f.close()

                global_df1.outlier_treated_df = modified_df
                global_df1.current_df = modified_df
                # modified_df.to_csv(constants.INTERMEDIATE_DF_PATH + "train.csv", index=False)
                return "Outlier treated by replacing with mean using IQR method"

            elif method == "zscore":
                mean = df[column_name].mean()
                std = df[column_name].std()

                column_mean = df[column_name].mean()
                modified_df[column_name] = df[column_name].apply(
                    lambda x: column_mean if np.abs((x - mean) / std) > threshold else x
                )

                # fig = go.Figure()

                # fig.add_trace(go.Scatter(x=temp_df[date_column], y=temp_df[column_name], mode='lines', name='Original Data', line=dict(color='gray', dash='dot')))
                # fig.add_trace(go.Scatter(x=modified_df[date_column], y=modified_df[column_name], mode='lines', name='Modified Data', line=dict(color='blue')))
                # fig.update_layout(
                #     title=f"Outlier Detection using {method}: Original vs Modified Data",
                #     xaxis_title=date_column,
                #     yaxis_title=column_name,
                #     legend_title="Legend",
                # )
                # fig_json = json.loads(fig.to_json())


                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    # x=temp_df[date_column],
                    # y=temp_df[column_name],
                    x=global_df1.typecasted_df[date_column],
                    y=global_df1.typecasted_df[column_name],
                    mode='lines',
                    name='Original',
                    line=dict(color='grey')
                ))

                if global_df1.missing_values_treated_df is None:
                    missing_values_plot_df = global_df1.missing_values_detection_df
                else:
                    missing_values_plot_df = global_df1.missing_values_treated_df

                fig.add_trace(go.Scatter(
                    x=missing_values_plot_df[date_column],
                    y=missing_values_plot_df[column_name],
                    mode='lines',
                    name='Missing Value Treated',
                    line=dict(color='green')
                ))

                fig.add_trace(go.Scatter(
                    x=modified_df[date_column],
                    y=modified_df[column_name],
                    mode='lines',
                    name='Missing Value & Outlier Treated',
                    line=dict(color='blue')
                ))

                fig.update_layout(
                    title=f"Preprocessed - Adjusted Post Missing Value & Outlier Treatmend",
                    xaxis_title=date_column,
                    yaxis_title=column_name,
                    legend_title="Legend",
                    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                    plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot area
                    font=dict(color='white'),  # White font color for dark theme
                    xaxis=dict(
                        gridcolor='gray',  # Grid color for better visibility
                        linecolor='white',  # Axis line color
                        zerolinecolor='gray'  # Zero line color
                    ),
                    yaxis=dict(
                        gridcolor='gray',  # Grid color for better visibility
                        linecolor='white',  # Axis line color
                        zerolinecolor='gray'  # Zero line color
                    ),
                    legend=dict(
                        bgcolor='rgba(0,0,0,0)',  # Transparent legend background
                        bordercolor='white'  # Legend border color
                    ),
                    updatemenus=[
                        {
                            "buttons": [
                                {
                                    "args": [{"xaxis.gridcolor": 'gray', "yaxis.gridcolor": 'gray'}],
                                    "label": "Show Grid",
                                    "method": "relayout"
                                },
                                {
                                    "args": [{"xaxis.gridcolor": 'rgba(0,0,0,0)', "yaxis.gridcolor": 'rgba(0,0,0,0)'}],
                                    "label": "Hide Grid",
                                    "method": "relayout"
                                }
                            ],
                            "direction": "down",
                            "showactive": True,
                            "x": 1,
                            "xanchor": "right",
                            "y": 1.1,
                            "yanchor": "top"
                        }
                    ]
                )

                fig_json = json.loads(fig.to_json())

                with open(constants.GRAPH_FLAG_PATH + "outliers.txt", "w") as f:
                    f.write(json.dumps(fig_json))
                    f.close()

                if not os.path.exists(constants.GRAPH_FLAG_PATH + "graph_flag.txt"):
                    open(constants.GRAPH_FLAG_PATH + "graph_flag.txt", 'w').close()

                with open(constants.GRAPH_FLAG_PATH + "graph_flag.txt", "r+") as f:
                    lines = f.readlines()
                    if not lines:
                        lines.append("1\n")
                        lines.append("outliers.txt\n")
                    else:
                        flag = True
                        while flag:
                            f.seek(0)
                            lines = f.readlines()
                            if int(lines[0]) == 1:
                                time.sleep(2)
                                continue
                            first_line = 1
                            lines[0] = f"{first_line}\n"
                            lines.append("outliers.txt\n")
                            flag=False

                    f.seek(0)
                    f.writelines(lines)
                    f.close()

                global_df1.outlier_treated_df = modified_df
                global_df1.current_df = modified_df
                # modified_df.to_csv(constants.INTERMEDIATE_DF_PATH + "train.csv", index=False)
                return "Outlier treated by replacing with mean using Z-score method"
            else:
                raise ValueError("Method must be either 'iqr' or 'zscore'")

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(1)
                  # Wait a bit before retrying
            else:
                print("All attempts failed.")
                return str(e)


# @tool
# def stationarity_ADF(column_name, d:int):
#     """
#     Perform the Augmented Dickey-Fuller (ADF) test to check the stationarity of a time series.

#     The ADF test is used to determine whether a time series is stationary or has a unit root (i.e., is non-stationary).
#     It tests the null hypothesis that the time series has a unit root, against the alternative hypothesis that the series is stationary.

#     Parameters:
#     - it takes target column name (column used for forecasting) for Dickey-fuller test as input.

#     Returns:
#     - str: A string dictionary containing the results of the ADF test with the following key-value pairs:
#       - "ADF": The test statistic from the ADF test.
#       - "P VALUE": The p-value corresponding to the test statistic.
#       - "NUM OF LAGS": The number of lags used in the test.
#       - "CRITICAL VALUES": A dictionary with critical values for different significance levels (1%, 5%, and 10%).
#     """
#     csv_path = constants.INTERMEDIATE_DF_PATH + "intermediate.csv"
#     df = pd.read_csv(csv_path)
#     df = df[column_name]
#     dftest = adfuller(df, autolag="AIC")
#     adf_test_res = {
#         "ADF": dftest[0],
#         "P VALUE": dftest[1],
#         "NUM OF LAGS": dftest[2],
#         "CRITICAL VALUES": dftest[4],
#     }

#     return str(adf_test_res)


@tool
def stationarity_ADF(column_name, d: int):
    """
    Perform the Augmented Dickey-Fuller (ADF) test to check the stationarity of a time series.

    The ADF test is used to determine whether a time series is stationary or has a unit root (i.e., is non-stationary).
    It tests the null hypothesis that the time series has a unit root, against the alternative hypothesis that the series is stationary.

    Parameters:
    - column_name (str): it takes target column name (column used for forecasting) for Dickey-fuller test as input.
    - d (int): it takes d parameter as d-order differencing, it performs d-order differencing on the target column

    """
    max_retries = constants.MAX_RETRIES
    for attempt in range(max_retries):
        try:
            csv_path = constants.INTERMEDIATE_DF_PATH + "train.csv"
            try:
                # df = pd.read_csv(csv_path)
                if global_df1.outlier_treated_df is None:
                    df = global_df1.current_df
                else:
                    df = global_df1.outlier_treated_df
            except Exception as e:
                print('ERROR'. e)
            if d > 0:
                df[f"d{d}"] = df[column_name]
                for _ in range(1, d + 1):
                    df[f"d{d}"] = df[f"d{d}"].diff().dropna()

                differenced_series = df[f"d{d}"].dropna()
            else:
                differenced_series = df[column_name]

            dftest = adfuller(differenced_series, autolag="AIC")
            global_df1.adf_df = df
            # global_df1.current_df = df
            # df.to_csv(constants.INTERMEDIATE_DF_PATH + "train.csv", index=False)

            adf_test_res = {
                "ADF": dftest[0],
                "P VALUE": dftest[1],
                "NUM OF LAGS": dftest[2],
                "CRITICAL VALUES": dftest[4],
            }

            return str(adf_test_res)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(1)
                  # Wait a bit before retrying
            else:
                print("All attempts failed.")
                return str(e)


@tool
def ljung_box_test(column_name):
    """
    Perform the Ljung-Box test on a time series.

    Parameters:
    - it takes target column name (column used for forecasting) for ljuang_box test as input.

    Returns:
    pd.DataFrame: A DataFrame containing the test statistics and p-values for each lag.
    """
    max_retries = constants.MAX_RETRIES
    for attempt in range(max_retries):
        try:
            csv_path = constants.INTERMEDIATE_DF_PATH + "train.csv"
            try:
                # df = pd.read_csv(csv_path)
                if global_df1.outlier_treated_df is None:
                    df = global_df1.current_df
                else:
                    df = global_df1.outlier_treated_df
            except Exception as e:
                print("error", e)
            df = df[column_name]
            lags = int(np.sqrt(len(df)))
            time_series = np.asarray(df)

            lb_test = sm.stats.acorr_ljungbox(time_series, lags=[lags], return_df=True)
            lb_pvalue = lb_test["lb_pvalue"].tolist()[0]
            lb_stat = lb_test["lb_stat"].tolist()[0]
            # if lb_pvalue < 0.05:
            #     description = "There is evidence to suggest that the residuals are not independently distributed; they exhibit serial correlation."
            # else:
            #     description = "There is not enough evidence to suggest that the residuals are not independently distributed; they may be considered as independently distributed."

            result = {"lb_stat": str(lb_stat), "lb_pvalue": str(lb_pvalue), "lag": lags}

            info_string = f"\n ljung_box test insights: {str(result)}"
            return info_string
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(1)
                  # Wait a bit before retrying
            else:
                print("All attempts failed.")
                return str(e)

# @tool
# def stationarity_kpss(column_name, regression="ct"):
#     """
#     Perform the KPSS test for stationarity on a given time series.

#     Parameters:
#         column_name (str): The column name of the target column in the dataframe on which time series analysis is to be performed.
#         regression (str): 'c' for stationarity around a constant (default),
#                            'ct' for stationarity around a deterministic trend.

#     Returns:
#         dict: A dictionary containing the KPSS statistic, p-value, critical values, and lags.
#     """

#     csv_path = constants.INTERMEDIATE_DF_PATH + "intermediate.csv"
#     df = pd.read_csv(csv_path)
#     time_series = df[column_name]
#     statistic, p_value, lags, critical_values = kpss(time_series, regression=regression)
#     if p_value < 0.05:
#         stationarity = True
#     else:
#         stationarity = False

#     result = {
#         "KPSS Statistic": statistic,
#         "p-value": p_value,
#         "Lags": lags,
#         "Critical Values": critical_values,
#         "is_stationary": stationarity,
#     }

#     return result


@tool
def stationarity_kpss(column_name, d: int, regression="ct"):
    """
    Perform the KPSS test for stationarity on a given time series after applying differencing if required.

    Parameters:
        column_name (str): The column name of the target column in the dataframe on which time series analysis is to be performed.
        d (int): The order of differencing to make the series stationary (0 means no differencing).
        regression (str): 'c' for stationarity around a constant (default),
                           'ct' for stationarity around a deterministic trend.

    Returns:
        str: A dictionary containing the KPSS statistic, p-value, critical values, lags, and whether the series is stationary.
    """
    max_retries = constants.MAX_RETRIES
    for attempt in range(max_retries):
        try:

            csv_path = constants.INTERMEDIATE_DF_PATH + "train.csv"
            try:
                # df = pd.read_csv(csv_path)
                if global_df1.outlier_treated_df is None:
                    df = global_df1.current_df
                else:
                    df = global_df1.outlier_treated_df
            except Exception as e:
                print("error", e)

            if d > 0:
                df[f"d{d}"] = df[column_name]
                for i in range(1, d + 1):
                    df[f"d{d}"] = df[f"d{d}"].diff().dropna()
                time_series = df[f"d{d}"].dropna()
            else:
                time_series = df[column_name]

            statistic, p_value, lags, critical_values = kpss(time_series, regression=regression)

            if p_value < 0.05:
                stationarity = True
            else:
                stationarity = False

            global_df1.kpss_df = df
            # global_df1.current_df = df
            # df.to_csv(constants.INTERMEDIATE_DF_PATH + "train.csv", index=False)

            result = {
                "KPSS Statistic": statistic,
                "p-value": p_value,
                "Lags": lags,
                "Critical Values": critical_values,
                # "is_stationary": stationarity,
            }
            return str(result)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(1)
                  # Wait a bit before retrying
            else:
                print("All attempts failed.")
                return str(e)



@tool
def missing_values_treatment(target_column: str, method: str = "ffill", n_neighbors=5):
    """
    Impute missing values in the target column of the DataFrame.
    Parameters:
    - df: DataFrame containing the time series data
    - target_column: Name of the target column that needs to be forecasted.
    - method: Imputation method to use ('ffill', 'bfill', 'linear', 'mean', 'median', 'mode', 'knn', 'interpolate', etc.)
    - n_neighbors: Number of neighbors to use for KNN imputation (default is 5)
    Returns:
    - DataFrame with the imputed values in the target column
    """
    max_retries = constants.MAX_RETRIES
    for attempt in range(max_retries):
        try:
            try:
                # df = pd.read_csv(constants.INTERMEDIATE_DF_PATH + "intermediate.csv")
                if global_df1.missing_values_detection_df is None:
                    df = global_df1.current_df
                else:
                    df = global_df1.missing_values_detection_df
            except Exception as e:
                print("error", e)

            if target_column not in df.columns:
                raise ValueError(f"Column '{target_column}' does not exist in the DataFrame.")

            target = df[target_column]

            if method == "ffill":
                df[target_column] = target.ffill()
            elif method == "bfill":
                df[target_column] = target.bfill()
            elif method == "linear":
                df[target_column] = target.interpolate(method="linear")
            elif method == "mean":
                mean_value = target.mean()
                df[target_column] = target.fillna(mean_value)
            elif method == "median":
                median_value = target.median()
                df[target_column] = target.fillna(median_value)
            elif method == "mode":
                mode_value = target.mode().iloc[0]
                df[target_column] = target.fillna(mode_value)
            elif method == "interpolate":
                df[target_column] = target.interpolate(method="polynomial", order=2)
            elif method == "knn":
                df_numeric = df.select_dtypes(include=[np.number])
                imputer = KNNImputer(n_neighbors=n_neighbors)
                df_imputed = pd.DataFrame(
                    imputer.fit_transform(df_numeric), columns=df_numeric.columns
                )
                df[target_column] = df_imputed[target_column]
            else:
                raise ValueError(f"Unknown method '{method}'. Please choose a valid method.")

            # print("Missing Values treatment df: ", df)
            # df.to_csv(constants.INTERMEDIATE_DF_PATH + "intermediate.csv", index=False)
            global_df1.missing_values_treated_df = df
            global_df1.current_df = df
            return f"The missing values in the target column {target_column} have been imputed using the {method} method."
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(1)
                  # Wait a bit before retrying
            else:
                print("All attempts failed.")
                return str(e)

@tool
def get_fft(column_name, N=5, T=1.0):
    """
    Perform FFT on a time series from a CSV file and estimate the dominant frequencies and their corresponding seasonality periods.

    Parameters:
    column_name (str): The name of the column containing the time series data.
    N (int): The number of dominant frequencies to consider (default is 5).
    T (float): The sampling period (default is 1.0 for monthly data).

    Returns:
    list: Estimated seasonality periods.
    """
    max_retries = constants.MAX_RETRIES
    for attempt in range(max_retries):
        try:
            # Read the CSV file
            try:
                # df = pd.read_csv(constants.INTERMEDIATE_DF_PATH + "train.csv")
                if global_df1.outlier_treated_df is None:
                    df = global_df1.current_df
                else:
                    df = global_df1.outlier_treated_df
            except Exception as e:
                print("error", e)

            # Extract the time series data
            series = (
                df[column_name].dropna().values
            )  # Drop any NaN values and convert to numpy array

            # Perform FFT
            n = len(series)  # Length of the time series
            yf = fft(series)
            xf = fftfreq(n, T)[: n // 2]

            # Magnitude of the FFT result
            magnitude = (2.0 / n) * np.abs(yf[: n // 2])

            # Find the indices of the dominant frequencies
            peak_indices = np.argsort(magnitude)[-N:]
            dominant_frequencies = xf[peak_indices]

            # Calculate the seasonality periods
            seasonality_periods = [1 / freq for freq in dominant_frequencies if freq != 0]

            buf = io.BytesIO()

            buf.seek(0)

            # print(f"Estimated seasonality period: {seasonality_periods}")

            plt.figure(figsize=(10, 6))
            plt.plot(xf, magnitude)
            plt.title("FFT Plot")
            plt.xlabel("Frequency (cycles per time unit)")
            plt.ylabel("Magnitude")
            plt.grid(True)
            plt.savefig(buf, format="png")
            buf.seek(0)
            # plt.show()
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            plt.close()

            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "This is the graph of the time series in frequency domain post applying fft in it, what are the dominant  seasonality periods present in it. Note: Give period only if the graph has enough evidence which suggests presece of seasonality. Also give a definite answer.",
                        # "text": "The following is the freqency domain graph for a time series data. How many seasonal periods or cycles do you see in the graph? Also find if there are single or multiple seasonal components in the data."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ],
            )
            response = AgentTemplate.llm.invoke([message])
            # print(response)
            fft_insigths = {"seasonality_description": response}
            return str(fft_insigths)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(1)
                  # Wait a bit before retrying
            else:
                print("All attempts failed.")
                return str(e)


@tool
def acf_pacf_insights(column_name, lags=0):
    """
    column_name (str): The name of the column containing the time series data.
    lags (int): Number of lags being sent for ACF, PACF. Default is 0 which will set lags to half the length of the column.
    """
    max_retries = constants.MAX_RETRIES
    for attempt in range(max_retries):
        try:
            csv_train_path = constants.INTERMEDIATE_DF_PATH + "train.csv"
            try:
                # train = pd.read_csv(csv_train_path)
                if global_df1.outlier_treated_df is None:
                    train = global_df1.current_df
                else:
                    train = global_df1.outlier_treated_df
            except Exception as e:
                print("error", e)

            if lags == 0:
                lags = (
                    len(train[column_name]) // 2
                )  # Correction: use length of the column data, not the name

            # Create a figure and axis to plot on
            fig, ax = plt.subplots(1, 2, figsize=(15, 6))

            plot_acf(train[column_name], lags=lags, ax=ax[0])
            ax[0].set_title("ACF (Autocorrelation Function)")
            plot_pacf(train[column_name], lags=lags, ax=ax[1])
            ax[1].set_title("PACF (Partial Autocorrelation Function)")

            buffer = BytesIO()
            plt.savefig(buffer, format="png")
            plt.close(fig)  # Close the figure to free memory

            # Encode the buffer to base64
            buffer.seek(0)  # Seek to the start of the BytesIO buffer before reading
            base64_str = base64.b64encode(buffer.read()).decode("utf-8")

            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Below is a combined image containing both the Autocorrelation Function (ACF) and the Partial Autocorrelation Function (PACF) plots of a time series, encoded in base64 format. Could you analyze these plots and provide insights on any significant autocorrelations or partial autocorrelations that suggest the presence of seasonality or other patterns? Please interpret both ACF and PACF values, indicating potential seasonal patterns or other insights derived from the plots. Also if needed, give logical parameters with justification for choosing those for model fitting as well.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_str}"},
                    },
                ],
            )

            # Assuming you have a mechanism to send this message to an LLM or similar service and receive a response
            response = AgentTemplate.llm.invoke([message])
            # print(response)
            fft_insights = {"acf_pacf_values": response}
            return str(fft_insights)

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(1)
                  # Wait a bit before retrying
            else:
                print("All attempts failed.")
                return str(e)


def json_serializable(value):
    """Convert non-serializable numpy values to JSON-compatible types."""

    max_retries = constants.MAX_RETRIES
    for attempt in range(max_retries):
        try:
            if isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, float) and (value != value):
                return None
            elif isinstance(value, dict):
                # return {k: json_serializable(v) for k, v in value.items()}
                filtered_dict = {k: json_serializable(v) for k, v in value.items()}
                return {k: v for k, v in filtered_dict.items() if v is not None and v is not np.nan and v != []}
            elif isinstance(value, np.float64):
                if np.isnan(value):
                    return np.nan
            else:
                return value
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(1)
                  # Wait a bit before retrying
            else:
                print("All attempts failed.")
                return str(e)


@tool
def optimize_hyperparameters(model_names, params, column_name):
    """
    Optimize hyperparameters for selected time series forecasting models.

    Parameters:
    - model_names : list[str]
        List of model names to optimize, e.g., ['ARIMA', 'SARIMA', 'Holt-Winters'].
    - params : dict
        Dictionary with model-specific hyperparameters in a simplified format.
        Example:
        {
            "ARIMA": {"p": [1], "d": [0], "q": [0]},
            "SARIMA": {"p": [1], "d": [0], "q": [0], "m": [12]},
            "Holt-Winters": {"seasonal_periods": [12]}
        }
    - column_name (str): Column with time series data.
    """

    warnings.filterwarnings('ignore')
    max_retries = constants.MAX_RETRIES
    # max_retries = 3
    for attempt in range(max_retries):
        try:
            result = {}
            # train = train_df

            for model_name in model_names:
                try:
                    if global_df1.outlier_treated_df is None:
                        train = global_df1.current_df
                    else:
                        train = global_df1.outlier_treated_df

                    train = train[column_name]
                    if model_name == "ARIMA" or model_name == "SARIMA":
                        result[model_name] = optimize_arima_sarima(model_name, train, params.get(model_name, {}))
                    elif model_name == "Holt-Winters":
                        result[model_name] = optimize_holt_winters(train, params.get(model_name, {}))
                    elif model_name == "Single Exponential Smoothing":
                        result[model_name] = optimize_single_exponential_smoothing(train)
                    else:
                        raise ValueError(f"Model '{model_name}' is not supported")
                except Exception as e:
                    print(f"Error occurred during {model_name} optimization:")
                    print(f"Parameters: {params.get(model_name, {})}")
                    print(f"Exception: {e}")
                    print(f"Traceback: {traceback.format_exc()}")

            return json.dumps(result)

        except Exception as e:
            print(f"Error: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(1)
            else:
                print("All attempts failed.")

def optimize_arima_sarima(model_name, train, params):
    """
    Optimizes ARIMA or SARIMA model hyperparameters.
    """
    try:
        p, d, q = params.get("p", [1]), params.get("d", [0]), params.get("q", [0])
        m = params.get("m", [None])  # SARIMA specific
        if m is not None:
            for i in range(len(m)):
                if m[i] == 1 or m[i] == 0:
                    m[i] = 12
        
        order = list(product(p, d, q))
        seasonal_order = list(product(p, d, q, m)) if model_name == "SARIMA" else [None]

        best_score, best_cfg = float("inf"), None
        for param in order:
            for s_param in seasonal_order:
                try:
                    if model_name == "SARIMA":
                        model = SARIMAX(train, order=param, seasonal_order=s_param, enforce_stationarity=False, enforce_invertibility=False)
                    else:
                        model = ARIMA(train, order=param)

                    model_fit = model.fit()
                    forecast = model_fit.fittedvalues
                    mse = mean_absolute_percentage_error(train, forecast)
                    # print("the best score: ", best_score)
                    if mse < best_score:
                        best_score, best_cfg = mse, [param, s_param]
                        # print("best cnfg: ", best_cfg)
                except Exception as e:
                    print(f"Error during model fitting in {model_name}: {e}")
                    print(f"Params: order={param}, seasonal_order={s_param}")
                    print(f"Traceback: {traceback.format_exc()}")
                    continue

        if model_name == "SARIMA":
            return {
                "params": {"order": best_cfg[0], "seasonal_order": best_cfg[1], "best_score": best_score}
            }
        else:
            return {
                "params": {"order": best_cfg[0], "best_score": best_score}
            }
    except Exception as e:
        print(f"Error in {model_name} optimization: {e}")
        print(f"Params: {params}")
        print(f"Traceback: {traceback.format_exc()}")
        raise

def optimize_holt_winters(train, params):
    """
    Optimizes Holt-Winters model hyperparameters.
    """
    try:
        seasonal_periods = params.get("seasonal_periods", [12])
        best_score, best_cfg, seasonal_period = float("inf"), None, None

        for seasonal in seasonal_periods:
            try:
                if isinstance(seasonal, list):
                    if len(train) >= 2 * seasonal[0] and seasonal[0] > 1:
                        model = ExponentialSmoothing(train, seasonal="add", trend="add", seasonal_periods=seasonal)
                    else:
                        model = ExponentialSmoothing(train, seasonal=None, trend="add")
                else:
                    if len(train) >= 2 * seasonal and seasonal > 1:
                        model = ExponentialSmoothing(train, seasonal="add", trend="add", seasonal_periods=seasonal)
                    else:
                        model = ExponentialSmoothing(train, seasonal=None, trend="add")
                model_fit = model.fit()
                forecast = model_fit.fittedvalues
                mse = mean_absolute_percentage_error(train, forecast)
                # print("the best score: ", best_score)
                if mse < best_score:
                    best_score, best_cfg, seasonal_period = mse, json_serializable(model_fit.params), seasonal
                    # print("best cnfg: ", best_cfg)
            except Exception as e:
                print(f"Error during model fitting in Holt-Winters: {e}")
                print(f"Params: seasonal_periods={seasonal}")
                print(f"Traceback: {traceback.format_exc()}")
                continue

        return {
            "params": {"model_config": best_cfg, "seasonal_periods": seasonal_period, "best_score": best_score}
        }
    except Exception as e:
        print(f"Error in Holt-Winters optimization: {e}")
        print(f"Params: {params}")
        print(f"Traceback: {traceback.format_exc()}")
        raise

def optimize_single_exponential_smoothing(train):
    """
    Optimizes Single Exponential Smoothing model.
    """
    try:
        model = SimpleExpSmoothing(train)
        model_fit = model.fit()
        forecast = model_fit.fittedvalues
        mse = mean_absolute_percentage_error(train, forecast)

        return {
            "params": {"params": json_serializable(model_fit.params), "best_score": mse}
        }
    except Exception as e:
        print(f"Error during Single Exponential Smoothing optimization: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise




# @tool
# def optimize_hyperparameters(model_names, params, column_name):
#     """
#     Optimize hyperparameters for selected time series forecasting models.
#     Parameters:
#     - model_names : list[str]
#         List of model names to optimize, e.g., ['ARIMA', 'SARIMA', 'Holt-Winters'].
#         Example: ['ARIMA', 'SARIMA']
#     - params : dict
#         Dictionary with model-specific hyperparameters:
#         - 'ARIMA' and 'SARIMA':
#            {'order': {'p': [list], 'd': [list], 'q': [list], 'm': [list]}}
#            Here, 'p', 'd', 'q' values should be informed by insights from previous EDA. For 'SARIMA', an additional 'm' parameter for seasonality effects is required.
#         - 'Holt-Winters': {'seasonal_periods': [int]}
#         Example (NOTE: while giving the parmas for multiple models, make sure to end the curly bracket of one model before starting the next model):'):
#             params = {
#                 'ARIMA': {'order': {'p': [1], 'd': [0], 'q': [0]}},
#                 'SARIMA': {'order': {'p': [1], 'd': [0], 'q': [0], 'm': [12]}},
#             }
#     - column_name (str): Column with time series data.
#     """
#     # parsed_params = JsonOutputParser(params)
#     # class Joke(BaseModel):
#     #     setup: str = Field(description="question to set up a joke")
#     #     punchline: str = Field(description="answer to resolve the joke")

#     max_retries = constants.MAX_RETRIES
#     for attempt in range(max_retries):
#         try:
#             result = {}
#             for model_name in model_names:
#                 try:
#                     # csv_path = constants.INTERMEDIATE_DF_PATH + "train.csv"
#                     # train = pd.read_csv(csv_path)
#                     if global_df1.outlier_treated_df is None:
#                         train = global_df1.current_df
#                     else:
#                         train = global_df1.outlier_treated_df
#                 except Exception as e:
#                     print("error", e)

#                 train = train[column_name]

#                 if model_name in ["ARIMA", "SARIMA"]:
#                     print("test1")
#                     p = params[model_name]["order"]["p"]
#                     d = params[model_name]["order"]["d"]
#                     q = params[model_name]["order"]["q"]
#                     if model_name == "SARIMA":
#                         m = params[model_name]["order"]["m"]
#                     order = list(product(p, d, q))
#                     seasonal_order = (
#                         list(product(p, d, q, m)) if model_name == "SARIMA" else [None]
#                     )

#                     best_score, best_cfg = float("inf"), None
#                     for param in order:
#                         for s_param in seasonal_order:
#                             try:
#                                 if model_name == "SARIMA":
#                                     print("test2")
#                                     model = SARIMAX(
#                                         train,
#                                         order=param,
#                                         seasonal_order=s_param,
#                                         enforce_stationarity=False,
#                                         enforce_invertibility=False,
#                                     )
#                                 else:
#                                     print("test3")
#                                     model = ARIMA(train, order=param)
#                                 print("test4")
#                                 model_fit = model.fit()
#                                 print("test5")
#                                 forecast = model_fit.fittedvalues
#                                 mse = mean_absolute_percentage_error(train, forecast)
#                                 if mse < best_score:
#                                     best_score, best_cfg = mse, [param, s_param]
#                             except Exception as e:
#                                 print("opti error", e)
#                                 continue
#                     if model_name == "SARIMA":
#                         print("test6")
#                         result[model_name] = {
#                             "params": {
#                                 "order": best_cfg[0],
#                                 "seasonal_order": best_cfg[1],
#                                 "best_score": best_score,
#                             }
#                         }
#                     else:
#                         result[model_name] = {
#                             "params": {"order": best_cfg[0], "best_score": best_score}
#                         }

#                 elif model_name == "Holt-Winters":
#                     seasonal_periods = params[model_name]["seasonal_periods"]
#                     best_score, best_cfg, seasonal_period = float("inf"), None, None
#                     for seasonal in seasonal_periods:
#                         try:
#                             if len(train) >= 2 * seasonal and seasonal > 1:
#                                 model = ExponentialSmoothing(
#                                     train,
#                                     seasonal="add",
#                                     trend="add",
#                                     seasonal_periods=seasonal,
#                                 )
#                             else:
#                                 model = ExponentialSmoothing(train, seasonal=None, trend="add")
#                             model_fit = model.fit()
#                             forecast = model_fit.fittedvalues
#                             mse = mean_absolute_percentage_error(train, forecast)
#                             print(mse)
#                             if mse < best_score:
#                                 best_score, best_cfg, seasonal_period = (
#                                     mse,
#                                     json_serializable(model_fit.params),
#                                     seasonal,
#                                 )
#                         except Exception as e:
#                             continue
#                     result[model_name] = {
#                         "params": {
#                             "model_config": best_cfg,
#                             "seasonal_periods": seasonal_period,
#                             "best_score": best_score,
#                         }
#                     }

#                 elif model_name in ["Random Forest", "XGBoost"]:
#                     param_grid = {
#                         "n_estimators": params[model_name]["n_estimators"],
#                         "max_depth": params[model_name]["max_depth"],
#                     }
#                     model = (
#                         RandomForestRegressor()
#                         if model_name == "Random Forest"
#                         else XGBRegressor()
#                     )
#                     grid_search = GridSearchCV(
#                         model, param_grid, cv=3, scoring="neg_mean_absolute_percentage_error"
#                     )
#                     grid_search.fit(np.arange(len(train)).reshape(-1, 1), train.values)
#                     result[model_name] = {"params": grid_search.best_params_}

#                 elif model_name == "Single Exponential Smoothing":
#                     model = SimpleExpSmoothing(train)
#                     model_fit = model.fit()
#                     forecast = model_fit.fittedvalues
#                     mse = mean_absolute_percentage_error(train, forecast)
#                     result[model_name] = {
#                         "params": {"params": json_serializable(model_fit.params), "best_score": mse}
#                     }

#                 else:
#                     raise ValueError("Model not supported")
#             return json.dumps(result)

#         except Exception as e:
#             print(e)
#             print(f"Attempt {attempt + 1} failed: {e.with_traceback()}")
#             if attempt < max_retries - 1:
#                 print("Retrying...")
#                 time.sleep(1)  # Wait a bit before retrying
#             else:
#                 print("All attempts failed.")




def forecast_time_series(model_name, train, test, params):
    """
    Forecast time series using a specified model.

    Generates forecasts for a test period based on the provided training data and model
    parameters.

    Parameters:
    model_name : str
    params : dict
        Dictionary of model-specific parameters:
        - 'ARIMA'/'SARIMA': 'order' (tuple: p, d, q, seasonal_order)
        - 'Holt-Winters': 'seasonal_periods' (int)
        - 'Random Forest'/'XGBoost': Model parameters (e.g., 'n_estimators', 'max_depth')
        - 'Single Exponential Smoothing': (no specific parameters)
    """

    # csv_train_path=constants.INTERMEDIATE_DF_PATH + "train.csv"
    # csv_test_path=constants.INTERMEDIATE_DF_PATH + "test.csv"
    # train=pd.read_csv(csv_train_path)
    # test=pd.read_csv(csv_test_path)
    max_retries = constants.MAX_RETRIES
    for attempt in range(max_retries):
        try:
            if model_name == "ARIMA":
                model = ARIMA(train, order=params["order"])
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=len(test))
            elif model_name == "SARIMA":
                if params["seasonal_order"]:
                    temp = params["seasonal_order"]
                    temp[-1] = 12
                    params["seasonal_order"] = temp
                model = SARIMAX(
                    train, order=params["order"], seasonal_order=params["seasonal_order"]
                )
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=len(test))
            elif model_name == "Holt-Winters":
                seasonal_periods = params["seasonal_periods"]
                if isinstance(seasonal_periods, list):
                    if len(train) >= 2 * seasonal_periods[0] and seasonal_periods[0] > 1:
                        model = ExponentialSmoothing(
                            train, seasonal="add", trend="add", seasonal_periods=seasonal_periods
                        )
                    else:
                        model = ExponentialSmoothing(train, seasonal=None, trend="add")
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=len(test))
                else:
                    if len(train) >= 2 * seasonal_periods and seasonal_periods > 1:
                        model = ExponentialSmoothing(
                            train, seasonal="add", trend="add", seasonal_periods=seasonal_periods
                        )
                    else:
                        model = ExponentialSmoothing(train, seasonal=None, trend="add")
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=len(test))
            # elif model_name == "Prophet":
            #     df = pd.DataFrame({'ds': train.index, 'y': train.values})
            #     model = Prophet(daily_seasonality=params['daily_seasonality'])
            #     model.fit(df)
            #     future = model.make_future_dataframe(periods=len(test))
            #     forecast = model.predict(future)['yhat'][-len(test):]
            elif model_name == "Random Forest":
                model = RandomForestRegressor(**params)
                model.fit(np.arange(len(train)).reshape(-1, 1), train.values)
                forecast = model.predict(
                    np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
                )
            elif model_name == "XGBoost":
                # model = XGBRegressor(**params)
                model = XGBRegressor()
                model.fit(np.arange(len(train)).reshape(-1, 1), train.values)
                forecast = model.predict(
                    np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
                )
            elif model_name == "Single Exponential Smoothing":
                model = SimpleExpSmoothing(train)
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=len(test))
            else:
                raise ValueError("Model not supported")
            return forecast
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(1)  # Wait a bit before retrying
            else:
                print("All attempts failed.")

def plot_model_selection_evaluation(model_dict):
    """
    This function takes a dictionary of models with their evaluation metrics and
    generates a side-by-side bar chart for MSE and MAPE, with a grid toggle button.

    Args:
    - model_dict (dict): A dictionary where each key is a model name and the value is
        another dictionary containing the model's parameters and evaluation metrics.
    """

    # Prepare the data for the bar charts
    models = list(model_dict.keys())
    mse_values = [model_dict[model]['eval']['mse'] for model in models]
    mape_values = [model_dict[model]['eval']['mape'] for model in models]
    params = [model_dict[model]['params'] for model in models]

    # Create subplots (side-by-side)
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["MSE (Mean Squared Error)", "MAPE (Mean Absolute Percentage Error)"]
    )

    # Add MSE bar chart
    fig.add_trace(
        go.Bar(
            x=models,
            y=mse_values,
            name='MSE',
            hovertext=[str(param) for param in params],
            hoverinfo='text',
            marker=dict(color='blue'),
        ),
        row=1, col=1
    )

    # Add MAPE bar chart
    fig.add_trace(
        go.Bar(
            x=models,
            y=mape_values,
            name='MAPE',
            hovertext=[str(param) for param in params],
            hoverinfo='text',
            marker=dict(color='green'),
        ),
        row=1, col=2
    )

    # Update layout for a dark theme and transparent background
    fig.update_layout(
        title="Model Evaluation: MSE and MAPE",
        paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot area
        font=dict(color='white'),  # White font color for dark theme
        showlegend=False,
        xaxis=dict(
            title="Models",
            color='white',  # Axis line color
            gridcolor='gray',  # Grid color for better visibility
        ),
        yaxis=dict(
            title="MSE",
            color='white',
            gridcolor='gray',
        ),
        xaxis2=dict(
            title="Models",
            color='white',
            gridcolor='gray',  # Grid color for the second chart
        ),
        yaxis2=dict(
            title="MAPE",
            color='white',
            gridcolor='gray',  # Grid color for the second chart
        ),
        margin=dict(t=50, b=50, l=50, r=50),
        updatemenus=[{
            "buttons": [
                {
                    "args": [{"xaxis.gridcolor": 'gray', "yaxis.gridcolor": 'gray', "xaxis2.gridcolor": 'gray', "yaxis2.gridcolor": 'gray'}],
                    "label": "Show Grid",
                    "method": "relayout"
                },
                {
                    "args": [{"xaxis.gridcolor": 'rgba(0,0,0,0)', "yaxis.gridcolor": 'rgba(0,0,0,0)', "xaxis2.gridcolor": 'rgba(0,0,0,0)', "yaxis2.gridcolor": 'rgba(0,0,0,0)'}],
                    "label": "Hide Grid",
                    "method": "relayout"
                }
            ],
            "direction": "down",
            "showactive": True,
            "x": 1,  # Position at the right
            "xanchor": "right",  # Right align the button
            "y": 1.25,  # Move the button slightly higher to avoid overlap
            "yanchor": "top"
        }]
    )

    fig_json = json.loads(fig.to_json())
    return fig_json

@tool
def evaluate_model(model_names, params, column_name):
    """
    Evaluate model performance using MSE and MAPE.
    Fits the model on the combined training and test data, forecasts the validation period,
    and computes MSE and MAPE.

    Parameters:
    model_name : list[str]
    params : dict
        Dictionary of model-specific parameters:
        Takes the input from the optimize_hyperparameters function
        Example:
        {
            'ARIMA': {'params': {'order': [0, 0, 1]}, 'best_score': 0.114},
            'SARIMA': {'params': {'order': [2, 1, 1], 'seasonal_order': [0, 0, 2, 12]}, 'best_score': 0.174},
            'Holt-Winters': {'params': {'model_config': {'initial_seasons': [1,2], "use_boxcox": false, "remove_bias": false}, "seasonal_periods": 2, "best_score": 0.145}}},
        }
    column_name (str): The name of the column containing the time series data.
    """

    max_retries = constants.MAX_RETRIES
    for attempt in range(max_retries):
        try:

            try:
                test = global_df1.test_df
                validation = global_df1.validation_df
                if global_df1.outlier_treated_df is None:
                    train = global_df1.current_df
                else:
                    train = global_df1.outlier_treated_df
            except Exception as e:
                print("error", e)

            for model_name in model_names:

                train_series = train[column_name]
                test_series = test[column_name]
                validation_series = validation[column_name]

                combined_train_test = pd.concat([train_series, test_series])
                # print("params before forecast_time_series func")
                # print(params[model_name])
                forecast = forecast_time_series(
                    model_name, combined_train_test, validation_series, params[model_name]["params"]
                )
                mse = mean_squared_error(validation_series, forecast)
                mape = mean_absolute_percentage_error(validation_series, forecast)
                # params[model_name]['eval'] = {"mse": mse, "mape": mape, "forecast": forecast}
                params[model_name]["eval"] = {"mse": float(mse), "mape": float(mape)}

            # model_names_list = []
            # params_list = []
            # evals_list = []

            # for model_name, model_data in params.items():
            #     model_names_list.append(model_name)
            #     params_data = model_data.get('params', {})
            #     eval_data = model_data.get('eval', {})

            #     params_list.append(str(params_data))
            #     evals_list.append(str(eval_data))

            # df = pd.DataFrame({
            #     'Model Name': model_names_list,
            #     'Parameters': params_list,
            #     'Score': evals_list
            # })

            # header = dict(values=["Model Name", "Parameters", "Score"])
            # cells = dict(values=[df["Model Name"], df["Parameters"], df["Score"]])

            # fig = go.Figure(data=[go.Table(header=header, cells=cells)])

            # table_rows = []
            # for model, data in params.items():
            #     params = data['params']
            #     mse = f"{data['eval']['mse']:.2e}"
            #     mape = f"{data['eval']['mape']*100:.2f}%"


            #     for key, value in params.items():
            #         if isinstance(value, dict):
            #             for subkey, subvalue in value.items():
            #                 table_rows.append([model, f"{key}_{subkey}", subvalue])
            #         else:
            #             table_rows.append([model, key, value])

            #     table_rows.append([model, '<b>MSE</b>', f'<b>{mse}</b>'])
            #     table_rows.append([model, '<b>MAPE</b>', f'<b>{mape}</b>'])

            # columns = ['Model', 'Parameter Name', 'Parameter Value']

            # fig = go.Figure(data=[go.Table(
            #     header=dict(values=columns, fill_color='paleturquoise', align='center'),
            #     cells=dict(values=list(zip(*table_rows)), fill_color='lavender', align='left'))
            # ])

            # fig.update_layout(
            #     title="Model Evaluation and Parameters",
            #     margin=dict(t=50, b=50, l=50, r=50),
            #     height=600
            # )
            # fig_json = json.loads(fig.to_json())

            fig_json = plot_model_selection_evaluation(params)

            with open(constants.GRAPH_FLAG_PATH + "model_selection.txt", "w") as f:
                f.write(json.dumps(fig_json))
                f.close()

            if not os.path.exists(constants.GRAPH_FLAG_PATH + "graph_flag.txt"):
                open(constants.GRAPH_FLAG_PATH + "graph_flag.txt", 'w').close()

            with open(constants.GRAPH_FLAG_PATH + "graph_flag.txt", "r+") as f:
                lines = f.readlines()
                if not lines:
                    lines.append("1\n")
                    lines.append("model_selection.txt\n")
                else:
                    flag = True
                    while flag:
                        f.seek(0)
                        lines = f.readlines()
                        if int(lines[0]) == 1:
                            time.sleep(2)
                            continue
                        first_line = 1
                        lines[0] = f"{first_line}\n"
                        lines.append("model_selection.txt\n")
                        flag = False

                f.seek(0)
                f.writelines(lines)
                f.close()
            return json.dumps(params)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(1)  # Wait a bit before retrying
            else:
                print("All attempts failed.")


def outlier_treatment_bound_for_forecasting(
    date_column, column_name, method="iqr", threshold=3
):
    """
    **OUTLIER TREATMENT METHOD**
    Treat outliers in a pandas DataFrame or Series using the specified method: IQR or Z-score. The outlier is capped at a certain value above the 75th percentile value or floored at a factor below the 25th percentile value.

    Parameters:
    - it takes date column name as an input
    - it takes target column name (column used for forecasting) for Outlier Treatment as input.
    - method (str): The method to use for outlier detection. Options are:
      - "iqr" : Use Interquartile Range (IQR) method.
      - "zscore" : Use Z-score method.
    - threshold (float): For the Z-score method, the threshold above which data points are considered outliers.
      Default is 3. For the IQR method, this parameter is ignored.
    """
    max_retries = constants.MAX_RETRIES
    for attempt in range(max_retries):
        try:
            try:
                if global_df1.missing_values_treated_df is None:
                    df = global_df1.grouped_dates_df
                else:
                    df = global_df1.missing_values_treated_df
            except Exception as e:
                print("error", e)
            try:
                # train_df = pd.read_csv(csv_train_path)
                if global_df1.outlier_treated_df is None:
                    train_df = global_df1.current_df
                else:
                    train_df = global_df1.outlier_treated_df
            except Exception as e:
                print("error", e)

            # print("df: ",df)
            # print("train_df: ",train_df)
            # print("type of traindfdfd: ", type(train_df))
            target_df = train_df.reset_index()[[date_column, column_name]]
            # print("target_df: ",target_df)
            modified_df = df.reset_index()

            if method == "iqr":
                Q1 = target_df[column_name].quantile(0.25)
                Q3 = target_df[column_name].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                modified_df["modified"] = modified_df[column_name].clip(
                    lower=lower_bound, upper=upper_bound
                )
                modified_df = modified_df.rename(columns={f"{column_name}": "actual"})
                # modified_df['actual'] = modified_df[column_name]
                global_df1.forecasted_outlier_df = modified_df
                global_df1.current_df = modified_df
                # modified_df.to_csv(constants.INTERMEDIATE_DF_PATH + "final.csv", index=False)
                return "Outlier treated using IQR method"
                # return modified_df

            elif method == "zscore":
                mean = train_df[column_name].mean()
                std = train_df[column_name].std()
                z_scores = (df[column_name] - mean) / std

                modified_df[column_name] = modified_df[column_name].where(
                    np.abs(z_scores) <= threshold, mean
                )
                modified_df = modified_df.rename(columns={f"{column_name}": "actual"})
                global_df1.forecasted_outlier_df = modified_df
                global_df1.current_df = modified_df
                # modified_df.to_csv(constants.INTERMEDIATE_DF_PATH + "final.csv", index=False)
                return "Outlier treated using Z-score method"
                # return modified_df

            else:
                raise ValueError("Method must be either 'iqr' or 'zscore'")

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(1)  # Wait a bit before retrying
            else:
                print("All attempts failed.")


def outlier_treatment_mean_for_forecasting(
    date_column, column_name, method="iqr", threshold=3
):
    """
    **OUTLIER TREATMENT METHOD**
    Treat outliers in a pandas DataFrame or Series using the specified method (IQR or Z-score),
    and replace the outliers with the mean of the respective column/series.

    Parameters:
    - it takes date column name as an input.
    - it takes target column name (column used for forecasting) for Outlier Treatment as input.
    - method (str): The method to use for outlier detection. Options are:
      - "iqr" : Use Interquartile Range (IQR) method.
      - "zscore" : Use Z-score method.
    - threshold (float): For the Z-score method, the threshold above which data points are considered outliers.
      Default is 3. For the IQR method, this parameter is ignored.

    """
    max_retries = constants.MAX_RETRIES
    for attempt in range(max_retries):
        try:
            try:
                if global_df1.missing_values_treated_df is None:
                    df = global_df1.grouped_dates_df
                else:
                    df = global_df1.missing_values_treated_df
            except Exception as e:
                print("error", e)
            try:
                # train_df = pd.read_csv(csv_train_path)
                if global_df1.outlier_treated_df is None:
                    train_df = global_df1.current_df
                else:
                    train_df = global_df1.outlier_treated_df
            except Exception as e:
                print("error", e)

            # print("df: ",df)
            # print("train_df: ",train_df)
            # print("type of traindfdfd: ", type(train_df))
            target_df = train_df.reset_index()[[date_column, column_name]]
            modified_df = df.reset_index()
            # print("target_df: ",target_df)
            if method == "iqr":
                Q1 = target_df[column_name].quantile(0.25)
                Q3 = target_df[column_name].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                for column_name in target_df.select_dtypes(include=["number"]).columns:
                    column_mean = target_df[column_name].mean()
                    modified_df["modified"] = modified_df[column_name].apply(
                        lambda x: column_mean if (x < lower_bound or x > upper_bound) else x
                    )
                modified_df = modified_df.rename(columns={f"{column_name}": "actual"})
                global_df1.forecasted_outlier_df = modified_df
                global_df1.current_df = modified_df
                # modified_df.to_csv(constants.INTERMEDIATE_DF_PATH + "final.csv", index=False)
                return "Outlier treated by replacing with mean using IQR method"
                # return modified_df

            elif method == "zscore":
                mean = train_df[column_name].mean()
                std = train_df[column_name].std()

                column_mean = train_df[column_name].mean()
                modified_df["modified"] = modified_df[column_name].apply(
                    lambda x: column_mean if np.abs((x - mean) / std) > threshold else x
                )
                modified_df = modified_df.rename(columns={f"{column_name}": "actual"})
                global_df1.forecasted_outlier_df = modified_df
                global_df1.current_df = modified_df
                # modified_df.to_csv(constants.INTERMEDIATE_DF_PATH + "final.csv", index=False)
                return "Outlier treated by replacing with mean using Z-score method"
                # return modified_df
            else:
                raise ValueError("Method must be either 'iqr' or 'zscore'")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(1)  # Wait a bit before retrying
            else:
                print("All attempts failed.")

def add_forecasted_dates(df, date_column, forecasted_dates):
    """
    Adds rows with forecasted dates and NaN values for all other columns.

    Parameters:
        df: pandas.DataFrame
            The original DataFrame containing the time series data.
        date_column: str
            The name of the column containing the date or timestamp values.
        forecasted_dates: int
            The number of forecasted rows to add to the DataFrame.

    Returns:
        pandas.DataFrame
            A new DataFrame with forecasted rows added, with NaN values for all columns except the date column.
    """

    df[date_column] = pd.to_datetime(df[date_column])
    freq = pd.infer_freq(df[date_column])

    if freq is None:
        raise ValueError("Cannot infer frequency from the date column. Ensure the date column has a consistent frequency.")

    last_date = df[date_column].iloc[-1]
    forecasted_dates_list = pd.to_datetime(pd.date_range(start=last_date, periods=forecasted_dates + 1, freq=freq))
    # print(forecasted_dates_list)
    # print(type(forecasted_dates_list))
    # forecasted_df = pd.DataFrame({date_column: forecasted_dates_list})
    # for col in df.columns:
    #     if col != date_column:
    #         forecasted_df[col] = np.nan
    return forecasted_dates_list

@tool
def forecast_time_series_steps(date_column, column_name, model_name, params, forecast_steps, tsid_column_name, treatment_method, detection_method, threshold = 3):
    """
    Parameters:

    date_column: str - The column containing date or timestamp values.
    column_name: str - The column with time series data.
    model_name: str - The model to use for forecasting.
    params: dict - Model-specific parameters:
    For ARIMA/SARIMA: "order", "seasonal_order".
    For Holt-Winters/Exponential Smoothing: "seasonal", "trend", "seasonal_periods".
    forecast_steps: int - Number of steps to forecast.
    tsid_column_name: str - Product ID for outlier forecasting.
    detection_method: str - Outlier detection method ("iqr" or "zscore").
    treatment_method: str - Outlier treatment ("mean" or "bound").
    threshold: float - Z-score threshold for outliers.
    """
    max_retries = constants.MAX_RETRIES
    for attempt in range(max_retries):
        try:
            if treatment_method == "mean":
                _ = outlier_treatment_mean_for_forecasting(date_column, column_name, detection_method, threshold)
            elif treatment_method == "bound":
                _ = outlier_treatment_bound_for_forecasting(date_column, column_name, detection_method, threshold)
            try:
                # df = pd.read_csv(csv_path)
                if global_df1.forecasted_outlier_df is None:
                    df = global_df1.current_df
                else:
                    df = global_df1.forecasted_outlier_df
            except Exception as e:
                print("error", e)
            df = df.drop(tsid_column_name, axis=1)
            # print('final forecasting: df.dtype', df.dtypes)
            # df.reset_index(inplace=True)
            # df.set_index(date_column, inplace=True)
            # result_df = pd.DataFrame(index=df.index)
            # result_df['actual'] = df['actual']
            forecast_date_list = add_forecasted_dates(df, date_column, forecast_steps)
            # outlier_not_invoked = False
            # if 'modified' not in df.columns:
            #     # print("Outlier function was not invoked. Duplicating target column into modified column")
            #     df['actual'] = df[column_name]
            #     df['modified'] = df[column_name]
            #     outlier_not_invoked = True
            # print(forecast_date_list)


            if model_name == "ARIMA":
                model = ARIMA(df["modified"], order=params["order"])
                model_fit = model.fit()
                df["fitted"] = model_fit.fittedvalues
                forecast = model_fit.forecast(steps=forecast_steps)
                # print("This is the forecasted df: ",forecast)
                # print("This is the forecasting index asdf: ", forecast.index)
                # if outlier_not_invoked:
                #     data = {'forecasted': forecast.values,
                #             f'{date_column}': forecast_date_list}
                #     forecast_df = pd.DataFrame(data)
                #     forecast_df.set_index(date_column, inplace=True)
                # else:
                forecast_df = pd.DataFrame(
                    forecast.values, index=forecast.index, columns=["forecasted"]
                )
                df = pd.concat([df, forecast_df])

            elif model_name == "SARIMA":
                model = SARIMAX(
                    df["modified"],
                    order=params["order"],
                    seasonal_order=params["seasonal_order"],
                )
                model_fit = model.fit()
                df["fitted"] = model_fit.fittedvalues
                forecast = model_fit.forecast(steps=forecast_steps)
                # print("This is the forecasted df: ",forecast)
                # print("This is the forecasting index asdf: ", forecast.index)
                # if outlier_not_invoked:
                #     data = {'forecasted': forecast.values,
                #             f'{date_column}': forecast_date_list}
                #     forecast_df = pd.DataFrame(data)
                #     forecast_df.set_index(date_column, inplace=True)
                # else:
                forecast_df = pd.DataFrame(
                    forecast.values, index=forecast.index, columns=["forecasted"]
                )
                df = pd.concat([df, forecast_df])

            elif model_name == "Holt-Winters":
                seasonal_periods = params["seasonal_periods"]
                if isinstance(seasonal_periods, list):
                    if len(df) >= 2 * seasonal_periods[0] and seasonal_periods[0] > 1:
                        model = ExponentialSmoothing(
                            df["modified"],
                            seasonal="add",
                            trend="add",
                            seasonal_periods=seasonal_periods,
                        )
                    else:
                        model = ExponentialSmoothing(
                            df["modified"], seasonal=None, trend="add"
                    )
                else:
                    if isinstance(seasonal_periods, list):
                        if len(df) >= 2 * seasonal_periods[0] and seasonal_periods[0] > 1:
                            model = ExponentialSmoothing(
                                df["modified"],
                                seasonal="add",
                                trend="add",
                                seasonal_periods=seasonal_periods,
                            )
                        else:
                            model = ExponentialSmoothing(
                                df["modified"], seasonal=None, trend="add"
                        )
                    elif isinstance(seasonal_periods, int):
                        if len(df) >= 2 * seasonal_periods and seasonal_periods > 1:
                            model = ExponentialSmoothing(
                                df["modified"], seasonal="add", trend="add", seasonal_periods=seasonal_periods
                            )
                        else:
                            model = ExponentialSmoothing(df["modified"], seasonal=None, trend="add")

                model_fit = model.fit()
                df["fitted"] = model_fit.fittedvalues
                forecast = model_fit.forecast(steps=forecast_steps)
                # print("This is the forecastasdf: ",forecast)
                # print("This is the forecasting index asdf: ", forecast.index)
                # if outlier_not_invoked:
                #     data = {'forecasted': forecast.values,
                #             f'{date_column}': forecast_date_list}
                #     forecast_df = pd.DataFrame(data)
                #     forecast_df.set_index(date_column, inplace=True)
                # else:
                forecast_df = pd.DataFrame(
                    forecast.values, index=forecast.index, columns=["forecasted"]
                )
                df = pd.concat([df, forecast_df])

            elif model_name == "Single Exponential Smoothing":
                model = SimpleExpSmoothing(df["modified"])
                model_fit = model.fit()
                df["fitted"] = model_fit.fittedvalues
                forecast = model_fit.forecast(steps=forecast_steps)
                # print("This is the forecastasdf: ",forecast)
                # print("This is the forecasting index asdf: ", forecast.index)
                # if outlier_not_invoked:
                #     data = {'forecasted': forecast.values,
                #             f'{date_column}': forecast_date_list}
                #     forecast_df = pd.DataFrame(data)
                #     forecast_df.set_index(date_column, inplace=True)
                # else:
                forecast_df = pd.DataFrame(
                    forecast.values, index=forecast.index, columns=["forecasted"]
                )
                df = pd.concat([df, forecast_df])

            else:
                raise ValueError("Model not supported")

            df["fitted"] = df["fitted"].reindex(df.index).fillna(np.nan)
            df["forecasted"] = df["forecasted"].reindex(forecast_df.index).fillna(np.nan)

            # fig = go.Figure()
            # fig.add_trace(go.Scatter(x=df[date_column], y=df["actual"], mode='lines', name='Actual'))
            # fig.add_trace(go.Scatter(x=df[date_column], y=df["modified"], mode='lines', name="Outlier treated data"))
            # # fig.add_trace(go.Scatter(x=df[date_column], y=df["fitted"], mode='lines', name="Fitted"))
            # # fig.add_trace(go.Scatter(x=df[date_column], y=df["forecasted"], mode='lines', name="Forecasted"))
            # fig.update_layout(
            #     title="Time Series Forecasting (Actual, Modified)",
            #     xaxis_title="Date",
            #     yaxis_title="Values",
            #     legend_title="Legend",
            #     hovermode="x unified"
            # )
            # fig_json = json.loads(fig.to_json())
            # print("fomal forecast df: ", df)
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=global_df1.typecasted_df[date_column],
                y=global_df1.typecasted_df[column_name],
                mode='lines',
                name='Original',
                line=dict(color='grey')
            ))

            if global_df1.missing_values_treated_df is None:
                missing_values_plot_df = global_df1.missing_values_detection_df
            else:
                missing_values_plot_df = global_df1.missing_values_treated_df

            fig.add_trace(go.Scatter(
                x=missing_values_plot_df[date_column],
                y=missing_values_plot_df[column_name],
                mode='lines',
                name='Missing Value Treated'
            ))

            fig.add_trace(go.Scatter(
                x=df[date_column],
                y=df["modified"],
                mode='lines',
                name="Missing Value & Outlier treated",
            ))

            fig.add_trace(go.Scatter(
                x=df[date_column],
                y=df["fitted"],
                mode='lines',
                name="Fitted Values",
                line=dict(color='blue')
            ))

            fig.add_trace(go.Scatter(
                x=global_df1.validation_df[date_column],
                y=global_df1.validation_df[column_name],
                mode='lines',
                name="Test",
                line=dict(color='red')
            ))

            # print("plot forecast x axis", forecast_date_list)
            fig.add_trace(go.Scatter(
                x=forecast_date_list.values,
                # x=['2022-07-11', '2022-07-18', '2022-07-25', '2022-08-01', '2022-08-08'],
                # y=df["forecasted"],
                y=forecast,
                mode='lines',
                name="Forecast",
                line=dict(color='green')
            ))

            fig.update_layout(
                title="Forecast",
                xaxis_title="Date",
                yaxis_title="Values",
                legend_title="Legend",
                hovermode="x unified",
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot area
                font=dict(color='white'),  # White font color for dark theme
                xaxis=dict(
                    gridcolor='gray',  # Grid color for better visibility
                    linecolor='white',  # Axis line color
                    zerolinecolor='gray'  # Zero line color
                ),
                yaxis=dict(
                    gridcolor='gray',  # Grid color for better visibility
                    linecolor='white',  # Axis line color
                    zerolinecolor='gray'  # Zero line color
                ),
                legend=dict(
                    bgcolor='rgba(0,0,0,0)',  # Transparent legend background
                    bordercolor='white'  # Legend border color
                ),
                updatemenus=[
                    {
                        "buttons": [
                            {
                                "args": [{"xaxis.gridcolor": 'gray', "yaxis.gridcolor": 'gray'}],
                                "label": "Show Grid",
                                "method": "relayout"
                            },
                            {
                                "args": [{"xaxis.gridcolor": 'rgba(0,0,0,0)', "yaxis.gridcolor": 'rgba(0,0,0,0)'}],
                                "label": "Hide Grid",
                                "method": "relayout"
                            }
                        ],
                        "direction": "down",
                        "showactive": True,
                        "x": 1,
                        "xanchor": "right",
                        "y": 1.1,
                        "yanchor": "top"
                    }
                ]
            )

            fig_json = json.loads(fig.to_json())
            # print("Final forecast df\n\n", df)

            with open(constants.GRAPH_FLAG_PATH + "forecasting_modified.txt", "w") as f:
                f.write(json.dumps(fig_json))
                f.close()

            if not os.path.exists(constants.GRAPH_FLAG_PATH + "graph_flag.txt"):
                open(constants.GRAPH_FLAG_PATH + "graph_flag.txt", 'w').close()

            with open(constants.GRAPH_FLAG_PATH + "graph_flag.txt", "r+") as f:
                lines = f.readlines()
                if not lines:
                    lines.append("1\n")
                    lines.append("forecasting_modified.txt\n")
                else:
                    flag = True
                    while flag:
                        f.seek(0)
                        lines = f.readlines()
                        if int(lines[0]) == 1:
                            time.sleep(2)
                            continue
                        first_line = 1
                        lines[0] = f"{first_line}\n"
                        lines.append("forecasting_modified.txt\n")
                        flag = False
                f.seek(0)
                f.writelines(lines)
                f.close()

            global_df1.final_forecast_df = df
            global_df1.current_df = df
            # df.to_csv(constants.INTERMEDIATE_DF_PATH + "forecasted_values.csv", index=False)

            return str(forecast)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(1)  # Wait a bit before retrying
            else:
                print("All attempts failed.")

from langchain.tools import tool
import pandas as pd
import constants
import numpy as np
from langchain_core.messages import HumanMessage
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from src.mper.agent_template import AgentTemplate

global_data: pd.DataFrame | None = None

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

@tool
def forecast_time_series(model_name, train, test, params):
    # Split data into train and test

    if model_name == "ARIMA":
        model = ARIMA(train, order=params["order"])
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test))

    elif model_name == "SARIMA":
        model = SARIMAX(
            train, order=params["order"], seasonal_order=params["seasonal_order"]
        )
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test))

    elif model_name == "Holt-Winters":
        model = ExponentialSmoothing(
            train, seasonal=params["seasonal"], trend=params["trend"]
        )
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test))

    elif model_name == "Prophet":
        df = pd.DataFrame({"ds": train.index, "y": train.values})
        model = Prophet(yearly_seasonality=params["yearly_seasonality"])
        model.fit(df)
        future = model.make_future_dataframe(periods=len(test))
        forecast = model.predict(future)["yhat"][-len(test) :]

    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=params["n_estimators"])
        model.fit(np.arange(len(train)).reshape(-1, 1), train.values)
        forecast = model.predict(
            np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
        )

    elif model_name == "XGBoost":
        model = XGBRegressor(n_estimators=params["n_estimators"])
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

from pydantic import BaseModel, Field
from typing import Literal


class SupervisorOutput(BaseModel):
    next_node: Literal[
        "DataframeInfoExtraction",
        # "END",
        "OutlierDetection",
        # "ADF_test"
        "StationarityTest",
        "SeasonalityTest",
        "ModelSelection",
        "Validation",
        "model_fiting",
        "forecasting"
    ] | None = Field(description ="The next node that the supervisor should run", default = None)
    explanation: str | None = Field(description = "The explanation for choosing the next node.", default = None)



class DuplicateRemovalOutput(BaseModel):
    column_name: str | None = Field(description = "The column name to remove duplicates from", default=None)


class OutlierDetetction(BaseModel):
    outliers: str | None = Field(description = "The target column name for forecasting to Detect Outliers from", default=None)



class ModelSelectionOutput(BaseModel):
    model: list[str] | None = Field(description = "The models selected for forecasting", default=None)
    justification: str | None = Field(description = "The justification for selecting the model", default=None)


class HumanInterrupt(BaseModel):
    flag: bool = Field(description = "The flag to indicate if the user has approved the process or want to rerun the process with some changes. True means that the user has suggested some changes which will result in restart of the graph. False means that the user is satisfied with the results and wants to end the graph.", default = True)
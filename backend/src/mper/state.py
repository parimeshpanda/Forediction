from typing import Optional, List, Annotated
from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import pandas as pd


class GlobalGraphState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages] = []
    supervisor_history: Annotated[List[BaseMessage], add_messages] = []
    previous_episode_supervisor_history: list[BaseMessage] = []
    user_input: str = ""
    final_outcome: str = ""
    # data: Optional[pd.DataFrame] = None
    data: Optional[bool] = None
    path_to_csv: str = ""
    next_node: Optional[str] = None
    data_info: Optional[str] = None
    column_names: Optional[List[str]] = None
    model: Optional[List[str]] = None
    model_justification: Optional[str] = None
    human_response: Optional[bool] = None
    run_counter: int = 0
    # validation_flag: bool = False
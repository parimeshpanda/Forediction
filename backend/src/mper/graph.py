from src.mper.state import GlobalGraphState
from src.mper.agents import supervisor_node, info_extractor, outlier_detetcion,seasonality_check, stationarity_test, model_selection, validation_node, model_fiting, forecasting, human_interrupt
import constants
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

MEMBER_OPTIONS = {'DataframeInfoExtraction': 'DataframeInfoExtraction',
#  "END": END,
 "OutlierDetection": "OutlierDetection",
 "StationarityTest": "StationarityTest",
 "SeasonalityTest": "SeasonalityTest",
 "ModelSelection": "ModelSelection",
 "Validation": "Validation",
 "model_fiting": "model_fiting",
 "forecasting": "forecasting"
}

memory = MemorySaver()

def user_restart_condition(state: GlobalGraphState):
    if state.human_response == True:
        return "restart"
    return "end"

def supervisor_condition(state: GlobalGraphState):
    # print("IN CONDITIONAL EDGE\n next node:", state.next_node)
    return state.next_node

graph = StateGraph(GlobalGraphState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("DataframeInfoExtraction", info_extractor)
graph.add_node("OutlierDetection", outlier_detetcion)
# graph.add_node("ADF_test", ADF_test)
graph.add_node("StationarityTest", stationarity_test)
graph.add_node("SeasonalityTest", seasonality_check)
graph.add_node("ModelSelection", model_selection)
graph.add_node("Validation", validation_node)
graph.add_node("model_fiting", model_fiting)
graph.add_node("forecasting", forecasting)
graph.add_node("human_interrupt", human_interrupt)
graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", supervisor_condition, MEMBER_OPTIONS)
# graph.add_conditional_edges("supervisor", lambda state: state.validation_flag)
graph.add_edge("DataframeInfoExtraction", "supervisor")
graph.add_edge("OutlierDetection", "supervisor")
graph.add_edge("StationarityTest", "supervisor")
graph.add_edge("SeasonalityTest", "supervisor")
graph.add_edge("ModelSelection", "supervisor")
graph.add_edge("Validation", "supervisor")
graph.add_edge("model_fiting", "supervisor")
graph.add_edge("forecasting", "human_interrupt")
graph.add_conditional_edges("human_interrupt",user_restart_condition, {"restart": "supervisor", "end": END } )

compiled_graph = graph.compile(interrupt_before=["human_interrupt"], checkpointer=memory)
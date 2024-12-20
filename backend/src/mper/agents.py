from langchain_core.messages import HumanMessage, RemoveMessage

from src.mper.agent_template import AgentTemplate
from src.mper.state import GlobalGraphState
from src.mper.tools import (
    data_insights,
    normalisation_insights,
    stationarity_ADF,
    ljung_box_test,
    outlier_treatment_bound,
    outlier_treatment_mean,
    type_cast_columns,
    grouping_same_dates,
    missing_values_treatment,
    time_interval_missing,
    train_test_validation_split_df,
    get_fft,
    stationarity_kpss,
    optimize_hyperparameters,
    evaluate_model,
    acf_pacf_insights,
    outlier_treatment_bound_for_forecasting,
    outlier_treatment_mean_for_forecasting,
    forecast_time_series_steps,
)
from src.mper.agent_template import AgentTemplate
from src.mper.prompt import (
    supervisor_prompt,
    supervisor_prompt_rerun,
    info_extractor_prompt,
    outlier_sys_prompt,
    ADF_test_sys_prompt,
    seasonality_agent_prompt,
    stationarity_agent_prompt,
    model_selection_prompt,
    validation_prompt,
    model_fiting_sys_prompt,
    forecasting_sys_prompt,
    supervisor_replanning_prompt,
)
from src.mper.output_parsers import (
    SupervisorOutput,
    ModelSelectionOutput,
    HumanInterrupt,
)
import constants

from langchain_core.messages import AIMessage
import pandas as pd


MEMBERS = [
    "DataframeInfoExtraction",
    "OutlierDetection",
    "StationarityTest",
    "SeasonalityTest",
    "ModelSelection",
    "Validation",
    "model_fiting",
    "forecasting",
]


def supervisor_node(state: GlobalGraphState):
    if len(state.messages) > 0:
        llm_w_struct = AgentTemplate.llm.with_structured_output(SupervisorOutput)
        if state.run_counter == 0:
            chain = supervisor_prompt_rerun | llm_w_struct
            res = chain.invoke(
                {
                    "chat_history": state.messages,
                    "user_input": state.user_input,
                    "members": MEMBERS,
                }
            )
            return {
                "next_node": res.next_node,
                "messages": [
                    AIMessage(
                        content=f"The next node to visit is: {res.next_node} \njustification for it is as follows: {res.explanation}"
                    )
                ],
                "supervisor_history": [
                    AIMessage(
                        content=f"The next node to visit is: {res.next_node} \njustification for it is as follows: {res.explanation}"
                    )
                ],
            }
        else:
            ## logic for replanning

            chain = supervisor_prompt_rerun | llm_w_struct
            # if len(state.messages) >=1:
            #     print("last worker output: ", state.messages[-1])
            res = chain.invoke({"user_input": state.user_input, "chat_history": state.messages, "prev_worker_outputs": state.messages[-1], "members": MEMBERS})
            # print("length of supervisor history: ",len(state.supervisor_history))
            # return {"next_node": res.next_node}
            # print("result next node: ",res.next_node)
            # print("result explanation: ",res.explanation)
            # print("current state's next node: ", state.next_node)
            return {
                "next_node": res.next_node,
                "messages": [
                    AIMessage(
                        content=f"The next node to visit is: {res.next_node} \njustification for it is as follows: {res.explanation}"
                    )
                ],
                "supervisor_history": [
                    AIMessage(
                        content=f"The next node to visit is: {res.next_node} \njustification for it is as follows: {res.explanation}"
                    )
                ],
            }

    llm_w_struct = AgentTemplate.llm.with_structured_output(SupervisorOutput)
    chain = supervisor_prompt | llm_w_struct
    # print("askljdghfajsdfgh", state.user_input)
    res = chain.invoke({"user_input": state.user_input, "members": MEMBERS})
    # print(res)
    # return {"next_node": res.next_node}
    return {
        "next_node": res.next_node,
        "messages": [
            AIMessage(
                content=f"The next node to visit is: {res.next_node} \njustification for it is as follows: {res.explanation}"
            )
        ],
        "supervisor_history": [
            AIMessage(
                content=f"The next node to visit is: {res.next_node} \njustification for it is as follows: {res.explanation}"
            )
        ],
    }


def info_extractor(state: GlobalGraphState):
    agent_template = AgentTemplate(
        system_prompt=info_extractor_prompt,
        toolkit=[
            data_insights,
            normalisation_insights,
            type_cast_columns,
            grouping_same_dates,
            missing_values_treatment,
            time_interval_missing,
            train_test_validation_split_df,
        ],
        output_parser=None,
    )
    agent = agent_template.create_agent_with_tools()
    # print(state.user_input)
    res = agent.invoke({"chat_history": state.messages, "user_input": state.user_input})
    # print("Initial RES: ", res)
    if state.data is None:
        csv_path = res["intermediate_steps"][0][0].tool_input["csv_path"]
        try:
            df = pd.read_csv(csv_path)
            path_to_csv = csv_path
            columns = list(df.columns)
            return {
                "messages": [AIMessage(content=res["output"])],
                "data": True,
                "path_to_csv": path_to_csv,
                "column_names": columns,
            }
        except Exception as e:
            print(e)
            res = "Invalid Path"
    else:
        # print(res)
        # csv_path = constants.INTERMEDIATE_DF_PATH + "intermediate.csv"
        csv_path = res["intermediate_steps"][0][0].tool_input["csv_path"]
        try:
            df = pd.read_csv(csv_path)
            path_to_csv = csv_path
            return {
                "messages": [AIMessage(content=res["output"])],
                "data": True,
                "path_to_csv": path_to_csv,
            }
        except Exception as e:
            print(e)
            res = "Invalid Path"
    # print("RES: ", res)
    # print("RES type: ", type(res))
    return {"messages": [AIMessage(content=res["output"])]}


# def duplicate_data_removal(state: GlobalGraphState):
#     """
#     Removes duplicate rows from dataframe. Also shows the number of duplicated rows that were removed.
#     """
#     df = pd.read_csv(constants.INTERMEDIATE_DF_PATH + "intermediate.csv")
#     res_df = df.drop_duplicates().reset_index()
#     res_df.to_csv(constants.INTERMEDIATE_DF_PATH + "intermediate.csv")
#     num_duplicates = df.duplicated().sum() - res_df.duplicated().sum()
#     return {
#         "data": res_df,
#         "messages": [
#             AIMessage(
#                 content=f"Performed removal of duplicated data, {num_duplicates} rows were removed."
#             )
#         ],
#     }


def outlier_detetcion(state: GlobalGraphState):

    agent_template = AgentTemplate(
        system_prompt=outlier_sys_prompt,
        toolkit=[outlier_treatment_bound, outlier_treatment_mean],
        output_parser=None,
    )
    # print(state.messages)
    agent = agent_template.create_agent_with_tools()
    res = agent.invoke({"chat_history": state.messages, "user_input": state.user_input})
    return {"messages": [AIMessage(content=res["output"])]}


def ADF_test(state: GlobalGraphState):

    agent_template = AgentTemplate(
        system_prompt=ADF_test_sys_prompt,
        toolkit=[stationarity_ADF],
        output_parser=None,
    )
    # print(state.messages)
    agent = agent_template.create_agent_with_tools()
    res = agent.invoke({"user_input": state.messages})
    return {"messages": [AIMessage(content=res["output"])]}


def stationarity_test(state: GlobalGraphState):
    agent_template = AgentTemplate(
        system_prompt=stationarity_agent_prompt,
        toolkit=[stationarity_ADF, stationarity_kpss],
        output_parser=None,
    )
    agent = agent_template.create_agent_with_tools()
    res = agent.invoke({"user_input": state.user_input, "chat_history": state.messages})
    return {"messages": [AIMessage(content=res["output"])]}


def seasonality_check(state: GlobalGraphState):
    agent_template = AgentTemplate(
        system_prompt=seasonality_agent_prompt,
        toolkit=[ljung_box_test, get_fft, acf_pacf_insights],
        output_parser=None,
    )
    agent = agent_template.create_agent_with_tools()
    res = agent.invoke({"user_input": state.user_input, "chat_history": state.messages})
    return {"messages": [AIMessage(content=res["output"])]}


def model_selection(state: GlobalGraphState):
    # print("INSIDE MODEL SELECTION!!!!")
    llm_w_struct = AgentTemplate.llm.with_structured_output(ModelSelectionOutput)

    chain = model_selection_prompt | llm_w_struct
    res = chain.invoke({"user_input": state.user_input, "chat_history": state.messages})
    # print("Model Selection Output: ", res)
    # print(res)
    return {
        "messages": [
            AIMessage(
                content=f"The model has been selected. The model selected is {res.model} and the justification for selecting the mode is, {res.justification}"
            )
        ],
        "model": res.model,
        "model_justification": res.justification,
    }


def validation_node(state: GlobalGraphState):
    chain = validation_prompt | AgentTemplate.llm
    res = chain.invoke({"user_input": state.user_input, "chat_history": state.messages})
    # print("validation response:",res)
    return {"messages": [AIMessage(content=f"Validation response is {res}")]}


def model_fiting(state: GlobalGraphState):
    agent_template = AgentTemplate(
        system_prompt=model_fiting_sys_prompt,  ## changes to be done in the prompt
        toolkit=[optimize_hyperparameters, evaluate_model],
        output_parser=None,
    )
    agent = agent_template.create_agent_with_tools()
    res = agent.invoke({"user_input": state.user_input, "chat_history": state.messages})
    return {"messages": [AIMessage(content=res["output"])]}


def forecasting(state: GlobalGraphState):
    agent_template = AgentTemplate(
        system_prompt=forecasting_sys_prompt,
        toolkit=[
            forecast_time_series_steps,
        ],
        output_parser=None,
    )
    agent = agent_template.create_agent_with_tools()
    res = agent.invoke({"user_input": state.user_input, "chat_history": state.messages})
    return {
        "messages": [AIMessage(content=res["output"])],
        "run_counter": state.run_counter + 1,
    }


def human_interrupt(state: GlobalGraphState):
    # user_input=str(input("Enter"))
    user_input = state.user_input
    llm_w_structured_output = AgentTemplate.llm.with_structured_output(HumanInterrupt)
    prompt = f"""You are an information collector. Based on the following question and answer fill the fields:
    Based on the user's response, Would you like to re-run the graph with additional user inputs or would you like to end the graph if the user is satisfied?
    User response: {user_input}"""

    res = llm_w_structured_output.invoke(prompt)
    # print("THIS is the res of human interrupt", res)
    if res.flag:
        supervisor_history = state.supervisor_history
        state_messages = state.messages
        return {
            "human_response": res.flag,
            "previous_episode_supervisor_history": supervisor_history,
            "messages": [RemoveMessage(id=m.id) for m in state_messages],
            "supervisor_history": [RemoveMessage(id=m.id) for m in supervisor_history]
        }

    return {
        "human_response": res.flag,
        "messages": [AIMessage(content="User is satisfied with the forecast results")],
        "previous_episode_supervisor_history":state.supervisor_history
    }

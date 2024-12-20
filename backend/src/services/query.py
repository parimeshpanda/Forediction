import uuid

from fastapi import BackgroundTasks
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command
from langchain_community.callbacks import OpenAICallbackHandler, get_openai_callback

from src.schemas.query import UserQueryRequest, UserQueryResponse
from src.mper.agent_template import AgentTemplate
from src.mper.graph import compiled_graph


async def user_query_service(user_query: UserQueryRequest):
    """
    Service to ask a question
    """
    prompt_str = "Given the user query regarding forcasting, your task is to generate a response that acknowledges the query and directs the user to view the dashboard for further insights."
    prompt = ChatPromptTemplate(
    [("system", prompt_str), ("human", "user_input: {user_input}")]
    )
    chain = prompt | AgentTemplate.llm
    res = chain.astream_events({"user_input": user_query.query}, version="v2")
    async for event in res:
        kind = event["event"]
        if kind == "on_chat_model_stream":
            # print(event)
            yield {
                "event": event["data"]["chunk"].content,
            }

callback_handler = get_openai_callback()

async def main_graph_run(user_query: str, thread_config: dict | None):
    # with callback_handler as cb:
    graph_state = compiled_graph.get_state(config=thread_config)
    # print("hello state: ", graph_state)
    # print("hello")
    # compiled_graph.update_state()
    # print("This is the graph state of the graph:", graph_state)
    # run_counter = graph_state.values["run_counter"]
    if graph_state.values:
        # print("yo yo")
        prev_user_input = graph_state.values["user_input"]
        # print("Previous User Input: ", prev_user_input)
        new_user_input = prev_user_input + user_query
        # print("New User Input: ", new_user_input)
        # print("test yoyo")
        compiled_graph.update_state(config=thread_config, values={"user_input": new_user_input})
        # print("heylo")
        async for s in compiled_graph.astream_events(
        None,
        version="v2",
        config=thread_config
        # callbacks=[callback_handler]
        ):
            if s["event"] == "on_chat_model_stream":
                c = s["data"]["chunk"].content
                if c:
                    yield c
        yield "RESPONSE_GENERATION_COMPLETED"

    else:
        # print("hello again")
        async for s in compiled_graph.astream_events(
            {
                # "user_input": r"The following is the path to CSV: C:\Users\703398727\Downloads\sample_Timeseries_1 1.csv. The time series column is Sum of Ord Lbs, date column is Date and time series identifier column is Product_id"
                # "user_input": r"The following is the path to CSV: src\original_df\sample_Timeseries_1 1.csv. The time series column is Sum of Ord Lbs, date column is Date and time series identifier column is Product_id"
                "user_input": rf"{user_query}"
            },
            version="v2",
            config=thread_config
            # callbacks=[callback_handler]
        ):
            if s["event"] == "on_chat_model_stream":
                c = s["data"]["chunk"].content
                if c:
                    yield c
        yield "RESPONSE_GENERATION_COMPLETED"
            # if "__end__" not in s:
            #     print("Event: ", s)
            #     print("----")
        # print(f"Total Tokens: {cb.total_tokens}")
        # print(f"Prompt Tokens: {cb.prompt_tokens}")
        # print(f"Completion Tokens: {cb.completion_tokens}")
        # print(f"Total Cost (USD): ${cb.total_cost}")

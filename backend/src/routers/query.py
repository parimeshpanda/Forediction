from fastapi import APIRouter, status, Depends, UploadFile, WebSocket, WebSocketDisconnect
from sse_starlette.sse import EventSourceResponse
import asyncio
from sqlalchemy.orm import Session
import plotly.graph_objects as go
import random
import pandas as pd
import os
import shutil


from src.database.db import get_db
from src.schemas.query import UserQueryRequest
from src.services.query import user_query_service, main_graph_run
import constants

query_router = APIRouter()

@query_router.post("/", description= "Ask a query related to forecasting", status_code=status.HTTP_202_ACCEPTED)
async def user_query_router(user_query: UserQueryRequest, db: Session = Depends(get_db)):
    return EventSourceResponse(main_graph_run(user_query))



# Example function that simulates some work and generates a response over time.
async def my_foo(input_data: str):
    # for i in range(5):  # Simulate generating some output over time
    await asyncio.sleep(1)  # Simulate delay (e.g., processing time)
    yield f"{input_data}"  # Streaming responses



active_tasks = {}

async def process_message(input_data: str, thread_config: dict, websocket: WebSocket):
    # print(thread_config)
    try:
                # Call my_foo with the input data, streaming the result back
        async for response in main_graph_run(input_data, thread_config):
        # async for response in my_foo(input_data):
            # Send each part of the response as it is generated
            await websocket.send_text(response)
    except asyncio.CancelledError:
        print(f"Processing of message '{input_data}' was interrupted.")
        await websocket.send_text(f"Processing of message '{input_data}' was interrupted.")
        raise

@query_router.websocket("/ws/chat")
async def chat(websocket: WebSocket):
    await websocket.accept()
    client_id = id(websocket)
    print('unique client id: ', client_id)
    try:
        while True:
            # Receive data from the WebSocket client
            input_data = await websocket.receive_text()
            print(f"Received message from client {client_id}: {input_data}")
            thread_config =  {"configurable": {"thread_id": client_id}}
            if input_data == "END":
                if os.path.isdir(constants.GRAPH_FLAG_PATH):
                    shutil.rmtree(constants.GRAPH_FLAG_PATH)
                if os.path.isdir(constants.INTERMEDIATE_DF_PATH):
                    shutil.rmtree(constants.INTERMEDIATE_DF_PATH)
                active_tasks[client_id].cancel()
                try:
                    await active_tasks[client_id]
                    continue  # Await to ensure proper cancellation
                except asyncio.CancelledError:
                    print(f"Previous task for client {client_id} cancelled.")
                    continue

            if client_id in active_tasks:
                print(f"Cancelling previous task for client {client_id}")
                active_tasks[client_id].cancel()
                try:
                    await active_tasks[client_id]
                except asyncio.CancelledError:
                    print(f"Previous task for client {client_id} cancelled.")

            task = asyncio.create_task(process_message(input_data, thread_config, websocket))
            active_tasks[client_id] = task

    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected.")
        if client_id in active_tasks:
            active_tasks[client_id].cancel()
            try:
                await active_tasks[client_id]
            except asyncio.CancelledError:
                pass
            del active_tasks[client_id]


@query_router.websocket("/ws/outliers")
async def outliers_data(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(6)
            try:
                with open(constants.GRAPH_FLAG_PATH + "graph_flag.txt", "r+") as f:
                    lines = f.readlines()
                    graph_flag = 0
                    if not lines:
                        f.close()
                        await asyncio.sleep(5)
                        continue
                    graph_flag = int(lines[0].strip())
                    if graph_flag == 1:
                        # contents = last_line.split(": ")
                        if lines[-1] == "outliers.txt\n":
                            with open(constants.GRAPH_FLAG_PATH + "outliers.txt", "r") as file:
                                data = file.read()
                                await websocket.send_text(data)
                                file.close()
                            lines[0] = "0\n"
                            f.seek(0)
                            f.writelines(lines)
                            f.close()
                        else:
                            f.close()
                            await asyncio.sleep(5)
                    else:
                        f.close()
                        await asyncio.sleep(5)
            except FileNotFoundError:
                print("File not found")
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(e)


@query_router.websocket("/ws/train_test_data")
async def train_test_split_data(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(6)
            try:
                with open(constants.GRAPH_FLAG_PATH + "graph_flag.txt", "r+") as f:
                    lines = f.readlines()
                    graph_flag = 0
                    if not lines:
                        f.close()
                        await asyncio.sleep(5)
                        continue
                    graph_flag = int(lines[0].strip())
                    if graph_flag == 1:
                        # contents = last_line.split(": ")
                        if lines[-1] == "train_test_split.txt\n":
                            with open(constants.GRAPH_FLAG_PATH + "train_test_split.txt", "r") as file:
                                data = file.read()
                                await websocket.send_text(data)
                                file.close()
                            lines[0] = "0\n"
                            f.seek(0)
                            f.writelines(lines)
                            f.close()
                        else:
                            f.close()
                            await asyncio.sleep(5)
                    else:
                        f.close()
                        await asyncio.sleep(5)
            except FileNotFoundError:
                print("File not found")
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(e)


@query_router.websocket("/ws/timeseries")
async def timeseries(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(6)
            try:
                with open(constants.GRAPH_FLAG_PATH + "graph_flag.txt", "r+") as f:
                    lines = f.readlines()
                    graph_flag = 0
                    if not lines:
                        f.close()
                        await asyncio.sleep(5)
                        continue
                    graph_flag = int(lines[0].strip())
                    if graph_flag == 1:
                        # contents = last_line.split(": ")
                        if lines[-1] == "timeseries_original.txt\n":
                            with open(constants.GRAPH_FLAG_PATH + "timeseries_original.txt", "r") as file:
                                data = file.read()
                                await websocket.send_text(data)
                                file.close()
                            lines[0] = "0\n"
                            f.seek(0)
                            f.writelines(lines)
                            f.close()
                        else:
                            f.close()
                            await asyncio.sleep(5)
                    else:
                        f.close()
                        await asyncio.sleep(5)
            except FileNotFoundError:
                print("File not found")
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(e)


@query_router.websocket("/ws/model_selection")
async def model_selection(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(6)
            try:
                with open(constants.GRAPH_FLAG_PATH + "graph_flag.txt", "r+") as f:
                    lines = f.readlines()
                    graph_flag = 0
                    if not lines:
                        f.close()
                        await asyncio.sleep(5)
                        continue
                    graph_flag = int(lines[0].strip())
                    if graph_flag == 1:
                        # contents = last_line.split(": ")
                        if lines[-1] == "model_selection.txt\n":
                            with open(constants.GRAPH_FLAG_PATH + "model_selection.txt", "r") as file:
                                data = file.read()
                                await websocket.send_text(data)
                                file.close()
                            lines[0] = "0\n"
                            f.seek(0)
                            f.writelines(lines)
                            f.close()
                        else:
                            f.close()
                            await asyncio.sleep(5)
                    else:
                        f.close()
                        await asyncio.sleep(5)
            except FileNotFoundError:
                print("File not found")
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(e)

@query_router.websocket("/ws/forecasting_modified")
async def forecasting_modified(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(6)
            try:
                with open(constants.GRAPH_FLAG_PATH + "graph_flag.txt", "r+") as f:
                    lines = f.readlines()
                    graph_flag = 0
                    if not lines:
                        f.close()
                        await asyncio.sleep(5)
                        continue
                    graph_flag = int(lines[0].strip())
                    if graph_flag == 1:
                        # contents = last_line.split(": ")
                        if lines[-1] == "forecasting_modified.txt\n":
                            with open(constants.GRAPH_FLAG_PATH + "forecasting_modified.txt", "r") as file:
                                data = file.read()
                                await websocket.send_text(data)
                                file.close()
                            lines[0] = "0\n"
                            f.seek(0)
                            f.writelines(lines)
                            f.close()
                        else:
                            f.close()
                            await asyncio.sleep(5)
                    else:
                        f.close()
                        await asyncio.sleep(5)
            except FileNotFoundError:
                print("File not found")
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(e)
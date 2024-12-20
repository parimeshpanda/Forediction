from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routers.ingestion import ingestion_router
from src.routers.query import query_router
import src.mper.global_df
#test
app = FastAPI()

app.title = "MPER-Multistep Planning and Exaplainable Reasoning"
app.description = "MPER API Documentation"
app.version = "1.0.0"
app.openapi_tags = [
    {
        "name": "Default",
        "description": "Default endpoints"
    },
    {
        "name": "Ingestion",
        "description": "CSV Ingestion/Upload endpoint"
    },
    {
        "name": "Query",
        "description": "Query endpoint for user query."
    },
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    ingestion_router,
    prefix="/api/v1/ingestion",
    tags=["Ingestion"],
)

app.include_router(
    query_router,
    prefix="/api/v1/query",
    tags=["Query"],
)

@app.get("/", summary="Check server Health", tags=["Default"])
def index():
    return "Server is up and running!"




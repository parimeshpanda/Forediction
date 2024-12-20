from fastapi import APIRouter, status, Depends, UploadFile
from sqlalchemy.orm import Session

from src.database.db import get_db
from src.schemas.ingestion import FileUploadResponse
from src.services.ingestion import upload_file_service
import constants

ingestion_router = APIRouter()

@ingestion_router.post("/", description= "Ingest a CSV/Excel or link for CSV/Excel",status_code=status.HTTP_201_CREATED)
async def upload_file(file: UploadFile, db: Session = Depends(get_db)) -> FileUploadResponse:
    filename = file.filename
    return await upload_file_service(file, filename, db)
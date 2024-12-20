import os
from fastapi import UploadFile
from sqlalchemy.orm import Session

from src.schemas.ingestion import FileUploadResponse
import constants

async def upload_file_service(file: UploadFile, filename:str, db: Session) -> FileUploadResponse:
    try:
        os.makedirs(constants.ORIGINAL_DF_PATH, exist_ok=True)
        file_location = os.path.join(constants.ORIGINAL_DF_PATH, filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())
        # response = ingestion_crud.upload_file(filename, db)
        response = FileUploadResponse(filename=filename, status="success", message=f"File {filename} uploaded successfully")

    except Exception as e:
        return FileUploadResponse(filename=filename, status="failed", message=f"File {filename} failed to upload due to errir: {e}")
    
    finally:
        await file.close()

    return response
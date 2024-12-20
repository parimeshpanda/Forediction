from pydantic import BaseModel

class FileUploadResponse(BaseModel):
    """
    Response schema for uploading a file
    """
    filename: str
    status: str
    message: str
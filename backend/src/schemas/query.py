from pydantic import BaseModel
from datetime import datetime
import uuid

class UserQueryRequest(BaseModel):
    """
    Request schema for asking a question
    """
    query_timestamp: datetime
    query: str
    first_name: str = "palash"
    last_name: str = "munshi"
    email: str = "palash.munshi@genpact.com"


class UserQueryResponse(BaseModel):
    """
    Response schema for asking a question
    """
    question_id: uuid.UUID
    answer: str
    status: str
    message: str
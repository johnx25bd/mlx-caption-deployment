from pydantic import BaseModel
from typing import List, Union

class QueryRequest(BaseModel):
    query: str  

class DocumentResponse(BaseModel):
    rel_docs: List[str]
    urls: List[str]
    rel_docs_sim: List[Union[float, int]]
    indices: List[int]

class SelectRequest(BaseModel):
    event_type: str
    query: str
    selected_doc_id: int

class CaptionRequest(BaseModel):
    image: bytes

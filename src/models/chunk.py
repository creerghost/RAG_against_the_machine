from pydantic import BaseModel


class Chunk(BaseModel):
    file_path: str
    content: str
    first_char_idx: int
    last_char_idx: int

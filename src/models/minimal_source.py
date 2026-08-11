from pydantic import BaseModel


class MinimalSource(BaseModel):
    file_path: str
    first_char_idx: int
    last_char_idx: int

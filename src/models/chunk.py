from pydantic import BaseModel


class Chunk(BaseModel):
    """Store chunk text together with its exact source-file character span."""

    file_path: str
    content: str
    first_character_index: int
    last_character_index: int

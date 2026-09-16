from pydantic import BaseModel


class MinimalSource(BaseModel):
    """Identify a half-open character span in an indexed source file."""

    file_path: str
    first_character_index: int
    last_character_index: int

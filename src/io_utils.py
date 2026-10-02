"""JSON load/save helpers that turn I/O failures into ``CliError``."""

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

M = TypeVar("M", bound=BaseModel)


class CliError(Exception):
    """An expected, user-facing error: printed without a traceback."""


def load_model(path: Path, model_cls: type[M]) -> M:
    """Return the JSON file ``path`` validated as ``model_cls``.

    Raises ``CliError`` if it is missing, unreadable, not JSON or invalid.
    """
    if not path.is_file():
        raise CliError(f"File not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return model_cls.model_validate(data)
    except json.JSONDecodeError as e:
        raise CliError(f"Malformed JSON in {path}: {e}") from e
    except ValidationError as e:
        name = model_cls.__name__
        raise CliError(f"{path} is not a valid {name}:\n{e}") from e
    except (OSError, UnicodeDecodeError) as e:
        raise CliError(f"Cannot read {path}: {e}") from e


def save_model(model: BaseModel, path: Path) -> None:
    """Write ``model`` to ``path`` as indented JSON, creating parent dirs.

    Raises ``CliError`` if the file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(model.model_dump_json(indent=2))
    except OSError as e:
        raise CliError(f"Cannot write {path}: {e}") from e

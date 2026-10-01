"""JSON load/save helpers that turn I/O failures into ``CliError``."""

import json
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

M = TypeVar("M", bound=BaseModel)


class CliError(Exception):
    """An expected, user-facing error: printed without a traceback."""


def load_model(path: Path, model_cls: Type[M]) -> M:
    """Read ``path`` and validate it as ``model_cls``.

    Args:
        path: JSON file to read.
        model_cls: Pydantic model describing the file.

    Returns:
        M: The validated model.

    Raises:
        CliError: If the file is missing, unreadable, not JSON, or invalid.
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
        raise CliError(
            f"{path} is not a valid {model_cls.__name__}:\n{e}"
        ) from e
    except (OSError, UnicodeDecodeError) as e:
        raise CliError(f"Cannot read {path}: {e}") from e


def save_model(model: BaseModel, path: Path) -> None:
    """Write ``model`` as indented JSON, creating parent directories.

    Args:
        model: Model to serialise.
        path: Destination file.

    Raises:
        CliError: If the file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(model.model_dump_json(indent=2))
    except OSError as e:
        raise CliError(f"Cannot write {path}: {e}") from e

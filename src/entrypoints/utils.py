"""Argument checks, output helpers and index error messages shared by the
CLI and the HTTP server."""

from pathlib import Path
from typing import Any

from src.io_utils import CliError
from src.models import MinimalSource


# errors of a missing, corrupt, old-format or mismatched index
INDEX_ERRORS = (OSError, ValueError, KeyError, AttributeError)


def env_name(loc: tuple[Any, ...]) -> str:
    """Return the environment variable for a config error location."""
    name = "__".join(str(part) for part in loc).upper()
    # unknown .env keys keep their prefix in the location
    return name if name.startswith("RAG_") else "RAG_" + name


def as_text(value: Any, name: str) -> str:
    """Return ``value`` as a non-empty stripped string (Fire may pass ints)."""
    if value is None:
        raise CliError(f"--{name} is required")
    text = str(value).strip()
    if not text:
        raise CliError(f"--{name} must not be empty")
    return text


def as_positive_int(value: Any, name: str, maximum: int = 0) -> int:
    """Return ``value`` if it is an int in ``[1, maximum]`` (0: no maximum)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise CliError(f"--{name} must be an integer, got {value!r}")
    if value < 1:
        raise CliError(f"--{name} must be >= 1, got {value}")
    if maximum and value > maximum:
        raise CliError(f"--{name} must be <= {maximum}, got {value}")
    return value


def as_path(value: Any, name: str) -> Path:
    """Return ``value`` as a ``Path``; it must be a non-empty string."""
    return Path(as_text(value, name))


def print_sources(sources: list[MinimalSource]) -> None:
    """Print one ``path [start:end]`` line per source."""
    if not sources:
        print("No relevant source found.")
    for s in sources:
        print(f"{s.file_path} "
              f"[{s.first_character_index}:{s.last_character_index}]")


def refuse_global_flags(args: list[str]) -> None:
    """Raise ``CliError`` for options before the command (old settings)."""
    for arg in args:
        if not arg.startswith("-"):
            return  # the command: its own options follow
        if arg not in ("-h", "--help"):
            raise CliError(f"{arg.split('=')[0]} before the command: "
                           "settings are not command-line options; set "
                           "RAG_<GROUP>__<FIELD> in .env or the environment "
                           "(see .env.example)")


def index_error(error: Exception, processed_dir: str, mode: str) -> str:
    """Return what went wrong loading the index and the command to fix it."""
    # semantic and hybrid also need embeddings.npy
    command = "'index'" if mode == "lexical" else "'index --semantic'"
    if isinstance(error, FileNotFoundError):
        if Path(str(error.filename)).name == "embeddings.npy":
            return (f"RAG_RETRIEVAL__MODE={mode} needs embeddings in "
                    f"{processed_dir}. Run {command} first.")
        return (f"No index in {processed_dir} ({error}). "
                f"Run {command} first.")
    return (f"Index in {processed_dir} is unreadable ({error}). "
            f"Run {command} again.")

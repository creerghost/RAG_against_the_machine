from functools import wraps
import json
import pickle
import sys
from typing import Any, Callable


def catch(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Convert expected command failures into concise CLI errors."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Execute a command and report expected failures without tracebacks."""
        try:
            return fn(*args, **kwargs)
        except FileNotFoundError as error:
            _fail(f"File missing: {error}")
        except json.JSONDecodeError as error:
            _fail(f"Invalid JSON dataset format: {error}")
        except pickle.UnpicklingError as error:
            _fail(f"Corrupted index file; re-run index: {error}")
        except KeyError as error:
            _fail(f"Missing expected key in data: {error}")
        except RuntimeError as error:
            if "CUDA out of memory" in str(error) or "OOM" in str(error):
                _fail(
                    f"GPU memory error: {error}. "
                    "Try reducing the model or chunk size."
                )
            _fail(f"Runtime error: {error}")
        except ImportError as error:
            _fail(f"Import error: {error}")
        except (ValueError, TypeError, UnicodeError) as error:
            _fail(str(error))
        except KeyboardInterrupt:
            print("Keyboard interrupt. Bye!", file=sys.stderr)
            raise SystemExit(130)
        except Exception as error:
            _fail(f"Unexpected error: {error}")
        raise AssertionError("unreachable")

    return wrapper


def _fail(message: str) -> None:
    """Print one user-facing error and terminate the current CLI command."""
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)

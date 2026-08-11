from functools import wraps
from typing import Any, Callable
import traceback
import sys
import json
import pickle


def catch(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a callable with universal error handling.

    Args:
        fn: Function or method to wrap.

    Returns:
        Wrapped callable that prints a friendly error message and exits.
    """

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """
        Executes the wrapped loader function and catches generic exceptions.
        """
        try:
            return fn(*args, **kwargs)
        except FileNotFoundError as e:
            print(f"Error: File missing: {e}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON dataset format: {e}")
            sys.exit(1)
        except pickle.UnpicklingError as e:
            print(f"Error: Corrupted index file. Please re-run the 'index' "
                  f"command. ({e})")
            sys.exit(1)
        except KeyError as e:
            print(f"Error: Missing expected key in data: {e}")
            sys.exit(1)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "OOM" in str(e):
                print(f"GPU Memory Error: {e}\nTry limiting max_model_len or "
                      f"chunk size!")
            else:
                print(f"Runtime Error: {e}")
            sys.exit(1)
        except ImportError as e:
            print(f"Import error: {e}")
            sys.exit(1)
        except (ValueError, TypeError) as e:
            print(f"Error: {e}")
            traceback.print_exc()
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt. Bye!")
            sys.exit(0)
        except Exception as e:
            print(f"Unexpected error: {e}")
            traceback.print_exc()
            sys.exit(1)

    return wrapper

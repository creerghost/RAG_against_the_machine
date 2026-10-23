"""Entry point: ``uv run python -m src <command> [options]``."""

import sys

import fire

from .entrypoints import RagCli
from .entrypoints.utils import refuse_global_flags
from .io_utils import CliError


def main() -> None:
    """Run the Fire CLI and print expected errors without a traceback."""
    try:
        refuse_global_flags(sys.argv[1:])
        fire.Fire(RagCli, name="src")
    except CliError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError as e:
        print(f"Error: not implemented yet: {e}", file=sys.stderr)
        sys.exit(1)
    except (MemoryError, RecursionError):  # Fire parsing a huge argument
        print("Error: an argument is too long or too complex to parse",
              file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:  # last resort: the CLI must never show a trace
        print(f"Unexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

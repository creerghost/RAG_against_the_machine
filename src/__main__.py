from .app import Pipeline
import fire


def main() -> None:
    """Start the Python Fire command-line interface."""
    fire.Fire(Pipeline())


if __name__ == "__main__":
    main()

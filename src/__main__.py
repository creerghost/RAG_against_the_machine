from .app import Pipeline
import fire


def main() -> None:
    fire.Fire(Pipeline)


if __name__ == "__main__":
    main()

import subprocess
import sys
from pathlib import Path


def test_cli_help_lists_all_required_commands() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    output = result.stdout + result.stderr
    for command in (
        "index",
        "search",
        "search_dataset",
        "answer",
        "answer_dataset",
        "evaluate",
    ):
        assert command in output


def test_cli_malformed_json_has_no_traceback(tmp_path: Path) -> None:
    dataset_path = tmp_path / "malformed.json"
    dataset_path.write_text("not json")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src",
            "search_dataset",
            "--dataset_path",
            str(dataset_path),
            "--k",
            "1",
            "--save_directory",
            str(tmp_path / "output"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr


def test_cli_index_help_uses_default_corpus() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src", "index", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "__main__.py index <flags>" in output

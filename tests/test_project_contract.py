import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_readme_has_required_first_line_and_sections() -> None:
    readme = ROOT / "README.md"
    content = readme.read_text(encoding="utf-8")
    lines = content.splitlines()
    required_sections = (
        "## Description",
        "## Instructions",
        "## Resources",
        "## System architecture",
        "## Chunking strategy",
        "## Retrieval method",
        "## Performance analysis",
        "## Design decisions",
        "## Challenges faced",
        "## Example usage",
    )

    assert lines[0] == (
        "*This project has been created as part of the 42 curriculum by "
        "vlnikola.*"
    )
    for section in required_sections:
        assert section in content


def test_make_run_executes_the_main_module() -> None:
    result = subprocess.run(
        ["make", "-n", "run"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "python -m src --help" in result.stdout


def test_pytest_is_scoped_to_project_tests() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "testpaths" in pyproject
    assert "tests" in pyproject


def test_project_does_not_reference_a_missing_uv_source() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "llm-sdk" not in pyproject

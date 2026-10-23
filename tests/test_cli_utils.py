"""Tests for the CLI helpers in cli_utils."""

import pytest

from src.entrypoints.cli_utils import refuse_global_flags
from src.io_utils import CliError


@pytest.mark.parametrize("args", [["--mode", "hybrid", "search", "x"],
                                  ["--model_dtype=bfloat16", "answer", "x"]])
def test_flags_before_command_refused(args: list[str]) -> None:
    """Settings given as options before the command are an error."""
    with pytest.raises(CliError, match="before the command"):
        refuse_global_flags(args)


@pytest.mark.parametrize("args", [["search", "x", "--k", "3"], ["--help"],
                                  []])
def test_command_options_allowed(args: list[str]) -> None:
    """Options after the command, help and no arguments pass through."""
    refuse_global_flags(args)

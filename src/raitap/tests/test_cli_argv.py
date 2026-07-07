from __future__ import annotations

from pathlib import Path

from raitap import _cli_argv


def test_find_flag_value_space_form() -> None:
    assert _cli_argv.find_flag_value(["run", "--config-name", "x"], ("--config-name", "-cn")) == "x"


def test_find_flag_value_equals_form() -> None:
    assert _cli_argv.find_flag_value(["run", "-cn=x"], ("--config-name", "-cn")) == "x"


def test_find_flag_value_absent() -> None:
    assert _cli_argv.find_flag_value(["run"], ("--config-name",)) is None


def test_rewrite_demo_injects_config_name() -> None:
    assert _cli_argv.rewrite_demo(["prog", "--demo"]) == ["prog", "--config-name", "demo"]


def test_rewrite_demo_noop_without_flag() -> None:
    assert _cli_argv.rewrite_demo(["prog", "foo=1"]) == ["prog", "foo=1"]


def test_needs_help_frame_true_when_no_selector() -> None:
    assert _cli_argv.needs_help_frame([]) is True
    assert _cli_argv.needs_help_frame(["foo=1"]) is True


def test_needs_help_frame_false_with_demo() -> None:
    assert _cli_argv.needs_help_frame(["--demo"]) is False


def test_needs_help_frame_false_with_config_name() -> None:
    assert _cli_argv.needs_help_frame(["--config-name", "x"]) is False


def test_needs_help_frame_false_with_introspection() -> None:
    assert _cli_argv.needs_help_frame(["--help"]) is False


def test_inject_config_dir_skips_default_name() -> None:
    assert _cli_argv.inject_config_dir(["prog", "--config-name", "demo"]) == [
        "prog",
        "--config-name",
        "demo",
    ]


def test_inject_config_dir_injects_for_custom_name() -> None:
    out = _cli_argv.inject_config_dir(["prog", "--config-name", "mycfg"])
    assert out[:2] == ["prog", "--config-dir"]
    assert out[2] == str(Path.cwd())
    assert out[3:] == ["--config-name", "mycfg"]


def test_inject_config_dir_respects_explicit_config_dir() -> None:
    argv = ["prog", "--config-name", "mycfg", "--config-dir", "/somewhere"]
    assert _cli_argv.inject_config_dir(argv) == argv

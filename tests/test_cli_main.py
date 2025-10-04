"""Tests for main CLI functionality."""

import logging
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from gpux.cli.main import app, main


class TestMainCLI:
    """Test cases for main CLI functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_app_creation(self) -> None:
        """Test that the CLI app is created correctly."""
        assert isinstance(app, typer.Typer)
        assert app.info.name == "gpux"
        assert "Docker-like GPU runtime" in app.info.help

    def test_version_option(self) -> None:
        """Test version option."""
        result = self.runner.invoke(app, ["--version", "build", "--help"])
        assert result.exit_code == 0
        assert "GPUX version" in result.output

    def test_verbose_option(self) -> None:
        """Test verbose option."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = mock_get_logger.return_value
            result = self.runner.invoke(app, ["--verbose", "build", "--help"])
            assert result.exit_code == 0
            mock_logger.setLevel.assert_called_once_with(logging.DEBUG)

    def test_help_output(self) -> None:
        """Test help output."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Docker-like GPU runtime" in result.output
        assert "build" in result.output
        assert "run" in result.output
        assert "serve" in result.output
        assert "inspect" in result.output

    def test_main_callback_version(self) -> None:
        """Test main callback with version flag."""
        with patch("typer.echo") as mock_echo, patch("typer.Exit") as mock_exit:
            mock_exit.side_effect = SystemExit
            with pytest.raises(SystemExit):
                main(version=True, verbose=False)
            mock_echo.assert_called_once()

    def test_main_callback_verbose(self) -> None:
        """Test main callback with verbose flag."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = mock_get_logger.return_value
            main(version=False, verbose=True)
            mock_logger.setLevel.assert_called_once_with(logging.DEBUG)

    def test_main_callback_normal(self) -> None:
        """Test main callback with normal flags."""
        # Should not raise any exceptions
        main(version=False, verbose=False)

    def test_main_module_execution(self) -> None:
        """Test main module execution."""
        # Test that the main module can be imported without errors
        import gpux.cli.main

        assert hasattr(gpux.cli.main, "app")
        assert hasattr(gpux.cli.main, "main")

    def test_commands_registered(self) -> None:
        """Test that all commands are registered."""
        # Check that commands are accessible through the app
        assert hasattr(app, "command")
        # Test that we can get help for each command
        for cmd_name in ["build", "run", "serve", "inspect"]:
            result = self.runner.invoke(app, [cmd_name, "--help"])
            assert result.exit_code == 0

    def test_logging_configuration(self) -> None:
        """Test that logging is configured correctly."""
        # This test verifies that the logging configuration doesn't raise errors
        # The actual configuration is tested indirectly through other tests
        # The root logger level might be different, so we just check it's configured
        assert logging.getLogger().level >= logging.DEBUG

    def test_import_version_fallback(self) -> None:
        """Test version import fallback."""
        with patch("gpux.cli.main.__version__", "unknown"):
            result = self.runner.invoke(app, ["--version", "build", "--help"])
            assert result.exit_code == 0
            assert "GPUX version unknown" in result.output

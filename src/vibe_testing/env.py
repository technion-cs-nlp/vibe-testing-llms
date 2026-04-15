"""
Environment loading utilities.

This module is intentionally lightweight and safe to import from anywhere.
It provides `.env` loading without enforcing other environment requirements
like HM_HOME (those checks belong in `utils.ensure_environment()`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_DOTENV_PATH: Optional[Path] = None


def _find_dotenv_path(start_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Find a `.env` file by walking upward from the start directory.

    Args:
        start_dir (Optional[Path]): Directory to start searching from. Defaults
            to the current working directory.

    Returns:
        Optional[Path]: Path to the first `.env` file found, otherwise None.
    """
    start = (start_dir or Path.cwd()).resolve()
    for directory in (start, *start.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_project_dotenv(override: bool = False, start_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Load environment variables from a project `.env` file if present.

    This function is idempotent within a process once a `.env` has been found
    and loaded. It is safe to call early in
    CLI entrypoints and right before model initialization (API keys commonly
    come from `.env` in local development, but Slurm jobs typically won't source
    it automatically).

    Args:
        override (bool): Whether to override existing environment variables.
        start_dir (Optional[Path]): Optional directory to start searching from.
            If not provided, searches upward from the current working directory.

    Returns:
        Optional[Path]: The path to the loaded `.env` file, or None if none was found.
    """
    global _DOTENV_PATH
    if _DOTENV_PATH is not None:
        return _DOTENV_PATH

    env_path = _find_dotenv_path(start_dir=start_dir)
    if env_path is None:
        logger.debug(
            "No .env file found while searching from start_dir=%s upward.",
            (start_dir or Path.cwd()),
        )
        return None

    load_dotenv(dotenv_path=env_path, override=override)
    _DOTENV_PATH = env_path
    logger.info("Loaded environment variables from %s (override=%s).", env_path, override)
    return env_path


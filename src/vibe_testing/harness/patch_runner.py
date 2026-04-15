"""
Wrapper around external repository-level evaluation harnesses (e.g., SWE-Bench).
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Dict, List, Optional


class PatchHarnessInvoker:
    """
    Prepares predictions and launches an external evaluation harness via subprocess.
    """

    def __init__(self, harness_config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Args:
            harness_config (Dict[str, Any]): Configuration describing how to run
                the harness. Expected keys:
                  - command (List[str]): Base command (with optional placeholders).
                  - working_dir (str, optional): Working directory for the command.
                  - env (Dict[str, str], optional): Extra environment variables.
                  - results_path (str, optional): Location of the harness results JSON.
            logger (Optional[logging.Logger]): Optional logger.
        """
        if "command" not in harness_config:
            raise ValueError("Harness configuration must include a 'command' field.")
        self.command_template: List[str] = harness_config["command"]
        self.working_dir: Optional[str] = harness_config.get("working_dir")
        self.env: Dict[str, str] = harness_config.get("env", {})
        self.results_path: Optional[str] = harness_config.get("results_path")
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def run(self, predictions_path: str) -> Dict[str, Any]:
        """
        Execute the harness with the provided predictions path.
        """
        command = [arg.format(predictions_path=predictions_path, results_path=self.results_path or "") for arg in self.command_template]
        env = os.environ.copy()
        env.update(self.env)
        self.logger.info("Running patch harness: %s", " ".join(command))
        completed = subprocess.run(
            command,
            cwd=self.working_dir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "results_path": self.results_path,
        }



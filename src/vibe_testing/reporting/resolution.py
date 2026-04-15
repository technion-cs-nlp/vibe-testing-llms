"""
Resolution rate reporter for repository-level patch evaluations.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from src.vibe_testing.utils import load_json


class ResolutionReporter:
    """
    Parses harness output artifacts and extracts resolution metrics.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def summarize(self, results_path: str) -> Dict[str, Any]:
        """
        Read a results JSON file and compute aggregate resolution statistics.
        """
        if not os.path.exists(results_path):
            self.logger.warning("Resolution file %s does not exist.", results_path)
            return {"results_path": results_path, "summary": {}}

        payload = load_json(results_path)
        summary: Dict[str, Any] = {}

        if isinstance(payload, dict):
            if "metrics" in payload and isinstance(payload["metrics"], dict):
                summary.update(payload["metrics"])
            if "resolution_rate" in payload:
                summary["resolution_rate"] = payload["resolution_rate"]
            if "instances" in payload and isinstance(payload["instances"], list):
                statuses = [bool(item.get("resolved")) for item in payload["instances"] if isinstance(item, dict)]
                if statuses:
                    summary.setdefault("resolution_rate", sum(statuses) / len(statuses))
        else:
            self.logger.warning("Unexpected results schema in %s. Parsed object type: %s", results_path, type(payload))

        return {"results_path": results_path, "summary": summary}



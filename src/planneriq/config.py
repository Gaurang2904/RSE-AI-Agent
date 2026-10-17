"""Paths and privacy settings.

Real client data never lives inside this repo: the database, logs and generated reports go to
PLANNERIQ_PRIVATE (default C:/Users/<you>/PlannerIQ_private). Only Ollama on localhost may
process it.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
BANDS_YAML = CONFIGS / "bands.yaml"

PRIVATE = Path(os.getenv("PLANNERIQ_PRIVATE", Path.home() / "PlannerIQ_private"))
DB_PATH = Path(os.getenv("PLANNERIQ_DB", PRIVATE / "clients.db"))
OUTPUTS = PRIVATE / "outputs"
LOGS = PRIVATE / "logs"

MODEL = os.getenv("PLANNERIQ_MODEL", "qwen3:8b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Columns that are never loaded into the database (personal contact data).
EXCLUDED_COLUMNS = {"contact no", "contact no.", "contact", "phone", "mobile", "email", "pan",
                    "aadhaar", "address"}

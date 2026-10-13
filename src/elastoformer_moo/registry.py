"""Registry of small language models and whether the elastic adapter supports them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from .arch import SUPPORTED_MODEL_TYPES

DEFAULT_REGISTRY = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_id: str
    model_type: str
    family: str
    gated: bool
    notes: str
    verified: bool
    license: Optional[str] = None

    @property
    def supported(self) -> bool:
        return self.model_type in SUPPORTED_MODEL_TYPES


def load_registry(path: Optional[str | Path] = None) -> dict[str, ModelSpec]:
    path = Path(path) if path else DEFAULT_REGISTRY
    if not path.exists():
        raise FileNotFoundError(f"model registry not found: {path}")
    raw = yaml.safe_load(path.read_text())
    models = raw.get("models") if isinstance(raw, dict) else None
    if not models:
        raise ValueError(f"{path} must contain a non-empty 'models' mapping")
    registry: dict[str, ModelSpec] = {}
    for name, entry in models.items():
        missing = {"hf_id", "model_type", "family"} - set(entry)
        if missing:
            raise ValueError(f"model {name!r} is missing fields: {sorted(missing)}")
        registry[name] = ModelSpec(
            name=name,
            hf_id=entry["hf_id"],
            model_type=entry["model_type"],
            family=entry["family"],
            gated=bool(entry.get("gated", False)),
            notes=entry.get("notes", ""),
            verified=bool(entry.get("verified", False)),
            license=entry.get("license"),
        )
    return registry


def get_spec(name: str, path: Optional[str | Path] = None) -> ModelSpec:
    registry = load_registry(path)
    if name not in registry:
        raise KeyError(f"unknown model {name!r}; known: {', '.join(sorted(registry))}")
    return registry[name]

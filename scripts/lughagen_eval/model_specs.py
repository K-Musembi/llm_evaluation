"""Describing where each model lives (Google Drive) and what it needs to load.

``models.json`` maps a model name to either a plain path string, or an object
when the checkpoint is a LoRA/QLoRA *adapter* that must be applied on top of a
base model::

    {
      "Llama-8B-FFT":    "/content/drive/MyDrive/lughagen/models/llama-8b-fft",
      "Llama-8B-QLoRA":  {"path": "/content/drive/MyDrive/lughagen/models/llama-8b-qlora",
                          "base_model_path": "/content/drive/MyDrive/lughagen/models/llama-8b-base"}
    }

For an adapter directory (it contains ``adapter_config.json``) ``base_model_path``
is optional: it defaults to the ``base_model_name_or_path`` recorded in the
adapter config, which may be a Drive path or a Hugging Face model id.

Everything here is plain filesystem logic so it is unit-testable without a GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WEIGHT_SUFFIXES = (".safetensors", ".bin")


@dataclass(frozen=True)
class ModelSpec:
    path: str
    base_path: str | None = None

    @property
    def is_adapter(self) -> bool:
        return (Path(self.path) / "adapter_config.json").exists()


def parse_model_spec(name: str, value: Any) -> ModelSpec:
    if isinstance(value, str):
        return ModelSpec(path=value)
    if isinstance(value, dict) and isinstance(value.get("path"), str):
        base = value.get("base_model_path")
        if base is not None and not isinstance(base, str):
            raise ValueError(f"{name!r}: base_model_path must be a string")
        return ModelSpec(path=value["path"], base_path=base)
    raise ValueError(f"{name!r}: expected a path string or {{'path': ..., 'base_model_path': ...}}, got {value!r}")


def parse_model_specs(raw: dict[str, Any]) -> dict[str, ModelSpec]:
    return {name: parse_model_spec(name, value) for name, value in raw.items()}


def read_adapter_base(adapter_dir: str) -> str | None:
    config_path = Path(adapter_dir) / "adapter_config.json"
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    base = data.get("base_model_name_or_path")
    return base if isinstance(base, str) and base else None


def resolve_model_spec(name: str, spec: ModelSpec) -> ModelSpec:
    """Fill in an adapter's base model from its adapter_config.json if not given."""
    if not spec.is_adapter or spec.base_path:
        return spec
    base = read_adapter_base(spec.path)
    if base is None:
        raise ValueError(
            f"{name!r} at {spec.path} is a LoRA adapter but its base model is unknown. "
            "Add 'base_model_path' for it in models.json."
        )
    return ModelSpec(path=spec.path, base_path=base)


def weights_size_gb(path: str) -> float:
    """Total size of weight files in a checkpoint directory (GB)."""
    root = Path(path)
    if not root.is_dir():
        return 0.0
    total = sum(f.stat().st_size for f in root.iterdir() if f.is_file() and f.suffix in WEIGHT_SUFFIXES)
    return total / 1e9


def estimate_vram_gb(weights_gb: float) -> dict[str, float]:
    """Rough GPU memory needed per loading mode, from on-disk weight size.

    Assumes the checkpoint is stored in 16-bit (fp16/bf16): 8-bit loading is then
    about half and 4-bit about a quarter, plus ~15% for activations/KV cache.
    If the checkpoint is stored in fp32 the disk size is double the 16-bit size,
    so halve the numbers. These are estimates for planning, not guarantees.
    """
    overhead = 1.15
    return {"none": weights_gb * overhead, "8bit": weights_gb * 0.5 * overhead, "4bit": weights_gb * 0.25 * overhead}


def check_model_specs(specs: dict[str, ModelSpec], model_names: list[str]) -> list[str]:
    """Return human-readable problems (empty list = all good)."""
    problems: list[str] = []
    for name in model_names:
        spec = specs.get(name)
        if spec is None:
            problems.append(f"{name}: not listed in the models file")
            continue
        root = Path(spec.path)
        if not root.is_dir():
            problems.append(f"{name}: folder not found: {spec.path} (is Google Drive mounted?)")
            continue
        has_config = (root / "config.json").exists() or (root / "adapter_config.json").exists()
        if not has_config:
            problems.append(f"{name}: no config.json or adapter_config.json in {spec.path}")
        if spec.is_adapter:
            base = spec.base_path or read_adapter_base(spec.path)
            if base is None:
                problems.append(f"{name}: adapter with unknown base model; add 'base_model_path'")
        elif weights_size_gb(spec.path) == 0.0:
            problems.append(f"{name}: no .safetensors/.bin weight files in {spec.path}")
    return problems

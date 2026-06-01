"""
Model loader for multi-knob .nam files.

Loads a .nam file and reconstructs a PyTorch MultiKnobModel.
Handles both flat .nam format and directory format (config.json + weights.npy).
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_knob_names(config: dict) -> List[str]:
    """Extract knob names from condition_dsp config."""
    cdsp = config.get("condition_dsp", {})
    if cdsp and cdsp.get("architecture") == "KnobConditioning":
        return cdsp.get("config", {}).get("knob_names", [])
    return []


def _get_embedding_dim(config: dict) -> int:
    """Extract embedding_dim from condition_dsp config."""
    cdsp = config.get("condition_dsp", {})
    if cdsp and cdsp.get("architecture") == "KnobConditioning":
        return cdsp.get("config", {}).get("embedding_dim", 8)
    return 8


def _get_total_embedding_dim(config: dict) -> int:
    """Extract sum of all embedding dims from knob_config."""
    kc = config.get("knob_config", {})
    return sum(c.get("embedding_dim", 8) for c in kc.values())


def _condition_dsp_weights_to_state_dict(
    cdsp_weights: List[float],
    knob_names: List[str],
    embedding_dim: int,
) -> Dict[str, torch.Tensor]:
    """
    Map flat condition_dsp weight array to KnobConditioningWaveNet state_dict.

    Layout: per knob [weight(embedding_dim), bias(embedding_dim)]
    PyTorch Linear weight shape: (out_features, in_features) = (embedding_dim, 1)
    """
    state_dict = {}
    expected = len(knob_names) * embedding_dim * 2
    if len(cdsp_weights) != expected:
        raise ValueError(
            f"condition_dsp weights: expected {expected}, got {len(cdsp_weights)}"
        )

    offset = 0
    for name in knob_names:
        # Weight: Linear(1, embedding_dim) -> weight shape (embedding_dim, 1)
        w = cdsp_weights[offset : offset + embedding_dim]
        state_dict[f"knob_embeddings.{name}.weight"] = torch.tensor(
            w, dtype=torch.float32
        ).unsqueeze(1)  # (embedding_dim, 1)

        # Bias: shape (embedding_dim,)
        b = cdsp_weights[offset + embedding_dim : offset + 2 * embedding_dim]
        state_dict[f"knob_embeddings.{name}.bias"] = torch.tensor(
            b, dtype=torch.float32
        )

        offset += 2 * embedding_dim

    return state_dict


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_nam(path: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a .nam file and reconstruct a PyTorch model.

    Returns:
        (model, metadata) where metadata contains:
            knob_names: List[str]
            knob_metadata: dict with name -> {min_value, max_value, default_value}
            sample_rate: float
            architecture: str
    """
    path = Path(path)

    if path.is_dir():
        # Directory format: config.json + weights.npy
        config_path = path / "config.json"
        weights_path = path / "weights.npy"
        with open(config_path) as f:
            nam = json.load(f)
        # If weights.npy exists, use it; otherwise weights are in config
        if weights_path.exists():
            external_weights = np.load(weights_path).tolist()
            nam["weights"] = external_weights
    elif path.suffix == ".nam":
        with open(path) as f:
            nam = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Expected .nam or directory.")

    # Determine model architecture
    architecture = nam.get("architecture", "")
    config = nam.get("config", {})
    main_weights = nam.get("weights", [])

    metadata = {
        "knob_names": [],
        "knob_metadata": {},
        "sample_rate": config.get("sample_rate", 48000),
        "architecture": architecture,
    }

    # Load based on architecture
    if architecture == "WaveNet" and config.get("condition_dsp", {}).get("architecture") == "KnobConditioning":
        return _load_multi_knob_model(nam, main_weights, metadata)
    else:
        # Could add other architecture handlers here
        raise ValueError(f"Unsupported architecture: {architecture} with condition_dsp "
                         f"{config.get('condition_dsp', {}).get('architecture', 'N/A')}")


def _load_multi_knob_model(nam: dict, main_weights: list, metadata: dict) -> Tuple[Any, Dict[str, Any]]:
    """Load a multi-knob model from parsed .nam JSON."""
    config = nam["config"]
    cdsp_config = config["condition_dsp"]
    cdsp_weights = cdsp_config.get("weights", [])

    knob_names = _get_knob_names(config)
    embedding_dim = _get_embedding_dim(config)
    total_embedding_dim = len(knob_names) * embedding_dim

    # Build knob_config for model constructor
    knob_meta = config.get("knob_metadata", {})
    knob_config = {}
    for name in knob_names:
        meta = knob_meta.get(name, {})
        knob_config[name] = {
            "embedding_dim": embedding_dim,
            "min_value": meta.get("min_value", 0.0),
            "max_value": meta.get("max_value", 1.0),
            "default_value": meta.get("default_value", 0.5),
        }

    metadata["knob_names"] = knob_names
    metadata["knob_metadata"] = knob_meta

    # Import the multi_knob extension
    _import_multi_knob_extension()

    from extensions.multi_knob import MultiKnobModel

    # Build knob_config for the model
    model = MultiKnobModel(
        knob_config=knob_config,
        base_model="WaveNet",
        sample_rate=config.get("sample_rate", 48000),
    )

    # Load main WaveNet weights
    model._wavenet.import_weights(torch.tensor(main_weights, dtype=torch.float32))

    # Load condition_dsp (knob embedding) weights
    cdsp_state = _condition_dsp_weights_to_state_dict(
        cdsp_weights, knob_names, embedding_dim
    )
    model._wavenet._condition_dsp.load_state_dict(cdsp_state)

    model.eval()
    return model, metadata


def _import_multi_knob_extension():
    """Ensure the multi_knob extension is importable."""
    # Try different strategies to make the extension available
    try:
        from extensions.multi_knob import MultiKnobModel  # noqa: F401
        return
    except ImportError:
        pass

    # Search for neural-amp-modeler in likely locations
    candidates = [
        # Same repo structure as this project
        Path(__file__).resolve().parent.parent.parent.parent / "neural-amp-modeler" / "extensions",
        # Current working directory
        Path.cwd() / "neural-amp-modeler" / "extensions",
        Path.cwd() / "extensions",
    ]

    # Also check if NAM_TRAINER_DIR or similar env var is set
    for env_var in ("NAM_TRAINER_DIR", "NAM_DIR", "NEURAL_AMP_MODELER"):
        val = os.environ.get(env_var)
        if val:
            candidates.insert(0, Path(val) / "extensions")

    for ext_path in candidates:
        multi_knob_path = ext_path / "multi_knob"
        if multi_knob_path.exists() and str(ext_path) not in sys.path:
            sys.path.insert(0, str(ext_path.parent))
            try:
                from extensions.multi_knob import MultiKnobModel  # noqa: F401
                return
            except ImportError:
                continue

    raise ImportError(
        "Could not import extensions.multi_knob. "
        "Set NAM_TRAINER_DIR to the neural-amp-modeler directory, "
        "or ensure the extension is in the Python path."
    )

"""LLM track: dataset construction, QLoRA fine-tuning, and evaluation.

The whole point of this package is that the LLM is a *second encoding of the
same experiment*, not a separate one. ``dataset.py`` derives its splits from
``src/ml/datasets/splits.py`` — the same function the sklearn path uses — and
``serialization.py`` is the single boundary where a feature row becomes text,
shared by training, evaluation and inference.

Submodules import torch/peft/trl lazily, so importing this package on a laptop
without the GPU extras is safe.
"""

from .serialization import serialize_frame, serialize_row

__all__ = ["serialize_frame", "serialize_row"]

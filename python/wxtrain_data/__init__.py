"""PyTorch dataset loaders for wxtrain-exported training data.

This package reads the NPY/JSON artifacts that wxtrain materializes
and presents them as standard ``torch.utils.data.Dataset`` objects.
It does **not** depend on the wxtrain binary at runtime.
"""

from wxtrain_data.dataset import (
    WxforgeDataset,
    WxforgeMultiSampleDataset,
    load_manifest,
)

__all__ = [
    "WxforgeDataset",
    "WxforgeMultiSampleDataset",
    "load_manifest",
]

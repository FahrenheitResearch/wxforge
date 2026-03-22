"""PyTorch dataset loaders for wxforge-exported training data.

This package reads the NPY/JSON artifacts that wxforge materializes
and presents them as standard ``torch.utils.data.Dataset`` objects.
It does **not** depend on the wxforge binary at runtime.
"""

from wxforge_data.dataset import (
    WxforgeDataset,
    WxforgeMultiSampleDataset,
    load_manifest,
)

__all__ = [
    "WxforgeDataset",
    "WxforgeMultiSampleDataset",
    "load_manifest",
]

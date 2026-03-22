"""Core dataset implementations for wxforge NPY-directory exports.

wxforge materializes GRIB data into per-channel ``.npy`` files plus JSON
manifests (``sample_manifest.json`` and ``dataset_manifest.json``).  The
classes here memory-map those arrays and serve random crops as PyTorch
tensors, ready for DataLoader.

Manifest schema (produced by ``wx-export``):

    sample_manifest.json
    --------------------
    {
      "dataset_name": str,
      "sample_id": str,
      "channels": [
        {
          "name": str,
          "level": str,
          "units": str,
          "width": int,
          "height": int,
          "data_file": str,          # relative .npy filename
          "stats": {"min": f, "mean": f, "max": f} | null
        }, ...
      ]
    }

    dataset_manifest.json
    ---------------------
    {
      "dataset_name": str,
      "channels": [str, ...],
      "labels": [str, ...]
    }
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_manifest(data_dir: Union[str, Path]) -> Dict[str, Any]:
    """Read the first available JSON manifest from *data_dir*.

    Tries ``sample_manifest.json`` first (richer per-channel metadata),
    then falls back to ``dataset_manifest.json``.

    Parameters
    ----------
    data_dir:
        Directory containing wxforge NPY output and manifest JSON.

    Returns
    -------
    dict
        Parsed manifest dictionary.

    Raises
    ------
    FileNotFoundError
        If neither manifest file exists in *data_dir*.
    """
    data_dir = Path(data_dir)
    for name in ("sample_manifest.json", "dataset_manifest.json"):
        path = data_dir / name
        if path.exists():
            with open(path, "r") as fh:
                return json.load(fh)
    raise FileNotFoundError(
        f"No sample_manifest.json or dataset_manifest.json found in {data_dir}"
    )


def _load_channel_stats(data_dir: Path) -> Optional[Dict[str, Dict[str, float]]]:
    """Load ``channel_stats.json`` if present.

    Returns a dict mapping channel name to ``{"min": ..., "mean": ..., "max": ...}``.
    """
    path = data_dir / "channel_stats.json"
    if not path.exists():
        return None
    with open(path, "r") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Single-sample dataset
# ---------------------------------------------------------------------------

class WxforgeDataset(Dataset):
    """PyTorch Dataset over a single wxforge NPY-directory export.

    Each item is a tuple ``(tensor, metadata)`` where *tensor* has shape
    ``(C, crop_size, crop_size)`` and *metadata* is a dict with keys such
    as ``dataset_name``, ``sample_id``, ``channel_names``, etc.

    Parameters
    ----------
    data_dir:
        Path to the directory containing ``.npy`` files and a manifest.
    channels:
        Optional list of channel names to include.  When ``None`` (default)
        all channels in the manifest are loaded.
    crop_size:
        Spatial crop size.  A random crop is taken each time ``__getitem__``
        is called.  Set to ``0`` or ``None`` to skip cropping and return
        the full grid.
    normalize:
        If ``True`` (default), normalize each channel to roughly [0, 1]
        using per-channel statistics.  Stats are read from
        ``channel_stats.json`` first; if that file is absent, the per-channel
        ``stats`` block inside ``sample_manifest.json`` is used instead.
        If no stats are available at all, the data is returned as-is.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        channels: Optional[Sequence[str]] = None,
        crop_size: int = 256,
        normalize: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.crop_size = crop_size if crop_size else 0
        self.normalize = normalize

        self.manifest = load_manifest(self.data_dir)

        # Resolve channel list from manifest
        manifest_channels: List[Dict[str, Any]] = self.manifest.get("channels", [])

        # dataset_manifest.json has channels as a flat list of strings;
        # sample_manifest.json has channels as a list of dicts.
        if manifest_channels and isinstance(manifest_channels[0], str):
            # dataset_manifest — we need to discover NPY files ourselves
            self._channel_entries = self._entries_from_dataset_manifest(
                manifest_channels, channels
            )
        else:
            # sample_manifest — full channel metadata available
            self._channel_entries = self._entries_from_sample_manifest(
                manifest_channels, channels
            )

        if not self._channel_entries:
            raise ValueError(
                f"No matching channels found in {self.data_dir}. "
                f"Requested: {channels}"
            )

        self.channel_names: List[str] = [e["name"] for e in self._channel_entries]

        # Memory-map all NPY arrays
        self._arrays: List[np.memmap] = []
        for entry in self._channel_entries:
            npy_path = self.data_dir / entry["data_file"]
            arr = np.load(str(npy_path), mmap_mode="r")
            self._arrays.append(arr)

        # Determine spatial dimensions (height, width) from the first array
        self._height, self._width = self._arrays[0].shape

        # Build normalization parameters (min, range) per channel
        self._norm_min: Optional[np.ndarray] = None
        self._norm_range: Optional[np.ndarray] = None
        if self.normalize:
            self._build_normalization()

    # -- internal helpers ---------------------------------------------------

    def _entries_from_sample_manifest(
        self,
        manifest_channels: List[Dict[str, Any]],
        requested: Optional[Sequence[str]],
    ) -> List[Dict[str, Any]]:
        """Select channel entries from a sample_manifest channel list."""
        if requested is None:
            return list(manifest_channels)
        name_set = set(requested)
        return [ch for ch in manifest_channels if ch["name"] in name_set]

    def _entries_from_dataset_manifest(
        self,
        channel_names: List[str],
        requested: Optional[Sequence[str]],
    ) -> List[Dict[str, Any]]:
        """Build synthetic channel entries when only dataset_manifest exists.

        Scans the directory for ``.npy`` files and matches them to the
        channel name list.
        """
        npy_files = sorted(self.data_dir.glob("*.npy"))
        entries: List[Dict[str, Any]] = []

        if requested is not None:
            name_set = set(requested)
        else:
            name_set = None

        for npy_path in npy_files:
            # Try to match by channel name appearing in the filename
            matched_name: Optional[str] = None
            for ch_name in channel_names:
                if ch_name in npy_path.stem:
                    matched_name = ch_name
                    break

            if matched_name is None:
                # Fall back: use filename stem as channel name
                matched_name = npy_path.stem

            if name_set is not None and matched_name not in name_set:
                continue

            entries.append({
                "name": matched_name,
                "data_file": npy_path.name,
            })

        return entries

    def _build_normalization(self) -> None:
        """Compute per-channel (min, range) for [0, 1] normalization."""
        external_stats = _load_channel_stats(self.data_dir)

        mins = []
        ranges = []
        for i, entry in enumerate(self._channel_entries):
            ch_name = entry["name"]
            stats = None

            # Priority 1: channel_stats.json
            if external_stats and ch_name in external_stats:
                stats = external_stats[ch_name]
            # Priority 2: inline stats in sample_manifest
            elif "stats" in entry and entry["stats"] is not None:
                stats = entry["stats"]

            if stats is not None:
                ch_min = stats["min"]
                ch_max = stats["max"]
                ch_range = ch_max - ch_min if ch_max != ch_min else 1.0
            else:
                # Fall back to computing from the array (reads it fully once)
                arr = self._arrays[i]
                ch_min = float(np.min(arr))
                ch_max = float(np.max(arr))
                ch_range = ch_max - ch_min if ch_max != ch_min else 1.0

            mins.append(ch_min)
            ranges.append(ch_range)

        self._norm_min = np.array(mins, dtype=np.float32)
        self._norm_range = np.array(ranges, dtype=np.float32)

    # -- Dataset interface --------------------------------------------------

    def __len__(self) -> int:
        """Number of random crops that can be drawn.

        Since crops are random, the 'length' is somewhat arbitrary.
        We define it as the number of non-overlapping crops that tile
        the full grid, giving a sensible epoch size.
        """
        if self.crop_size <= 0:
            return 1
        tiles_y = max(1, self._height // self.crop_size)
        tiles_x = max(1, self._width // self.crop_size)
        return tiles_y * tiles_x

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Return ``(tensor, metadata)`` for a random crop.

        Parameters
        ----------
        index:
            Sample index (used only for DataLoader compatibility; the crop
            position is random regardless of index).

        Returns
        -------
        tensor:
            Float32 tensor of shape ``(C, H, W)``.
        metadata:
            Dict with ``dataset_name``, ``sample_id``, ``channel_names``,
            ``crop_y``, ``crop_x``, ``height``, ``width``.
        """
        if self.crop_size > 0 and (
            self.crop_size <= self._height and self.crop_size <= self._width
        ):
            y = torch.randint(0, self._height - self.crop_size + 1, (1,)).item()
            x = torch.randint(0, self._width - self.crop_size + 1, (1,)).item()
            h, w = self.crop_size, self.crop_size
        else:
            y, x = 0, 0
            h, w = self._height, self._width

        # Stack channels → (C, H, W)
        slices = []
        for arr in self._arrays:
            patch = np.array(arr[y : y + h, x : x + w], dtype=np.float32)
            slices.append(patch)
        data = np.stack(slices, axis=0)  # (C, H, W)

        # Normalize
        if self.normalize and self._norm_min is not None:
            # Broadcast (C,) over (C, H, W)
            norm_min = self._norm_min[:, None, None]
            norm_range = self._norm_range[:, None, None]
            data = (data - norm_min) / norm_range

        tensor = torch.from_numpy(data)

        metadata: Dict[str, Any] = {
            "dataset_name": self.manifest.get("dataset_name", ""),
            "sample_id": self.manifest.get("sample_id", ""),
            "channel_names": self.channel_names,
            "crop_y": y,
            "crop_x": x,
            "height": h,
            "width": w,
        }

        return tensor, metadata

    def __repr__(self) -> str:
        return (
            f"WxforgeDataset(data_dir={str(self.data_dir)!r}, "
            f"channels={self.channel_names}, "
            f"grid={self._height}x{self._width}, "
            f"crop_size={self.crop_size})"
        )


# ---------------------------------------------------------------------------
# Multi-sample dataset
# ---------------------------------------------------------------------------

class WxforgeMultiSampleDataset(Dataset):
    """Concatenation of multiple wxforge sample directories.

    Each directory should contain its own NPY files and manifest.  The
    resulting dataset presents a unified index over all samples with
    consistent channel ordering.

    Parameters
    ----------
    sample_dirs:
        List of paths, each pointing to a wxforge NPY-directory export.
    channels:
        Optional channel filter (applied to every sub-dataset).
    crop_size:
        Spatial crop size passed to each ``WxforgeDataset``.
    normalize:
        Normalization flag passed to each ``WxforgeDataset``.
    """

    def __init__(
        self,
        sample_dirs: Sequence[Union[str, Path]],
        channels: Optional[Sequence[str]] = None,
        crop_size: int = 256,
        normalize: bool = True,
    ) -> None:
        if not sample_dirs:
            raise ValueError("sample_dirs must be a non-empty list of paths")

        self.datasets: List[WxforgeDataset] = []
        self._cumulative_lengths: List[int] = []
        cumulative = 0

        for d in sample_dirs:
            ds = WxforgeDataset(
                data_dir=d,
                channels=channels,
                crop_size=crop_size,
                normalize=normalize,
            )
            self.datasets.append(ds)
            cumulative += len(ds)
            self._cumulative_lengths.append(cumulative)

    def __len__(self) -> int:
        if not self._cumulative_lengths:
            return 0
        return self._cumulative_lengths[-1]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"index {index} out of range for length {len(self)}")

        # Binary search for the right sub-dataset
        for i, cum_len in enumerate(self._cumulative_lengths):
            if index < cum_len:
                prev = self._cumulative_lengths[i - 1] if i > 0 else 0
                local_index = index - prev
                return self.datasets[i][local_index]

        raise IndexError(f"index {index} out of range")

    def __repr__(self) -> str:
        return (
            f"WxforgeMultiSampleDataset(num_samples={len(self.datasets)}, "
            f"total_length={len(self)})"
        )

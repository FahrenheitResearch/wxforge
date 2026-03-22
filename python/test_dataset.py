"""Smoke test for wxtrain_data using an existing wxtrain NPY export."""

import sys
from pathlib import Path

# Ensure the package is importable from this directory
sys.path.insert(0, str(Path(__file__).parent))

from wxtrain_data import WxforgeDataset, WxforgeMultiSampleDataset, load_manifest


DATA_DIR = Path(r"C:\Users\drew\AppData\Local\Temp\wxtrain_dataset")


def main():
    if not DATA_DIR.exists():
        print(f"ERROR: data directory not found: {DATA_DIR}")
        print("Run wxtrain to materialize a dataset first.")
        sys.exit(1)

    # 1. Load manifest
    print("=== Manifest ===")
    manifest = load_manifest(DATA_DIR)
    print(f"  dataset_name : {manifest.get('dataset_name')}")
    print(f"  sample_id    : {manifest.get('sample_id', '(n/a)')}")
    channels = manifest.get("channels", [])
    if channels and isinstance(channels[0], dict):
        names = [ch["name"] for ch in channels]
    else:
        names = channels
    print(f"  channels     : {names}")
    print()

    # 2. Single-sample dataset (full grid, no crop)
    print("=== WxforgeDataset (full grid, no crop) ===")
    ds_full = WxforgeDataset(DATA_DIR, crop_size=0, normalize=False)
    print(f"  {ds_full}")
    tensor, meta = ds_full[0]
    print(f"  tensor shape : {tensor.shape}")
    print(f"  dtype        : {tensor.dtype}")
    print(f"  min/max      : {tensor.min().item():.4f} / {tensor.max().item():.4f}")
    print(f"  metadata     : { {k: v for k, v in meta.items() if k != 'channel_names'} }")
    print()

    # 3. Single-sample dataset with random crop + normalization
    print("=== WxforgeDataset (crop=256, normalized) ===")
    ds = WxforgeDataset(DATA_DIR, crop_size=256, normalize=True)
    print(f"  {ds}")
    print(f"  len(ds)      : {len(ds)}")
    tensor, meta = ds[0]
    print(f"  tensor shape : {tensor.shape}")
    print(f"  min/max      : {tensor.min().item():.4f} / {tensor.max().item():.4f}")
    print()

    # 4. DataLoader iteration
    print("=== DataLoader (batch_size=4, 3 batches) ===")
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    for i, (batch_tensor, batch_meta) in enumerate(loader):
        print(f"  batch {i}: shape={batch_tensor.shape}, "
              f"min={batch_tensor.min().item():.4f}, "
              f"max={batch_tensor.max().item():.4f}")
        if i >= 2:
            break
    print()

    # 5. Multi-sample dataset (using the same dir twice as a demo)
    print("=== WxforgeMultiSampleDataset ===")
    multi_ds = WxforgeMultiSampleDataset(
        [DATA_DIR, DATA_DIR], crop_size=128, normalize=True
    )
    print(f"  {multi_ds}")
    print(f"  len(multi_ds): {len(multi_ds)}")
    tensor, meta = multi_ds[0]
    print(f"  tensor shape : {tensor.shape}")
    print()

    print("All tests passed.")


if __name__ == "__main__":
    main()

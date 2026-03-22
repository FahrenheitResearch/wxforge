"""
Binary Severe Weather Classifier (MLP)
=======================================
Extracts scalar features from decoded HRRR GRIBs and trains a 3-layer MLP
to classify severe convective potential (CAPE > 1000 J/kg).

Features: max CAPE, mean CAPE, max wind speed, mean T, mean Td, T-Td spread.

Uses `wxtrain fetch batch` for reliable full-GRIB downloads, then
`wxtrain train build-grib-sample` to decode individual fields to NPY.

Usage:  python train_classifier.py [--hours 24] [--epochs 10]
"""
import argparse, glob, os, shutil, subprocess, sys
import numpy as np
import torch, torch.nn as nn, torch.optim as optim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HOME = os.path.expanduser("~")
WORK = os.path.join(HOME, "wxtrain_training", "train_classifier")

def find_wxtrain():
    for c in [shutil.which("wxtrain"),
              os.path.join(HOME, "wxtrain", "target", "release", "wxtrain.exe"),
              os.path.join(HOME, "wxtrain", "target", "release", "wxtrain")]:
        if c and os.path.isfile(c):
            return c
    sys.exit("ERROR: wxtrain binary not found.")

WXF = find_wxtrain()

def fetch_data(n_hours):
    grib_dir = os.path.join(WORK, "gribs")
    os.makedirs(grib_dir, exist_ok=True)
    existing = glob.glob(os.path.join(grib_dir, "hrrr_f*.grib2"))
    if len(existing) >= n_hours:
        print(f"  Using {len(existing)} cached GRIBs")
        return grib_dir
    print(f"  Downloading {n_hours} HRRR forecast hours...")
    subprocess.run([WXF, "fetch", "batch",
        "--model", "hrrr", "--product", "surface",
        "--forecast-hours", f"0-{n_hours-1}",
        "--output-dir", grib_dir, "--parallelism", "4"],
        capture_output=True, timeout=3600)
    return grib_dir

def decode_grib(grib_path):
    name = os.path.basename(grib_path).replace(".grib2", "")
    out_dir = os.path.join(WORK, "decoded", name)
    if not os.path.isdir(out_dir):
        subprocess.run([WXF, "train", "build-grib-sample", "--file", grib_path,
                        "--output-dir", out_dir, "--colormap", "heat"],
                       capture_output=True, timeout=60)
    return out_dir

def find_field(decoded_dir, pattern):
    matches = glob.glob(os.path.join(decoded_dir, f"*{pattern}*.npy"))
    if matches:
        return np.load(matches[0]).astype(np.float32)
    return None

# -- MLP Architecture (from train_all_quick.py) --
class MLP(nn.Module):
    def __init__(self, inp=6):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inp,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self, x): return self.net(x)

def main():
    parser = argparse.ArgumentParser(description="MLP severe weather classifier")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    print("=" * 60)
    print("Binary Severe Weather Classifier")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    os.makedirs(os.path.join(WORK, "decoded"), exist_ok=True)

    # Step 1: Fetch GRIBs
    print("\n[1/4] Fetching HRRR data...")
    grib_dir = fetch_data(args.hours)

    # Step 2: Decode and extract scalar features
    print("\n[2/4] Decoding GRIBs and extracting features...")
    gribs = sorted(glob.glob(os.path.join(grib_dir, "*.grib2")))
    print(f"  Found {len(gribs)} GRIB files")

    features, labels = [], []
    for grib in gribs:
        dec = decode_grib(grib)
        cape = find_field(dec, "cape_surface")
        t = find_field(dec, "2t_2_m")
        d = find_field(dec, "2d_2_m")
        u = find_field(dec, "10u_10_m")
        v = find_field(dec, "10v_10_m")
        if all(x is not None for x in [cape, t, d]):
            wspd_max = float(np.nanmax(np.sqrt(u**2 + v**2))) if u is not None and v is not None else 0.0
            feat = [float(np.nanmax(cape)), float(np.nanmean(cape)),
                    wspd_max,
                    float(np.nanmean(t)), float(np.nanmean(d)),
                    float(np.nanmean(t - d))]
            features.append(feat)
            labels.append(1.0 if np.nanmax(cape) > 1000 else 0.0)
            print(f"    {os.path.basename(grib)}: maxCAPE={feat[0]:.0f} -> {'SEVERE' if labels[-1] else 'non-severe'}")

    print(f"  Samples: {len(features)}, Severe: {sum(labels):.0f}, Non-severe: {len(labels)-sum(labels):.0f}")
    if len(features) < 3:
        sys.exit(f"ERROR: Only {len(features)} samples, need >= 3.")

    # Step 3: Train
    print(f"\n[3/4] Training MLP ({args.epochs * 20} iterations)...")
    X = torch.tensor(features, dtype=torch.float32)
    Y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    mu, std = X.mean(0), X.std(0) + 1e-8
    X = (X - mu) / std

    model = MLP(X.shape[1]).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss()
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    total_epochs = args.epochs * 20  # 200 for default
    for epoch in range(total_epochs):
        model.train()
        opt.zero_grad()
        out = model(X.to(DEVICE))
        loss = crit(out, Y.to(DEVICE))
        loss.backward()
        opt.step()
        if (epoch + 1) % 50 == 0:
            acc = ((out > 0).float() == Y.to(DEVICE)).float().mean()
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f} acc={acc.item():.1%}")

    # Step 4: Inference
    print("\n[4/4] Per-sample predictions...")
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(X.to(DEVICE))).cpu()
    for i in range(min(5, len(preds))):
        print(f"  Sample {i}: pred={preds[i,0]:.3f} true={labels[i]:.0f}")

    ckpt = os.path.join(WORK, "classifier.pt")
    torch.save({"model": model.state_dict(), "mean": mu, "std": std}, ckpt)
    print(f"\n  Saved: {ckpt}")
    print("=" * 60 + "\nDone!")

if __name__ == "__main__":
    main()

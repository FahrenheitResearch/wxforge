"""
Binary Severe Weather Classifier (MLP)
=======================================
Trains a 3-hidden-layer MLP to classify whether a forecast hour contains
"severe" convective potential (CAPE > 1000 J/kg anywhere in the domain).

Features extracted from HRRR GRIB2: max/mean CAPE, max/mean SRH,
max wind speed, mean T2m, mean Td2m, mean T-Td spread (8 scalars).

Pipeline: wxforge fetches HRRR subsets, decodes to NPY, Python extracts
scalar features, MLP trains on tabular data.
Usage:  python train_classifier.py
Requires: torch, numpy, wxforge binary
"""
from __future__ import annotations
import json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

WORK_DIR = Path(__file__).resolve().parent / "_classifier_workdir"
FHRS = list(range(0, 19))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BATCH_SIZE, LR = 30, 4, 1e-3
SEVERE_THRESHOLD = 1000.0
FIELDS = ["TMP:2 m above ground", "DPT:2 m above ground", "UGRD:10 m above ground",
          "VGRD:10 m above ground", "CAPE:surface", "HLCY:3000-0 m above ground"]
KEYS = ["2t", "2d", "10u", "10v", "cape", "hlcy"]

def find_wxforge():
    for c in [shutil.which("wxforge"), "/root/wxforge/target/release/wxforge",
              str(Path(__file__).resolve().parents[2] / "target/release/wxforge.exe"),
              str(Path(__file__).resolve().parents[2] / "target/release/wxforge")]:
        if c and Path(c).is_file(): return c
    sys.exit("ERROR: wxforge binary not found.")

def run(cmd, **kw):
    print(f"  > {' '.join(cmd[:6])}{'...' if len(cmd)>6 else ''}")
    return subprocess.run(cmd, capture_output=True, text=True, **kw)

def fetch_and_build(wxf):
    dirs = []
    for fhr in FHRS:
        d = WORK_DIR / f"fhr_{fhr:03d}"
        if d.exists() and (d / "sample_manifest.json").exists():
            dirs.append(d); continue
        parts = []
        for field in FIELDS:
            out = WORK_DIR / f"tmp_{field.split(':')[0]}_{fhr:03d}.grib2"
            if not out.exists():
                r = run([wxf, "fetch", "model-subset", "--model", "hrrr",
                         "--product", "surface", "--forecast-hour", str(fhr),
                         "--search", field, "--output", str(out)], check=False)
                if r.returncode != 0: break
            parts.append(out)
        if len(parts) < len(FIELDS): continue
        merged = WORK_DIR / f"hrrr_f{fhr:03d}.grib2"
        with open(merged, "wb") as f:
            for p in parts: f.write(p.read_bytes())
        d.mkdir(parents=True, exist_ok=True)
        r = run([wxf, "train", "build-grib-sample", "--file", str(merged),
                 "--output-dir", str(d)], check=False)
        if r.returncode == 0: dirs.append(d)
    return dirs

def extract_features(d):
    """Extract 8 scalar features + binary label from a wxforge NPY bundle."""
    m = json.loads((d / "sample_manifest.json").read_text())
    ch = {c["name"]: d / c["data_file"] for c in m["channels"]}
    if not all(k in ch for k in KEYS): return None
    data = {k: np.load(str(ch[k])).astype(np.float32) for k in KEYS}
    wspd = np.sqrt(data["10u"]**2 + data["10v"]**2)
    feats = np.array([data["cape"].max(), data["cape"].mean(), data["hlcy"].max(),
                      data["hlcy"].mean(), wspd.max(), data["2t"].mean(),
                      data["2d"].mean(), (data["2t"]-data["2d"]).mean()], dtype=np.float32)
    label = int(data["cape"].max() > SEVERE_THRESHOLD)
    return feats, label

class SevereClassifier(nn.Module):
    def __init__(self, n=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n,64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32,16), nn.ReLU(), nn.Linear(16,1))
    def forward(self, x): return self.net(x).squeeze(-1)

def main():
    print("="*60+"\nBinary Severe Weather Classifier\n"+"="*60)
    wxf = find_wxforge()
    print(f"wxforge: {wxf}\nDevice:  {DEVICE}\n")

    print("[1/4] Fetching HRRR surface data...")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    dirs = fetch_and_build(wxf)
    print(f"  Got {len(dirs)} samples.\n")

    print("[2/4] Extracting tabular features...")
    X_list, y_list = [], []
    for d in dirs:
        r = extract_features(d)
        if r: X_list.append(r[0]); y_list.append(r[1])
    if len(X_list) < 4: sys.exit(f"ERROR: Only {len(X_list)} valid samples, need >=4.")
    X = np.stack(X_list); y = np.array(y_list, dtype=np.float32)
    mu, sig = X.mean(0), X.std(0)+1e-8; X = (X-mu)/sig
    ns = int(y.sum())
    print(f"  {len(X)} samples ({ns} severe, {len(X)-ns} non-severe), 8 features\n")

    split = max(len(X)-3, 1)
    X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]

    print(f"[3/4] Training MLP (train={len(X_tr)}, test={len(X_te)})...")
    ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model = SevereClassifier().to(DEVICE)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss()
    for ep in range(1, EPOCHS+1):
        model.train(); tl=0; cor=0; tot=0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = crit(model(xb), yb); opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item()*len(xb); cor += ((model(xb)>0).long()==yb.long()).sum().item()
            tot += len(xb)
        if ep % 5 == 0 or ep == 1:
            print(f"  Epoch {ep:>3}/{EPOCHS}  loss={tl/max(tot,1):.4f}  acc={cor/max(tot,1)*100:.1f}%")

    print("\n[4/4] Inference on test samples...")
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_te).float().to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (logits > 0).long().cpu().numpy()
    for i in range(len(X_te)):
        tl = "SEVERE" if y_te[i]>0.5 else "non-severe"
        pl = "SEVERE" if preds[i]==1 else "non-severe"
        print(f"  Sample {i}: prob={probs[i]:.3f}  pred={pl:<12s}  truth={tl:<12s}  "
              f"[{'OK' if tl==pl else 'MISS'}]")
    acc = (preds == y_te.astype(int)).mean() * 100
    print(f"\n  Test accuracy: {acc:.1f}%")
    ckpt = WORK_DIR / "classifier.pt"
    torch.save({"model": model.state_dict(), "mu": mu, "sigma": sig}, ckpt)
    print(f"  Model saved to {ckpt}\n" + "="*60 + "\nDone!")

if __name__ == "__main__":
    main()

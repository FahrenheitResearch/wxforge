"""
UNet for CAPE Prediction from Surface Observations
====================================================
Trains a UNet (encoder-decoder with skip connections) to predict CAPE fields
from surface obs: TMP:2m, DPT:2m, UGRD:10m, VGRD:10m -> CAPE:surface.

Pipeline: wxforge fetches HRRR GRIB2 subsets, decodes to NPY, PyTorch trains.
Usage:  python train_unet.py
Requires: torch, numpy, wxforge binary (in PATH or target/release/)
"""
from __future__ import annotations
import json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

WORK_DIR = Path(__file__).resolve().parent / "_unet_workdir"
FHRS = list(range(0, 13))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BATCH_SIZE, LR, CROP = 5, 2, 1e-3, 256
INPUT_FIELDS = ["TMP:2 m above ground", "DPT:2 m above ground",
                "UGRD:10 m above ground", "VGRD:10 m above ground"]
TARGET_FIELD = "CAPE:surface"
INPUT_KEYS, TARGET_KEY = ["2t", "2d", "10u", "10v"], "cape"

def find_wxforge():
    for c in [shutil.which("wxforge"), "/root/wxforge/target/release/wxforge",
              str(Path(__file__).resolve().parents[2] / "target/release/wxforge.exe"),
              str(Path(__file__).resolve().parents[2] / "target/release/wxforge")]:
        if c and Path(c).is_file(): return c
    sys.exit("ERROR: wxforge binary not found. Build it or add to PATH.")

def run(cmd, **kw):
    print(f"  > {' '.join(cmd[:6])}{'...' if len(cmd)>6 else ''}")
    return subprocess.run(cmd, capture_output=True, text=True, **kw)

def fetch_and_build(wxf):
    """Download HRRR subsets and convert to NPY for each forecast hour."""
    dirs = []
    for fhr in FHRS:
        d = WORK_DIR / f"fhr_{fhr:03d}"
        if d.exists() and (d / "sample_manifest.json").exists():
            dirs.append(d); continue
        parts = []
        for field in INPUT_FIELDS + [TARGET_FIELD]:
            out = WORK_DIR / f"tmp_{field.split(':')[0]}_{fhr:03d}.grib2"
            if not out.exists():
                r = run([wxf, "fetch", "model-subset", "--model", "hrrr",
                         "--product", "surface", "--forecast-hour", str(fhr),
                         "--search", field, "--output", str(out)], check=False)
                if r.returncode != 0: break
            parts.append(out)
        if len(parts) < 5: continue
        merged = WORK_DIR / f"hrrr_f{fhr:03d}.grib2"
        with open(merged, "wb") as f:
            for p in parts: f.write(p.read_bytes())
        d.mkdir(parents=True, exist_ok=True)
        r = run([wxf, "train", "build-grib-sample", "--file", str(merged),
                 "--output-dir", str(d)], check=False)
        if r.returncode == 0: dirs.append(d)
    return dirs

# -- Dataset ----------------------------------------------------------------
class CAPEDataset(Dataset):
    def __init__(self, sample_dirs, crop=CROP):
        self.samples = []
        for d in sample_dirs:
            m = json.loads((d / "sample_manifest.json").read_text())
            ch = {c["name"]: d / c["data_file"] for c in m["channels"]}
            if all(k in ch for k in INPUT_KEYS + [TARGET_KEY]):
                self.samples.append(([ch[k] for k in INPUT_KEYS], ch[TARGET_KEY]))
        self.crop = crop
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        inps, tgt_f = self.samples[idx]
        arrs = [np.load(str(f)).astype(np.float32) for f in inps]
        tgt = np.load(str(tgt_f)).astype(np.float32)
        h, w = arrs[0].shape
        ch, cw = min(h, self.crop), min(w, self.crop)
        y0, x0 = (h-ch)//2, (w-cw)//2
        arrs = [a[y0:y0+ch, x0:x0+cw] for a in arrs]
        tgt = tgt[y0:y0+ch, x0:x0+cw]
        inp = np.stack(arrs)
        for c in range(4):
            mu, s = inp[c].mean(), inp[c].std()+1e-8; inp[c] = (inp[c]-mu)/s
        tgt = np.log1p(np.clip(tgt, 0, None))  # log-transform CAPE
        return torch.from_numpy(inp), torch.from_numpy(tgt[None])

# -- UNet Architecture ------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ic,oc,3,padding=1), nn.BatchNorm2d(oc), nn.ReLU(True),
                                 nn.Conv2d(oc,oc,3,padding=1), nn.BatchNorm2d(oc), nn.ReLU(True))
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, ic=4, oc=1, b=32):
        super().__init__()
        self.enc1, self.enc2, self.enc3 = DoubleConv(ic,b), DoubleConv(b,b*2), DoubleConv(b*2,b*4)
        self.pool = nn.MaxPool2d(2)
        self.bot = DoubleConv(b*4, b*8)
        self.up3 = nn.ConvTranspose2d(b*8,b*4,2,stride=2)
        self.dec3 = DoubleConv(b*8, b*4)
        self.up2 = nn.ConvTranspose2d(b*4,b*2,2,stride=2)
        self.dec2 = DoubleConv(b*4, b*2)
        self.up1 = nn.ConvTranspose2d(b*2,b,2,stride=2)
        self.dec1 = DoubleConv(b*2, b)
        self.head = nn.Conv2d(b, oc, 1)
    def forward(self, x):
        _,_,h,w = x.shape
        x = F.pad(x, (0,(8-w%8)%8,0,(8-h%8)%8))
        e1=self.enc1(x); e2=self.enc2(self.pool(e1)); e3=self.enc3(self.pool(e2))
        b=self.bot(self.pool(e3))
        d3=self.dec3(torch.cat([self.up3(b),e3],1))
        d2=self.dec2(torch.cat([self.up2(d3),e2],1))
        d1=self.dec1(torch.cat([self.up1(d2),e1],1))
        return self.head(d1)[:,:,:h,:w]

# -- Training & Inference ---------------------------------------------------
def main():
    print("="*60+"\nUNet CAPE Prediction\n"+"="*60)
    wxf = find_wxforge()
    print(f"wxforge: {wxf}\nDevice:  {DEVICE}\n")

    print("[1/4] Fetching HRRR surface data...")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    dirs = fetch_and_build(wxf)
    print(f"  Got {len(dirs)} samples.\n")
    if len(dirs) < 3: sys.exit("ERROR: Need >=3 samples. Check network.")

    print("[2/4] Building PyTorch dataset...")
    train_ds, test_ds = CAPEDataset(dirs[:-1]), CAPEDataset(dirs[-1:])
    print(f"  Train: {len(train_ds)}, Test: {len(test_ds)}\n")
    if len(train_ds) < 2: sys.exit("ERROR: Not enough valid samples.")
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print("[3/4] Training UNet...")
    model = UNet().to(DEVICE)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.SmoothL1Loss()
    for ep in range(1, EPOCHS+1):
        model.train(); total=0.0
        for inp, tgt in loader:
            inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
            loss = crit(model(inp), tgt); opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"  Epoch {ep}/{EPOCHS}  loss={total/max(len(loader),1):.4f}")

    print("\n[4/4] Inference on test sample...")
    model.eval()
    if len(test_ds) > 0:
        inp, tgt = test_ds[0]
        with torch.no_grad(): pred = model(inp.unsqueeze(0).to(DEVICE)).cpu().squeeze()
        pred_cape, tgt_cape = np.expm1(pred.numpy()), np.expm1(tgt.squeeze().numpy())
        rmse = np.sqrt(np.mean((pred_cape - tgt_cape)**2))
        print(f"  Pred CAPE: [{pred_cape.min():.0f}, {pred_cape.max():.0f}] J/kg")
        print(f"  True CAPE: [{tgt_cape.min():.0f}, {tgt_cape.max():.0f}] J/kg")
        print(f"  RMSE: {rmse:.1f} J/kg")

    ckpt = WORK_DIR / "unet_cape.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"\n  Model saved to {ckpt}\n" + "="*60 + "\nDone!")

if __name__ == "__main__":
    main()

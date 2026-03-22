"""
UNet for CAPE Prediction from Surface Fields
=============================================
Trains a UNet (3-level encoder/decoder + skip connections) to predict CAPE
from TMP:2m, DPT:2m, UGRD:10m, VGRD:10m (4ch -> 1ch).

Uses `wxforge fetch batch` for reliable full-GRIB downloads, then
`wxforge train build-grib-sample` to decode individual fields to NPY.

Usage:  python train_unet.py [--hours 24] [--epochs 10] [--crop 256]
"""
import argparse, glob, os, shutil, subprocess, sys, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HOME = os.path.expanduser("~")
WORK = os.path.join(HOME, "wxforge_training", "train_unet")

def find_wxforge():
    for c in [shutil.which("wxforge"),
              os.path.join(HOME, "wxforge", "target", "release", "wxforge.exe"),
              os.path.join(HOME, "wxforge", "target", "release", "wxforge")]:
        if c and os.path.isfile(c):
            return c
    sys.exit("ERROR: wxforge binary not found.")

WXF = find_wxforge()

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

# -- UNet Architecture (from train_all_quick.py) --
class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(i,o,3,1,1),nn.BatchNorm2d(o),nn.ReLU(True),
                                 nn.Conv2d(o,o,3,1,1),nn.BatchNorm2d(o),nn.ReLU(True))
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, inc=4):
        super().__init__()
        self.e1,self.e2,self.e3,self.b = DoubleConv(inc,64),DoubleConv(64,128),DoubleConv(128,256),DoubleConv(256,512)
        self.u3,self.d3 = nn.ConvTranspose2d(512,256,2,2),DoubleConv(512,256)
        self.u2,self.d2 = nn.ConvTranspose2d(256,128,2,2),DoubleConv(256,128)
        self.u1,self.d1 = nn.ConvTranspose2d(128,64,2,2),DoubleConv(128,64)
        self.out,self.pool = nn.Conv2d(64,1,1),nn.MaxPool2d(2)
    def forward(self,x):
        e1=self.e1(x);e2=self.e2(self.pool(e1));e3=self.e3(self.pool(e2));b=self.b(self.pool(e3))
        return self.out(self.d1(torch.cat([self.u1(self.d2(torch.cat([self.u2(self.d3(torch.cat([self.u3(b),e3],1))),e2],1))),e1],1)))

def main():
    parser = argparse.ArgumentParser(description="UNet CAPE prediction")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--crop", type=int, default=256)
    args = parser.parse_args()
    CS = args.crop

    print("=" * 60)
    print("UNet CAPE Prediction")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    os.makedirs(os.path.join(WORK, "decoded"), exist_ok=True)

    # Step 1: Fetch GRIBs
    print("\n[1/4] Fetching HRRR data...")
    grib_dir = fetch_data(args.hours)

    # Step 2: Decode and find fields
    print("\n[2/4] Decoding GRIBs and building samples...")
    gribs = sorted(glob.glob(os.path.join(grib_dir, "*.grib2")))
    print(f"  Found {len(gribs)} GRIB files")

    samples = []
    for grib in gribs:
        dec = decode_grib(grib)
        t = find_field(dec, "2t_2_m")
        d = find_field(dec, "2d_2_m")
        u = find_field(dec, "10u_10_m")
        v = find_field(dec, "10v_10_m")
        cape = find_field(dec, "cape_surface")
        if all(x is not None for x in [t, d, u, v, cape]):
            samples.append((np.stack([t, d, u, v], 0), cape))
            print(f"    {os.path.basename(grib)}: OK ({t.shape})")

    print(f"  Samples: {len(samples)}")
    if len(samples) < 1:
        sys.exit("ERROR: No valid samples found.")

    # Step 3: Train
    print(f"\n[3/4] Training UNet ({args.epochs} epochs)...")
    model = UNet().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.SmoothL1Loss()
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for inp, tgt in samples:
            h, w = inp.shape[1], inp.shape[2]
            if h < CS or w < CS:
                continue
            for _ in range(8):
                y, x = np.random.randint(0, h - CS), np.random.randint(0, w - CS)
                crop_in = inp[:, y:y+CS, x:x+CS].copy()
                crop_tgt = tgt[y:y+CS, x:x+CS].copy()
                for c in range(4):
                    mu, std = crop_in[c].mean(), crop_in[c].std() + 1e-8
                    crop_in[c] = (crop_in[c] - mu) / std
                crop_tgt = np.log1p(crop_tgt)
                ti = torch.from_numpy(crop_in[None]).to(DEVICE)
                tt = torch.from_numpy(crop_tgt[None, None]).to(DEVICE)
                opt.zero_grad()
                out = model(ti)
                loss = crit(out, tt)
                loss.backward()
                opt.step()
                losses.append(loss.item())
        print(f"  Epoch {epoch+1}/{args.epochs}: loss={np.mean(losses):.6f}")

    # Step 4: Inference
    print("\n[4/4] Inference on first sample...")
    model.eval()
    inp, tgt = samples[0]
    h, w = inp.shape[1], inp.shape[2]
    y0, x0 = min(100, h - CS), min(100, w - CS)
    crop_in = inp[:, y0:y0+CS, x0:x0+CS].copy()
    for c in range(4):
        mu, std = crop_in[c].mean(), crop_in[c].std() + 1e-8
        crop_in[c] = (crop_in[c] - mu) / std
    with torch.no_grad():
        pred = model(torch.from_numpy(crop_in[None]).to(DEVICE))
    pred_cape = np.expm1(pred.cpu().numpy()[0, 0])
    true_cape = tgt[y0:y0+CS, x0:x0+CS]
    rmse = np.sqrt(np.mean((pred_cape - true_cape) ** 2))
    print(f"  RMSE: {rmse:.1f} J/kg")
    print(f"  Pred range: [{pred_cape.min():.0f}, {pred_cape.max():.0f}]")
    print(f"  True range: [{true_cape.min():.0f}, {true_cape.max():.0f}]")

    ckpt = os.path.join(WORK, "unet_cape.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"\n  Saved: {ckpt}")
    print("=" * 60 + "\nDone!")

if __name__ == "__main__":
    main()

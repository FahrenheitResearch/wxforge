"""
Diffusion Model for CAPE Super-Resolution (4x)
================================================
Trains a conditional DDPM to upscale 4x-downsampled CAPE fields.
DiffUNet backbone with learned time embedding, 100 linear-beta timesteps,
noise-prediction objective.

Uses `wxforge fetch batch` for reliable full-GRIB downloads, then
`wxforge train build-grib-sample` to decode individual fields to NPY.

Usage:  python train_diffusion.py [--hours 24] [--epochs 10] [--crop 128]
Requires: torch, numpy, scipy (for zoom)
"""
import argparse, glob, os, shutil, subprocess, sys
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from scipy.ndimage import zoom

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HOME = os.path.expanduser("~")
WORK = os.path.join(HOME, "wxforge_training", "train_diffusion")

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

# -- Diffusion UNet Architecture (from train_all_quick.py) --
class TimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
    def forward(self, t):
        return self.net(t.view(-1, 1).float())

class DiffUNet(nn.Module):
    def __init__(self, inc=2, dim=64):
        super().__init__()
        self.te = TimeEmbed(dim)
        self.enc = nn.Sequential(nn.Conv2d(inc,dim,3,1,1),nn.SiLU(),nn.Conv2d(dim,dim,3,1,1),nn.SiLU())
        self.mid = nn.Sequential(nn.Conv2d(dim,dim*2,3,2,1),nn.SiLU(),nn.Conv2d(dim*2,dim*2,3,1,1),nn.SiLU())
        self.dec = nn.Sequential(nn.ConvTranspose2d(dim*2,dim,2,2),nn.SiLU(),nn.Conv2d(dim,dim,3,1,1),nn.SiLU())
        self.out = nn.Conv2d(dim*2, 1, 1)
        self.te_proj = nn.Linear(dim, dim)
    def forward(self, x, t):
        te = self.te_proj(self.te(t))  # (B, dim)
        e = self.enc(x)
        e = e + te[:, :, None, None].expand_as(e)
        m = self.mid(e)
        d = self.dec(m)
        return self.out(torch.cat([e, d], 1))

def main():
    parser = argparse.ArgumentParser(description="Diffusion CAPE super-resolution")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--crop", type=int, default=128)
    args = parser.parse_args()
    CS = args.crop
    T_STEPS = 100

    print("=" * 60)
    print("Diffusion Model — 4x CAPE Super-Resolution")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    os.makedirs(os.path.join(WORK, "decoded"), exist_ok=True)

    # Step 1: Fetch GRIBs
    print("\n[1/4] Fetching HRRR data...")
    grib_dir = fetch_data(args.hours)

    # Step 2: Decode and collect CAPE fields
    print("\n[2/4] Decoding GRIBs and collecting CAPE fields...")
    gribs = sorted(glob.glob(os.path.join(grib_dir, "*.grib2")))
    print(f"  Found {len(gribs)} GRIB files")

    samples = []
    for grib in gribs:
        dec = decode_grib(grib)
        cape = find_field(dec, "cape_surface")
        if cape is not None:
            samples.append(cape)
            print(f"    {os.path.basename(grib)}: OK ({cape.shape})")

    print(f"  CAPE samples: {len(samples)}")
    if len(samples) < 2:
        sys.exit("ERROR: Need at least 2 CAPE samples.")

    # Diffusion schedule
    betas = np.linspace(1e-4, 0.02, T_STEPS).astype(np.float32)
    alphas = 1 - betas
    alpha_bar = np.cumprod(alphas)

    # Step 3: Train
    print(f"\n[3/4] Training DiffUNet ({args.epochs} epochs)...")
    model = DiffUNet(inc=2, dim=64).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Diffusion timesteps: {T_STEPS}")

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for cape in samples:
            h, w = cape.shape
            if h < CS or w < CS:
                continue
            for _ in range(4):
                y, x = np.random.randint(0, h - CS), np.random.randint(0, w - CS)
                hr = cape[y:y+CS, x:x+CS].copy()
                hr = np.log1p(hr)
                mu, std = hr.mean(), hr.std() + 1e-8
                hr = (hr - mu) / std
                # Low-res condition (4x downsample then upsample)
                lr = zoom(zoom(hr, 0.25, order=1), 4.0, order=1)
                # Add noise
                t = np.random.randint(0, T_STEPS)
                ab = alpha_bar[t]
                noise = np.random.randn(*hr.shape).astype(np.float32)
                noisy = np.sqrt(ab) * hr + np.sqrt(1 - ab) * noise
                # Stack: noisy + condition
                inp = np.stack([noisy, lr], 0)
                ti = torch.from_numpy(inp[None]).to(DEVICE)
                tt = torch.tensor([t / T_STEPS]).float().to(DEVICE)
                tn = torch.from_numpy(noise[None, None]).to(DEVICE)
                opt.zero_grad()
                pred = model(ti, tt)
                loss = nn.MSELoss()(pred, tn)
                loss.backward()
                opt.step()
                losses.append(loss.item())
        print(f"  Epoch {epoch+1}/{args.epochs}: loss={np.mean(losses):.6f}")

    # Step 4: Inference — reverse diffusion
    print("\n[4/4] Reverse diffusion sampling...")
    model.eval()
    cape = samples[0]
    h, w = cape.shape
    y0, x0 = min(100, h - CS), min(100, w - CS)
    hr = cape[y0:y0+CS, x0:x0+CS].copy()
    hr_norm = (np.log1p(hr) - np.log1p(hr).mean()) / (np.log1p(hr).std() + 1e-8)
    lr = zoom(zoom(hr_norm, 0.25, order=1), 4.0, order=1)

    x_t = torch.randn(1, 1, CS, CS).to(DEVICE)
    lr_t = torch.from_numpy(lr[None, None]).to(DEVICE)
    with torch.no_grad():
        for t in reversed(range(T_STEPS)):
            tt = torch.tensor([t / T_STEPS]).float().to(DEVICE)
            inp = torch.cat([x_t, lr_t], 1)
            pred_noise = model(inp, tt)
            ab = alpha_bar[t]
            x_t = (x_t - (1 - alphas[t]) / np.sqrt(1 - ab) * pred_noise) / np.sqrt(alphas[t])
            if t > 0:
                x_t += np.sqrt(betas[t]) * torch.randn_like(x_t)

    sr = x_t.cpu().numpy()[0, 0]
    baseline = lr
    sr_rmse = np.sqrt(np.mean((sr - hr_norm) ** 2))
    bl_rmse = np.sqrt(np.mean((baseline - hr_norm) ** 2))
    print(f"  Diffusion RMSE: {sr_rmse:.4f}")
    print(f"  Bilinear RMSE:  {bl_rmse:.4f}")
    print(f"  Improvement: {(1 - sr_rmse / bl_rmse) * 100:.1f}%")

    ckpt = os.path.join(WORK, "diffusion_sr.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"\n  Saved: {ckpt}")
    print("=" * 60 + "\nDone!")

if __name__ == "__main__":
    main()

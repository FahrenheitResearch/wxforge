"""
Swin Transformer for 3-Hour Weather Forecasting
=================================================
Trains a simplified Swin-like transformer to predict TMP + CAPE 3 hours ahead.
Input: TMP:2m + CAPE:surface at time T (2 channels).
Target: same fields at T+3h.

Architecture: PatchEmbed (8x8) + 2 SwinBlocks with MultiheadAttention + linear head.

Uses `wxtrain fetch batch` for reliable full-GRIB downloads, then
`wxtrain train build-grib-sample` to decode individual fields to NPY.

Usage:  python train_swin.py [--hours 24] [--epochs 10] [--crop 256]
"""
import argparse, glob, os, shutil, subprocess, sys
import numpy as np
import torch, torch.nn as nn, torch.optim as optim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HOME = os.path.expanduser("~")
WORK = os.path.join(HOME, "wxtrain_training", "train_swin")

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

# -- Swin Architecture (from train_all_quick.py) --
class PatchEmbed(nn.Module):
    def __init__(self, ps=8, inc=2, dim=128):
        super().__init__()
        self.proj = nn.Conv2d(inc, dim, ps, ps)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class SwinBlock(nn.Module):
    def __init__(self, dim=128, heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        return x + self.ffn(self.norm2(x))

class SwinForecast(nn.Module):
    def __init__(self, inc=2, outc=2, ps=8, dim=128):
        super().__init__()
        self.ps = ps
        self.embed = PatchEmbed(ps, inc, dim)
        self.blocks = nn.Sequential(SwinBlock(dim), SwinBlock(dim))
        self.head = nn.Linear(dim, outc * ps * ps)
        self.outc = outc
    def forward(self, x):
        B, _, H, W = x.shape
        tokens = self.blocks(self.embed(x))
        out = self.head(tokens)
        pH, pW = H // self.ps, W // self.ps
        return out.transpose(1, 2).reshape(B, self.outc, pH, self.ps, pW, self.ps).permute(0,1,2,4,3,5).reshape(B, self.outc, H, W)

def main():
    parser = argparse.ArgumentParser(description="Swin transformer 3h forecast")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--crop", type=int, default=256)
    args = parser.parse_args()
    CS = args.crop
    LEAD = 3
    NC = 2  # channels: TMP + CAPE

    print("=" * 60)
    print("Swin Transformer 3h Forecasting")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    os.makedirs(os.path.join(WORK, "decoded"), exist_ok=True)

    # Step 1: Fetch GRIBs
    print("\n[1/4] Fetching HRRR data...")
    grib_dir = fetch_data(args.hours)

    # Step 2: Decode all GRIBs and index fields by forecast hour
    print("\n[2/4] Decoding GRIBs and building forecast pairs...")
    gribs = sorted(glob.glob(os.path.join(grib_dir, "*.grib2")))
    print(f"  Found {len(gribs)} GRIB files")

    # Index decoded fields by forecast hour
    fhr_fields = {}  # fhr -> {pattern: array}
    for grib in gribs:
        dec = decode_grib(grib)
        basename = os.path.basename(grib)
        # Extract forecast hour from filename (e.g., hrrr_f03.grib2)
        t = find_field(dec, "2t_2_m")
        cape = find_field(dec, "cape_surface")
        if t is not None and cape is not None:
            fhr_fields[basename] = {"2t": t, "cape": cape}
            print(f"    {basename}: OK ({t.shape})")

    # Build T -> T+LEAD pairs using sorted keys
    sorted_keys = sorted(fhr_fields.keys())
    pairs = []
    for i in range(len(sorted_keys) - LEAD):
        k_now = sorted_keys[i]
        k_fut = sorted_keys[i + LEAD]
        now_data = fhr_fields[k_now]
        fut_data = fhr_fields[k_fut]
        inp = np.stack([now_data["2t"], now_data["cape"]], 0)
        tgt = np.stack([fut_data["2t"], fut_data["cape"]], 0)
        pairs.append((inp, tgt))

    print(f"  Pairs: {len(pairs)} (t -> t+{LEAD}h)")
    if len(pairs) < 1:
        sys.exit("ERROR: Need at least 1 forecast pair.")

    # Step 3: Train
    print(f"\n[3/4] Training Swin ({args.epochs} epochs)...")
    model = SwinForecast(inc=NC, outc=NC, ps=8, dim=128).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=5e-4)
    crit = nn.SmoothL1Loss()
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for inp, tgt in pairs:
            h, w = inp.shape[1], inp.shape[2]
            if h < CS or w < CS:
                continue
            for _ in range(4):
                y, x = np.random.randint(0, h - CS), np.random.randint(0, w - CS)
                ci = inp[:, y:y+CS, x:x+CS].copy()
                ct = tgt[:, y:y+CS, x:x+CS].copy()
                for c in range(NC):
                    mu = (ci[c].mean() + ct[c].mean()) / 2
                    std = max(ci[c].std(), ct[c].std(), 1e-8)
                    ci[c] = (ci[c] - mu) / std
                    ct[c] = (ct[c] - mu) / std
                ti = torch.from_numpy(ci[None]).to(DEVICE)
                tt = torch.from_numpy(ct[None]).to(DEVICE)
                opt.zero_grad()
                out = model(ti)
                loss = crit(out, tt)
                loss.backward()
                opt.step()
                losses.append(loss.item())
        print(f"  Epoch {epoch+1}/{args.epochs}: loss={np.mean(losses):.6f}")

    # Step 4: Inference
    print("\n[4/4] Inference on first pair...")
    model.eval()
    inp, tgt = pairs[0]
    h, w = inp.shape[1], inp.shape[2]
    y0, x0 = min(100, h - CS), min(100, w - CS)
    ci = inp[:, y0:y0+CS, x0:x0+CS].copy()
    ct = tgt[:, y0:y0+CS, x0:x0+CS].copy()
    mus, stds = [], []
    for c in range(NC):
        mu = (ci[c].mean() + ct[c].mean()) / 2
        std = max(ci[c].std(), ct[c].std(), 1e-8)
        ci[c] = (ci[c] - mu) / std
        mus.append(mu)
        stds.append(std)
    with torch.no_grad():
        pred = model(torch.from_numpy(ci[None]).to(DEVICE)).cpu().numpy()[0]
    names = ["TMP:2m", "cape"]
    for c in range(NC):
        pred[c] = pred[c] * stds[c] + mus[c]
        ct_un = tgt[c, y0:y0+CS, x0:x0+CS]
        rmse = np.sqrt(np.mean((pred[c] - ct_un) ** 2))
        print(f"  {names[c]} RMSE: {rmse:.2f}")

    ckpt = os.path.join(WORK, "swin_forecast.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"\n  Saved: {ckpt}")
    print("=" * 60 + "\nDone!")

if __name__ == "__main__":
    main()

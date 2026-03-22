"""
Simple Diffusion Model for Weather Downscaling (Super-Resolution)
==================================================================
Trains a conditional DDPM to upscale 4x-downsampled weather fields to full
resolution. UNet backbone with sinusoidal time embedding, 100 linear-beta
timesteps, noise-prediction objective.

Input: 4x-downsampled TMP:2m, DPT:2m, UGRD:10m, VGRD:10m (bilinear-upscaled
back to target resolution as conditioning). Target: full-res original fields.

Pipeline: wxforge fetches HRRR GRIB2 subsets, decodes to NPY, Python handles
downsampling/upsampling and diffusion training.
Usage:  python train_diffusion.py
Requires: torch, numpy, wxforge binary
"""
from __future__ import annotations
import json, math, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

WORK_DIR = Path(__file__).resolve().parent / "_diffusion_workdir"
FHRS = list(range(0, 13))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BATCH_SIZE, LR, CROP = 8, 2, 2e-4, 128
SCALE, NCH, T_STEPS = 4, 4, 100
BETA_START, BETA_END = 1e-4, 0.02
FIELDS = ["TMP:2 m above ground", "DPT:2 m above ground",
          "UGRD:10 m above ground", "VGRD:10 m above ground"]
FKEYS = ["2t", "2d", "10u", "10v"]

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

class DownscaleDataset(Dataset):
    def __init__(self, sample_dirs, crop=CROP, scale=SCALE):
        self.samples, self.scale = [], scale
        for d in sample_dirs:
            m = json.loads((d / "sample_manifest.json").read_text())
            ch = {c["name"]: d / c["data_file"] for c in m["channels"]}
            if not all(k in ch for k in FKEYS): continue
            arrs = [np.load(str(ch[k])).astype(np.float32) for k in FKEYS]
            h, w = arrs[0].shape
            ch_, cw = (min(h,crop)//scale)*scale, (min(w,crop)//scale)*scale
            y0, x0 = (h-ch_)//2, (w-cw)//2
            hi = np.stack([a[y0:y0+ch_, x0:x0+cw] for a in arrs])
            for c in range(hi.shape[0]):
                mu, s = hi[c].mean(), hi[c].std()+1e-8; hi[c] = (hi[c]-mu)/s
            self.samples.append(torch.from_numpy(hi))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        hi = self.samples[i]
        lo = F.avg_pool2d(hi.unsqueeze(0), self.scale).squeeze(0)
        lo_up = F.interpolate(lo.unsqueeze(0), scale_factor=self.scale,
                              mode="bilinear", align_corners=False).squeeze(0)
        return lo_up, hi  # condition, target

# -- Diffusion schedule ----------------------------------------------------
class DiffSchedule:
    def __init__(self, T=T_STEPS):
        self.T = T
        betas = torch.linspace(BETA_START, BETA_END, T)
        alphas = 1.0 - betas; abar = torch.cumprod(alphas, 0)
        self.betas = betas
        self.sqrt_abar = torch.sqrt(abar)
        self.sqrt_1m_abar = torch.sqrt(1-abar)
        self.sqrt_recip_a = torch.sqrt(1/alphas)
        self.coef = betas / self.sqrt_1m_abar
    def q_sample(self, x0, t, noise=None):
        if noise is None: noise = torch.randn_like(x0)
        sa = self.sqrt_abar[t].view(-1,1,1,1).to(x0.device)
        sb = self.sqrt_1m_abar[t].view(-1,1,1,1).to(x0.device)
        return sa*x0 + sb*noise, noise
    def p_sample(self, model, xt, t_idx, cond):
        t = torch.full((xt.shape[0],), t_idx, device=xt.device, dtype=torch.long)
        eps = model(xt, t, cond)
        mean = self.sqrt_recip_a[t_idx].to(xt.device) * (xt - self.coef[t_idx].to(xt.device)*eps)
        if t_idx > 0:
            return mean + torch.sqrt(self.betas[t_idx]).to(xt.device) * torch.randn_like(xt)
        return mean

# -- UNet with time embedding -----------------------------------------------
class SinEmb(nn.Module):
    def __init__(self, d): super().__init__(); self.d = d
    def forward(self, t):
        h = self.d//2; e = math.log(10000)/(h-1)
        e = torch.exp(torch.arange(h, device=t.device, dtype=torch.float32)*-e)
        e = t.float().unsqueeze(1)*e.unsqueeze(0)
        return torch.cat([e.sin(), e.cos()], 1)

class CBlock(nn.Module):
    def __init__(self, ic, oc, td):
        super().__init__()
        self.c1=nn.Conv2d(ic,oc,3,padding=1); self.c2=nn.Conv2d(oc,oc,3,padding=1)
        self.b1=nn.BatchNorm2d(oc); self.b2=nn.BatchNorm2d(oc); self.tm=nn.Linear(td,oc)
    def forward(self, x, te):
        h = F.relu(self.b1(self.c1(x)))
        h = h + self.tm(te).unsqueeze(-1).unsqueeze(-1)
        return F.relu(self.b2(self.c2(h)))

class DiffUNet(nn.Module):
    def __init__(self, ic=NCH, cc=NCH, b=32):
        super().__init__()
        td = b*4
        self.temb = nn.Sequential(SinEmb(b), nn.Linear(b,td), nn.GELU(), nn.Linear(td,td))
        self.enc1=CBlock(ic+cc, b, td); self.enc2=CBlock(b, b*2, td)
        self.bot=CBlock(b*2, b*4, td)
        self.up2=nn.ConvTranspose2d(b*4,b*2,2,stride=2); self.dec2=CBlock(b*4,b*2,td)
        self.up1=nn.ConvTranspose2d(b*2,b,2,stride=2); self.dec1=CBlock(b*2,b,td)
        self.head=nn.Conv2d(b, ic, 1); self.pool=nn.MaxPool2d(2)
    def forward(self, xn, t, cond):
        te = self.temb(t); x = torch.cat([xn, cond], 1)
        e1 = self.enc1(x, te); e2 = self.enc2(self.pool(e1), te)
        b = self.bot(self.pool(e2), te)
        d2 = self.dec2(torch.cat([self.up2(b), e2], 1), te)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1), te)
        return self.head(d1)

def main():
    print("="*60+"\nDiffusion Model for Weather Downscaling\n"+"="*60)
    wxf = find_wxforge()
    print(f"wxforge: {wxf}\nDevice:  {DEVICE}\n")

    print("[1/4] Fetching HRRR surface data...")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    dirs = fetch_and_build(wxf)
    print(f"  Got {len(dirs)} samples.\n")

    print("[2/4] Building downscaling dataset...")
    ds = DownscaleDataset(dirs)
    print(f"  Valid samples: {len(ds)}")
    if len(ds) < 2: sys.exit("ERROR: Need >=2 samples.")
    nt = max(1, len(ds)//4)
    tr = DownscaleDataset.__new__(DownscaleDataset); tr.samples=ds.samples[:-nt]; tr.scale=SCALE
    te = DownscaleDataset.__new__(DownscaleDataset); te.samples=ds.samples[-nt:]; te.scale=SCALE
    loader = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"  Train: {len(tr)}, Test: {len(te)}\n")

    print("[3/4] Training diffusion model...")
    sched = DiffSchedule()
    model = DiffUNet().to(DEVICE)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Diffusion timesteps: {T_STEPS}")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS+1):
        model.train(); tl=0
        for cond, tgt in loader:
            cond, tgt = cond.to(DEVICE), tgt.to(DEVICE)
            t = torch.randint(0, sched.T, (cond.shape[0],), device=DEVICE)
            noisy, noise = sched.q_sample(tgt, t)
            loss = F.mse_loss(model(noisy, t, cond), noise)
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item()
        print(f"  Epoch {ep}/{EPOCHS}  noise_MSE={tl/max(len(loader),1):.6f}")

    print("\n[4/4] Generating high-res sample via reverse diffusion...")
    model.eval(); cond, truth = te[0]
    cb = cond.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        xt = torch.randn(1, NCH, cond.shape[1], cond.shape[2], device=DEVICE)
        for t in reversed(range(sched.T)): xt = sched.p_sample(model, xt, t, cb)
    gen = xt.cpu().squeeze(0)
    bl_rmse = torch.sqrt(torch.mean((cond-truth)**2)).item()
    df_rmse = torch.sqrt(torch.mean((gen-truth)**2)).item()
    print(f"\n  Bilinear baseline RMSE: {bl_rmse:.4f}")
    print(f"  Diffusion output RMSE:  {df_rmse:.4f}")
    print(f"  Improvement: {(1-df_rmse/bl_rmse)*100:+.1f}%")
    names = ["TMP:2m", "DPT:2m", "UGRD:10m", "VGRD:10m"]
    for c, nm in enumerate(names):
        b = torch.sqrt(torch.mean((cond[c]-truth[c])**2)).item()
        d = torch.sqrt(torch.mean((gen[c]-truth[c])**2)).item()
        print(f"    {nm:<10s}: baseline={b:.4f}  diffusion={d:.4f}")
    ckpt = WORK_DIR / "diffusion_downscaler.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"\n  Model saved to {ckpt}\n"+"="*60+"\nDone!")

if __name__ == "__main__":
    main()

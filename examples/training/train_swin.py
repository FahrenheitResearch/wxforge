"""
Swin Transformer for 6-Hour Weather Forecasting
=================================================
Trains a simplified Swin-like transformer to predict weather fields 6h ahead.
Input/target: TMP:2m, DPT:2m, UGRD:10m, VGRD:10m, CAPE:surface.
Architecture: patch embedding + 2 Swin blocks (window self-attention) + linear head.

Pipeline: wxforge fetches HRRR GRIB2 for fhr 0-18, pairs (T, T+6h) for training.
Usage:  python train_swin.py
Requires: torch, numpy, wxforge binary
"""
from __future__ import annotations
import json, math, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

WORK_DIR = Path(__file__).resolve().parent / "_swin_workdir"
FHRS = list(range(0, 19))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BATCH_SIZE, LR, CROP = 5, 1, 3e-4, 128
PATCH, NCH, EDIM, NHEAD, NBLK, WIN = 8, 5, 128, 4, 2, 4
FIELDS = ["TMP:2 m above ground", "DPT:2 m above ground",
          "UGRD:10 m above ground", "VGRD:10 m above ground", "CAPE:surface"]
FKEYS = ["2t", "2d", "10u", "10v", "cape"]

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
    result = {}
    for fhr in FHRS:
        d = WORK_DIR / f"fhr_{fhr:03d}"
        if d.exists() and (d / "sample_manifest.json").exists():
            result[fhr] = d; continue
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
        if r.returncode == 0: result[fhr] = d
    return result

def load_sample(d, crop):
    m = json.loads((d / "sample_manifest.json").read_text())
    ch = {c["name"]: d / c["data_file"] for c in m["channels"]}
    if not all(k in ch for k in FKEYS): return None
    arrs = [np.load(str(ch[k])).astype(np.float32) for k in FKEYS]
    h, w = arrs[0].shape; ch_, cw = min(h,crop), min(w,crop)
    y0, x0 = (h-ch_)//2, (w-cw)//2
    return np.stack([a[y0:y0+ch_, x0:x0+cw] for a in arrs])

class ForecastDataset(Dataset):
    def __init__(self, fhr_map, crop=CROP, lead=6):
        self.pairs = []
        for fhr in sorted(fhr_map):
            if fhr+lead not in fhr_map: continue
            inp, tgt = load_sample(fhr_map[fhr], crop), load_sample(fhr_map[fhr+lead], crop)
            if inp is None or tgt is None: continue
            for c in range(inp.shape[0]):
                mu, s = inp[c].mean(), inp[c].std()+1e-8
                inp[c] = (inp[c]-mu)/s; tgt[c] = (tgt[c]-mu)/s
            self.pairs.append((torch.from_numpy(inp), torch.from_numpy(tgt)))
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i): return self.pairs[i]

# -- Swin Architecture -----------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, ic, dim, ps):
        super().__init__(); self.proj=nn.Conv2d(ic,dim,ps,stride=ps); self.norm=nn.LayerNorm(dim)
    def forward(self, x):
        x=self.proj(x); B,C,H,W=x.shape
        return self.norm(x.flatten(2).transpose(1,2)), H, W

class WinAttn(nn.Module):
    def __init__(self, dim, nh, ws):
        super().__init__()
        self.nh, self.ws, self.scale = nh, ws, (dim//nh)**-0.5
        self.qkv=nn.Linear(dim, dim*3); self.proj=nn.Linear(dim,dim)
    def forward(self, x, H, W):
        B,N,C = x.shape; ws=self.ws
        x = x.view(B,H,W,C)
        pH, pW = (ws-H%ws)%ws, (ws-W%ws)%ws
        x = F.pad(x, (0,0,0,pW,0,pH)); Hp,Wp = H+pH, W+pW
        x = x.view(B,Hp//ws,ws,Wp//ws,ws,C).permute(0,1,3,2,4,5).reshape(-1,ws*ws,C)
        qkv = self.qkv(x).reshape(-1,ws*ws,3,self.nh,C//self.nh).permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2]
        attn = (q@k.transpose(-2,-1))*self.scale
        x = (attn.softmax(-1)@v).transpose(1,2).reshape(-1,ws*ws,C)
        x = self.proj(x).view(B,Hp//ws,Wp//ws,ws,ws,C).permute(0,1,3,2,4,5).reshape(B,Hp,Wp,C)
        return x[:,:H,:W,:].reshape(B,N,C)

class SwinBlock(nn.Module):
    def __init__(self, dim, nh, ws):
        super().__init__()
        self.n1=nn.LayerNorm(dim); self.attn=WinAttn(dim,nh,ws)
        self.n2=nn.LayerNorm(dim)
        self.ffn=nn.Sequential(nn.Linear(dim,dim*4), nn.GELU(), nn.Linear(dim*4,dim))
    def forward(self, x, H, W):
        x = x + self.attn(self.n1(x), H, W); return x + self.ffn(self.n2(x))

class SwinForecaster(nn.Module):
    def __init__(self, ic=NCH, oc=NCH, ps=PATCH, dim=EDIM, nh=NHEAD, nb=NBLK, ws=WIN):
        super().__init__()
        self.ps=ps; self.oc=oc; self.embed=PatchEmbed(ic,dim,ps)
        self.blocks=nn.ModuleList([SwinBlock(dim,nh,ws) for _ in range(nb)])
        self.norm=nn.LayerNorm(dim); self.head=nn.Linear(dim, oc*ps*ps)
    def forward(self, x):
        B,_,Hi,Wi = x.shape; tok,H,W = self.embed(x)
        for blk in self.blocks: tok=blk(tok,H,W)
        out = self.head(self.norm(tok)).view(B,H,W,self.oc,self.ps,self.ps)
        out = out.permute(0,3,1,4,2,5).reshape(B,self.oc,H*self.ps,W*self.ps)
        return out[:,:,:Hi,:Wi]

def main():
    print("="*60+"\nSwin Transformer 6h Forecasting\n"+"="*60)
    wxf = find_wxforge()
    print(f"wxforge: {wxf}\nDevice:  {DEVICE}\n")

    print("[1/4] Fetching HRRR surface data (fhr 0-18)...")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    fhr_map = fetch_and_build(wxf)
    print(f"  Got {len(fhr_map)} forecast hours.\n")

    print("[2/4] Building forecast-pair dataset (T -> T+6h)...")
    ds = ForecastDataset(fhr_map)
    print(f"  Valid pairs: {len(ds)}")
    if len(ds) < 2: sys.exit("ERROR: Need >=2 pairs.")
    nt = max(1, len(ds)//5)
    train_pairs, test_pairs = ds.pairs[:-nt], ds.pairs[-nt:]
    tr = ForecastDataset.__new__(ForecastDataset); tr.pairs = train_pairs
    te = ForecastDataset.__new__(ForecastDataset); te.pairs = test_pairs
    loader = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"  Train: {len(tr)}, Test: {len(te)}\n")

    print("[3/4] Training Swin Transformer...")
    model = SwinForecaster().to(DEVICE)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.SmoothL1Loss()
    for ep in range(1, EPOCHS+1):
        model.train(); tl=0
        for inp, tgt in loader:
            loss = crit(model(inp.to(DEVICE)), tgt.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item()
        print(f"  Epoch {ep}/{EPOCHS}  loss={tl/max(len(loader),1):.6f}")

    print("\n[4/4] Inference on test pair...")
    model.eval(); inp, tgt = te[0]
    with torch.no_grad(): pred = model(inp.unsqueeze(0).to(DEVICE)).cpu().squeeze(0)
    names = ["TMP:2m", "DPT:2m", "UGRD:10m", "VGRD:10m", "CAPE"]
    print(f"\n  Per-channel RMSE (normalized):")
    for c, nm in enumerate(names):
        rmse = torch.sqrt(torch.mean((pred[c]-tgt[c])**2)).item()
        print(f"    {nm:<10s}: {rmse:.4f}")
    print(f"    {'Total':<10s}: {torch.sqrt(torch.mean((pred-tgt)**2)).item():.4f}")
    ckpt = WORK_DIR / "swin_forecaster.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"\n  Model saved to {ckpt}\n"+"="*60+"\nDone!")

if __name__ == "__main__":
    main()

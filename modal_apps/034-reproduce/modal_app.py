# ========================================================================
# Quick Start
#   pip install modal
#   modal setup          # authenticate once (opens browser)
#   modal run modal_app.py
# 
#   App:  034-unic-reproduce
#   Mode: reproduce
#   Paper [034]: Beyond Image Borders: Learning Feature Extrapolation for Unbounded Image Composition
#   Generated: 2026-06-26
#   Repo: https://github.com/liuxiaoyu1104/UNIC
# 
# Cost Estimate
#   GPU:            T4:2
#   Price/hr:       $0.59 per GPU = $1.18/hr total
#   Est. duration:  20.0 hr (batch=2/GPU — OOM at batch=4, halved again)
#   Est. total:     $23.60
#   Note:           batch=4 OOM at step 170/242 epoch 0 (13.84GB peak, T4 usable=14.56GB).
#                   Reduced to batch=2. L4 (24GB) would fit batch=4: $0.80x2x10h=$16 cheaper.
#
# VRAM Analysis (observed from actual run)
#   At batch=4:     13.84 GB peak (OOM at step 170, epoch 0)
#   T4 usable:      14.56 GiB (not full 16GB — driver overhead ~1.4GB)
#   At batch=2:     ~7-8 GB peak (estimated, halved activations)
#   Chosen GPU:     T4:2 (16 GB VRAM each)
#   Note:           Memory grew 10.4→13.8GB across epoch 0 (EMA model accumulates state).
#                   batch=2 gives ~6GB headroom for EMA growth.
# ========================================================================

import modal
import os
import sys
from pathlib import Path

app = modal.App("034-unic-reproduce")

# ── Container image ──────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.10")
    .run_commands(
        "apt-get update -qq",
        "apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git",
    )
    # torch 1.13.1 required: attention.py copies pytorch internal C APIs (_infer_size,
    # _add_docstr, _VF, _jit_internal) that were restructured/removed in torch 2.x.
    # ema-pytorch and timm both declare torch>=2.0; install --no-deps to prevent pip
    # from upgrading our pinned torch 1.13.1. Their only real runtime dep is torch itself.
    .run_commands(
        "pip install "
        "torch==1.13.1+cu117 torchvision==0.14.1+cu117 "
        "--extra-index-url https://download.pytorch.org/whl/cu117 && "
        "pip install 'numpy<2' tqdm scipy Pillow requests einops opencv-python-headless matplotlib pycocotools && "
        "pip install timm==0.6.13 --no-deps && "
        "pip install ema-pytorch --no-deps && "
        "pip install pyyaml huggingface_hub"  # timm/ema-pytorch --no-deps skip these; add back manually
    )
)

# ── Persistent volumes ───────────────────────────────────────────────────
unic_034_data = modal.Volume.from_name("034-unic-data", create_if_missing=True)
unic_034_weights = modal.Volume.from_name("034-unic-weights", create_if_missing=True)
unic_034_output = modal.Volume.from_name("034-unic-output", create_if_missing=True)

# ── Main function ────────────────────────────────────────────────────────
def _download_file(url: str, dest: str) -> None:
    """Download a single file with progress bar."""
    import requests
    from tqdm import tqdm
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"  [cache] {dest_path.name}")
        return
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        desc=dest_path.name, total=total, unit="B",
        unit_scale=True, leave=False
    ) as bar:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"  [done]  {dest_path.name}")


@app.function(
    gpu="T4:2",
    image=image,
    timeout=36000,
    volumes={
        "/data": unic_034_data,
        "/weights": unic_034_weights,
        "/output": unic_034_output,
    },
)
def run_unic_training(config: dict = None) -> dict:
    """Full UNIC training pipeline on GAICD dataset.

    Paper 034 — ICCV 2023: Beyond Image Borders (UNIC)
    Repo: https://github.com/liuxiaoyu1104/UNIC

    Replicates train.sh:
      python -m torch.distributed.launch --nproc_per_node=2 main.py \
        --epochs 50 --lr_drop 30 --label_class soft \
        --outside_ratio --outpainting

    Args (via config dict):
        data_path (str):       GAIC dataset root, default /data/GAIC
        output_dir (str):      checkpoint output dir, default /output/train
        epochs (int):          total epochs, default 50  (Section 4.1)
        lr_drop (int):         epoch to drop LR by 1/10, default 30  (Section 4.1)
        batch_size (int):      per-GPU batch size, default 4  (reduced from 8 for T4 16GB)
        lr (float):            transformer LR, default 1e-4  (Section 4.1)
        lr_backbone (float):   backbone LR, default 1e-5  (Section 4.1)
        resume (str):          path to init checkpoint (ImageNet DETR weights)
        label_class (str):     soft | hard  (Section 3.2: soft labels used)
        outside_ratio (bool):  enable unbounded composition mode, default True
        outpainting (bool):    enable FEM extrapolation, default True
    """
    config = config or {}

    # ── Data & weight setup ──────────────────────────────────────────
    import subprocess
    import json
    import time

    repo_path = Path("/repo/UNIC")
    if not repo_path.exists():
        subprocess.run(["git", "clone", "https://github.com/liuxiaoyu1104/UNIC", str(repo_path)], check=True)
    os.chdir(str(repo_path))

    # ── Paper implementation ─────────────────────────────────────────

    # Volume was uploaded with: modal volume put 034-unic-data <GAIC_dir> /data/GAIC
    # Modal stores paths relative to volume root; volume is mounted at /data in container.
    # Result: volume-internal path "data/GAIC/GAIC" → container path /data/data/GAIC/GAIC
    data_path    = config.get('data_path',    '/data/data/GAIC/GAIC')
    output_dir_t = config.get('output_dir',   '/output/train')
    epochs       = config.get('epochs',       50)    # Section 4.1: 50 epochs
    lr_drop      = config.get('lr_drop',      30)    # Section 4.1: drop at epoch 30
    batch_size   = config.get('batch_size',   2)     # reduced 8→2/GPU: T4 OOM at batch=4 (13.8GB peak on T4 14.5GB usable)
    lr           = config.get('lr',           1e-4)  # Section 4.1
    lr_backbone  = config.get('lr_backbone',  1e-5)  # Section 4.1
    label_class  = config.get('label_class',  'soft')  # Section 3.2: soft labels
    outside_ratio= config.get('outside_ratio', True)
    outpainting  = config.get('outpainting',   True)   # Section 3.3: enable FEM
    resume       = config.get('resume',        '')

    Path(output_dir_t).mkdir(parents=True, exist_ok=True)

    # ── Verify dataset exists ─────────────────────────────────────────────────
    # Section 3.4 + 4.1: GAICD dataset — 2636 train / 200 val / 500 test images
    # Check the annotations file directly (Modal FUSE volume: dir.exists() can be unreliable)
    ann_file = Path(data_path) / 'annotations' / 'instances_train.json'
    if not ann_file.exists():
        # List what's actually in /data to help debug
        data_contents = list(Path('/data').rglob('*'))[:20] if Path('/data').exists() else []
        raise FileNotFoundError(
            f'GAIC annotations not found at {ann_file}.\n'
            f'Volume /data contents (first 20): {data_contents}\n'
            f'Re-upload: modal volume put 034-unic-data <local_GAIC_dir> /data/GAIC'
        )

    # ── Patch hardcoded CUDA device in main.py ───────────────────────────────
    # main.py hardcodes os.environ["CUDA_VISIBLE_DEVICES"] = "4" which breaks
    # Modal containers that expose 0-indexed devices. Patch it out.
    main_py = repo_path / 'main.py'
    content = main_py.read_text()
    if 'CUDA_VISIBLE_DEVICES' in content:
        patched = content.replace(
            'os.environ["CUDA_VISIBLE_DEVICES"] = "4"',
            '# patched: Modal exposes GPUs as 0,1 — do not hardcode device index'
        )
        main_py.write_text(patched)
        print('[patch] Removed hardcoded CUDA_VISIBLE_DEVICES=4 from main.py')


    # ── Build training command ─────────────────────────────────────────────────
    # Mirrors train.sh exactly (Section 4.1 training setup)
    # Modal provides 2×A10G, each 24GB — matches authors' 2-GPU setup
    nproc = 2  # matches nproc_per_node=2 in train.sh
    cmd = [
        'python', '-m', 'torch.distributed.launch',
        f'--nproc_per_node={nproc}',
        '--master_port=29500',
        '--use_env',
        'main.py',
        '--coco_path',    data_path,
        '--output_dir',   output_dir_t,
        '--epochs',       str(epochs),
        '--lr_drop',      str(lr_drop),
        '--lr',           str(lr),
        '--lr_backbone',  str(lr_backbone),
        '--batch_size',   str(batch_size),
        '--num_workers',  '4',
        '--label_class',  label_class,
    ]
    if outside_ratio:
        cmd.append('--outside_ratio')   # Section 3.1: unbounded box regression
    if outpainting:
        cmd.append('--outpainting')     # Section 3.3: enable FEM module
    if resume and Path(resume).exists():
        cmd += ['--resume', resume]     # optional: init from ImageNet DETR weights
    else:
        # main.py argparse default is './output/train_init/detr-r50_1.pth' (always truthy).
        # Override to empty string so the `if args.resume:` guard in main.py skips the load.
        cmd += ['--resume', '']

    print(f'[UNIC training] Command:\n  {" ".join(cmd)}')
    print(f'[UNIC training] Config: epochs={epochs}, lr={lr}, lr_backbone={lr_backbone}, batch={batch_size}/GPU x {nproc} GPUs')

    start = time.time()
    # capture_output=False: let stdout/stderr stream directly to Modal logs
    result = subprocess.run(cmd, cwd=str(repo_path))
    elapsed = (time.time() - start) / 3600

    if result.returncode != 0:
        raise RuntimeError(f'Training failed with exit code {result.returncode}')

    # ── Collect output checkpoints ────────────────────────────────────────────
    checkpoints = sorted(Path(output_dir_t).glob('checkpoint*.pth'))
    final_ckpt  = str(checkpoints[-1]) if checkpoints else 'not found'

    print(f'[UNIC training] Completed in {elapsed:.2f}h')
    print(f'[UNIC training] Final checkpoint: {final_ckpt}')
    print(f'[UNIC training] All checkpoints: {[c.name for c in checkpoints]}')

    return {
        'status':          'success',
        'elapsed_hours':   round(elapsed, 2),
        'epochs_trained':  epochs,
        'final_checkpoint': final_ckpt,
        'output_dir':      output_dir_t,
        'checkpoints':     [str(c) for c in checkpoints],
    }

# ── Local entrypoint ─────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    """Local entry point — runs run_unic_training on Modal.

    IMPORTANT: Always run with --detach flag to avoid job being cancelled when
    the terminal closes:
        modal run --detach modal_apps/034-reproduce/modal_app.py

    Monitor: modal app logs 034-unic-reproduce --follow
    Status:  modal app list
    """
    config = {
        "mode": "reproduce",
        "resume": "/output/train/checkpoint.pth",  # resume from last saved checkpoint
    }
    print("Submitting job to Modal...")
    result = run_unic_training.remote(config)
    print("Done. Result:", result)


# ── Manual steps required before running ─────────────────────────────
# 1. Download GAICD dataset from Google Drive: https://drive.google.com/file/d/1tDdQqDe8dMoMIVi9Z0WWI5vtRViy01nR/view
# 2. Download GAICD annotations (instances_train.json + instances_test.json): https://drive.google.com/drive/folders/1xkZFXOTV4zebm660Dgprl_yaKxWAHh0H
# 3. Create local directory GAIC/ with subdirs: images/ and annotations/
# 4. Upload dataset to Modal Volume: modal volume put 034-unic-data ./GAIC /data/GAIC
# 5. Verify upload: modal volume ls 034-unic-data /data/GAIC/annotations/ (should show instances_train.json)
# 6. (Optional) Download init checkpoint (ImageNet DETR-R50) from: https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth and upload: modal volume put 034-unic-weights detr-r50-e632da11.pth /weights/detr-r50-init.pth
# 7. Run: pip install modal && modal setup (authenticate)
# 8. Run training: modal run modal_apps/034-reproduce/modal_app.py
# 9. Monitor progress: modal app logs 034-unic-reproduce (full stdout in real time)
# 10. After training, retrieve checkpoints: modal volume get 034-unic-output /output/train ./local_output/
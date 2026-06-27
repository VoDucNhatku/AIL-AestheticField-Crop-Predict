# ========================================================================
# Quick Start
#   pip install modal
#   modal setup          # authenticate once (opens browser)
#   modal run modal_app.py
# 
#   App:  034-unic-run-results
#   Mode: run-results
#   Paper [034]: Beyond Image Borders: Learning Feature Extrapolation for Unbounded Image Composition
#   Generated: 2026-06-26
#   Repo: https://github.com/liuxiaoyu1104/UNIC
# 
# Cost Estimate
#   GPU:            T4
#   Price/hr:       $0.59
#   Est. duration:  1.0 hr
#   Est. total:     $0.59
#   Note:           Run evaluation on GAICD test set (500 images, ~1hr on T4)
# 
# VRAM Analysis
#   Required:       ~6 GB
#   Safety margin:  +2 GB
#   Needed total:   8 GB
#   Chosen GPU:     T4 (16 GB VRAM)
#   Rationale:      ResNet-50 backbone (~25M) + cDETR encoder/decoder (~50M) + FEM (~15M) = ~90M params total. fp16 inference: 90M x 2 bytes = ~180MB weights. With feature maps, attention buffers, batch=8: ~6GB peak. T4 16GB provides comfortable headroom.
# ========================================================================

import modal
import os
import sys
from pathlib import Path

app = modal.App("034-unic-run-results")

# ── Container image ──────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.10")
    .run_commands(
        "apt-get update -qq",
        "apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git",
    )
    .run_commands(
        "pip install "
        "torch==1.13.1+cu117 torchvision==0.14.1+cu117 "
        "--extra-index-url https://download.pytorch.org/whl/cu117 && "
        "pip install 'numpy<2' tqdm scipy Pillow requests einops opencv-python-headless matplotlib pycocotools && "
        "pip install timm==0.6.13 --no-deps && "
        "pip install ema-pytorch --no-deps && "
        "pip install pyyaml huggingface_hub"
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
    gpu="T4",
    image=image,
    timeout=3600,
    volumes={
        "/data": unic_034_data,
        "/weights": unic_034_weights,
        "/output": unic_034_output,
    },
)
def run_unic_evaluation(config: dict = None) -> dict:
    """Run UNIC evaluation on GAICD test set using pretrained checkpoint.

    UNIC (Unbounded Image Composition) — paper 034, ICCV 2023.
    Repo: https://github.com/liuxiaoyu1104/UNIC

    Args:
        config: dict with optional keys:
            checkpoint_path: str — path to .pth file (default: /weights/checkpoint.pth)
            data_path: str — path to GAIC dataset root (default: /data/GAIC)
            output_dir: str — where to save results (default: /output/eval)
            batch_size: int — evaluation batch size (default: 4)
            outside_ratio: bool — enable unbounded composition mode (default: True)
    """
    config = config or {}

    import subprocess
    import json
    from pathlib import Path

    # ── Data & weight setup ──────────────────────────────────────────
    if not Path("/repo/UNIC").exists():
        subprocess.run(["git", "clone", "https://github.com/liuxiaoyu1104/UNIC", "/repo/UNIC"], check=True)
    os.chdir("/repo/UNIC")

    # ── Paper implementation ─────────────────────────────────────────

    # ── Patch hardcoded CUDA device in main.py ───────────────────────────────
    main_py = Path("/repo/UNIC") / 'main.py'
    content = main_py.read_text()
    if 'CUDA_VISIBLE_DEVICES' in content:
        patched = content.replace(
            'os.environ["CUDA_VISIBLE_DEVICES"] = "4"',
            '# patched: Modal exposes GPUs as 0,1 — do not hardcode device index'
        )
        main_py.write_text(patched)
        print('[patch] Removed hardcoded CUDA_VISIBLE_DEVICES=4 from main.py')


    checkpoint_path = config.get('checkpoint_path', '/weights/checkpoint.pth')
    data_path       = config.get('data_path',       '/data/data/GAIC/GAIC')
    output_dir_val  = config.get('output_dir',      '/output/eval')
    batch_size      = config.get('batch_size',      4)
    outside_ratio   = config.get('outside_ratio',   True)

    Path(output_dir_val).mkdir(parents=True, exist_ok=True)

    # ── Verify checkpoint exists ──────────────────────────────────────────────
    # Section 4.1: pretrained checkpoint available at Google Drive
    # (download manually and upload to Modal Volume before first run)
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f'Checkpoint not found at {checkpoint_path}.\n'
            f'Download from Google Drive: https://drive.google.com/file/d/1owvtFQBCC4uxd5f6tRsh7Fgktuj8MzZy/view\n'
            f'Then upload to Modal Volume: modal volume put 034-unic-weights <local_ckpt.pth> /weights/checkpoint.pth'
        )

    # ── Verify dataset exists ─────────────────────────────────────────────────
    # Section 3.4 + 4.1: GAICD dataset — 500 test images, COCO annotations format
    # Download from: https://drive.google.com/file/d/1tDdQqDe8dMoMIVi9Z0WWI5vtRViy01nR/view
    # Annotations: https://drive.google.com/drive/folders/1xkZFXOTV4zebm660Dgprl_yaKxWAHh0H
    if not Path(data_path).exists():
        raise FileNotFoundError(
            f'GAIC dataset not found at {data_path}.\n'
            f'See README: https://github.com/liuxiaoyu1104/UNIC#data-preparation'
        )

    repo_name = 'UNIC'
    repo_path = Path(f'/repo/{repo_name}')

    # ── Build evaluation command ──────────────────────────────────────────────
    # Based on test.sh: distributed eval with --eval flag
    # Section 4.1: batch_size=8, backbone=resnet50, enc/dec layers=6, nheads=8
    cmd = [
        'python', '-m', 'torch.distributed.launch',
        '--nproc_per_node=1',          # single GPU on Modal T4
        '--master_port=29500',
        '--use_env',
        'main.py',
        '--coco_path', data_path,
        '--output_dir', output_dir_val,
        '--resume', checkpoint_path,
        '--batch_size', str(batch_size),
        '--label_class', 'soft',
        '--eval',                      # evaluation mode only
        '--print_pic',                 # save visualizations
    ]
    if outside_ratio:
        cmd.append('--outside_ratio')  # Section 3.1: unbounded composition
        cmd.append('--outpainting')    # Section 3.3: enable FEM

    print(f'Running evaluation command:\n  {" ".join(cmd)}')
    result = subprocess.run(cmd, cwd=str(repo_path), capture_output=True, text=True)

    print('STDOUT:', result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print('STDERR:', result.stderr[-2000:])
        raise RuntimeError(f'Evaluation failed with code {result.returncode}')

    # ── Parse results ─────────────────────────────────────────────────────────
    # Expected metrics from Table 1 (paper Section 4.2):
    #   GAICD: Acc_1/5(e=0.90)=23.2, Acc_1/10(e=0.85)=72.8, IoU=0.801, Disp=0.048
    output_log = Path(output_dir_val) / 'eval_results.json'
    metrics = {}
    if output_log.exists():
        with open(output_log) as f:
            metrics = json.load(f)
    else:
        # Extract from stdout
        for line in result.stdout.split('\n'):
            if 'IoU' in line or 'Acc' in line or 'Disp' in line:
                metrics['raw_output_line'] = line.strip()

    return {
        'status': 'success',
        'metrics': metrics,
        'output_dir': output_dir_val,
        'checkpoint': checkpoint_path,
    }

# ── Local entrypoint ─────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    """Local entry point — runs run_unic_evaluation on Modal."""
    config = {
        "mode": "run-results",
        "checkpoint_path": "/output/train/checkpoint.pth", # Use the fully trained checkpoint
    }
    print(f"Submitting job to Modal (mode=run-results)...")
    result = run_unic_evaluation.remote(config)
    print("Done. Result:", result)


# ── Manual steps required before running ─────────────────────────────
# 1. Download pretrained checkpoint from Google Drive: https://drive.google.com/file/d/1owvtFQBCC4uxd5f6tRsh7Fgktuj8MzZy/view
# 2. Upload checkpoint to Modal Volume: modal volume put 034-unic-weights <local_checkpoint.pth> /weights/checkpoint.pth
# 3. Download GAICD dataset from Google Drive: https://drive.google.com/file/d/1tDdQqDe8dMoMIVi9Z0WWI5vtRViy01nR/view
# 4. Download GAICD annotations: https://drive.google.com/drive/folders/1xkZFXOTV4zebm660Dgprl_yaKxWAHh0H
# 5. Upload dataset to Modal Volume: modal volume put 034-unic-data <local_GAIC_dir> /data/GAIC
# 6. Place annotation files under /data/GAIC/annotations/ (instances_train.json, instances_test.json)
"""Environment check: Python/PyTorch/CUDA versions, GPU, RAM, disk, and a tiny GPU training step.

Usage:  python -m src.check_env
Downloads nothing and writes no files.
GPU tests run separately (cuBLAS, convolution without cuDNN, with cuDNN) to locate failures.
"""
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import psutil
import torch
import torchvision

from src import config


def gb(n_bytes: float) -> str:
    return f"{n_bytes / 1024**3:.1f} GB"


def main() -> int:
    print("=== Software ===")
    print(f"Python       : {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform     : {platform.platform()}")
    print(f"torch        : {torch.__version__}")
    print(f"torchvision  : {torchvision.__version__}")
    print(f"CUDA (build) : {torch.version.cuda}")
    print(f"cuDNN        : {torch.backends.cudnn.version()}")

    print("\n=== Memory / disk ===")
    vm = psutil.virtual_memory()
    print(f"RAM total    : {gb(vm.total)}")
    print(f"RAM available: {gb(vm.available)}")
    if vm.available < 4 * 1024**3:
        print("WARNING      : less than 4 GB RAM available - close other apps before training.")
    du = shutil.disk_usage(config.PROJECT_ROOT)
    print(f"Disk free    : {gb(du.free)} (drive of {config.PROJECT_ROOT})")
    print(f"CPU threads  : {psutil.cpu_count(logical=True)}")

    print("\n=== GPU ===")
    if not torch.cuda.is_available():
        print("CUDA available: NO -> training will run on CPU (or install the matching CUDA build).")
        return 0

    props = torch.cuda.get_device_properties(0)
    print("CUDA available: yes")
    print(f"Device       : {props.name} (compute capability {props.major}.{props.minor})")
    print(f"GPU memory   : {gb(props.total_memory)}")

    report_dll_paths()

    # Test each layer separately so a failure points at the right component.
    print("\n=== GPU tests ===")
    tests = {
        "1. CUDA + cuBLAS (matrix multiply)": test_matmul,
        "2. Convolution WITHOUT cuDNN": lambda: test_conv(use_cudnn=False),
        "3. Convolution WITH cuDNN": lambda: test_conv(use_cudnn=True),
    }
    all_ok = True
    for name, fn in tests.items():
        print(f"{name:38s}: running ...", flush=True)   # printed first, so a hard crash is attributable
        msg = run_test(fn)
        all_ok &= msg.startswith("OK")
        print(f"{name:38s}: {msg}", flush=True)
    return 0 if all_ok else 1


def report_dll_paths() -> None:
    """List CUDA/cuDNN DLL locations that could clash with the ones bundled in PyTorch."""
    print("\n=== CUDA / cuDNN libraries ===")
    torch_lib = Path(torch.__file__).parent / "lib"
    bundled = sorted(p.name for p in torch_lib.glob("cudnn*.dll"))
    print(f"PyTorch lib folder : {torch_lib}")
    print(f"Bundled cuDNN DLLs : {len(bundled)} ({', '.join(bundled[:3])}{' ...' if len(bundled) > 3 else ''})")
    print(f"CUDA_PATH env var  : {os.environ.get('CUDA_PATH', '(not set)')}")
    suspicious = []
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        d = Path(entry)
        try:
            dlls = [p.name for p in d.glob("cudnn*.dll")] + [p.name for p in d.glob("cublas*.dll")]
        except OSError:
            dlls = []
        if dlls or any(k in entry.lower() for k in ("cuda", "cudnn")):
            suspicious.append((entry, sorted(dlls)[:4]))
    if suspicious:
        print("CUDA/cuDNN folders on PATH (possible conflict):")
        for entry, dlls in suspicious:
            print(f"   {entry}  {dlls if dlls else ''}")
    else:
        print("CUDA/cuDNN folders on PATH: none")


def run_test(fn) -> str:
    try:
        return "OK " + fn()
    except Exception as exc:  # noqa: BLE001 - report any failure
        return f"FAILED ({type(exc).__name__}: {str(exc).splitlines()[0]})"


def test_matmul() -> str:
    a = torch.randn(1024, 1024, device="cuda")
    start = time.perf_counter()
    (a @ a).sum().item()
    torch.cuda.synchronize()
    return f"({time.perf_counter() - start:.2f}s)"


def test_conv(use_cudnn: bool) -> str:
    """Tiny forward+backward pass, same shapes as a real training batch."""
    torch.backends.cudnn.enabled = use_cudnn
    torch.manual_seed(config.SEED)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(16, 12),
    ).cuda()
    x = torch.randn(config.BATCH_SIZE, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device="cuda")
    y = torch.randint(0, 12, (config.BATCH_SIZE,), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**2
    torch.backends.cudnn.enabled = True
    return f"({time.perf_counter() - start:.2f}s, peak GPU mem {peak:.0f} MB)"

if __name__ == "__main__":
    sys.exit(main())

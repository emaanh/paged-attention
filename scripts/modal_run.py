"""Build and run this repo on a Modal GPU.

    modal run scripts/modal_run.py --cmd "./build/decode_tests"
    modal run scripts/modal_run.py --cmd "./build/bench_decode sweep" --out results/sweep.csv
    modal run scripts/modal_run.py --cmd "python scripts/compare_flashinfer.py" --out results/flashinfer.csv

The command runs from the repo root after a CMake Release build. Stdout is
printed and, with --out, also written to that local path. Set GPU=H100 (default),
L40S, A100-80GB, B200, ... to pick hardware.
"""

import os
import pathlib
import subprocess

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent
GPU = os.environ.get("GPU", "H100")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install("cmake>=3.29", "ninja", "numpy", "pandas")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("flashinfer-python")
    .env({"TORCH_CUDA_ARCH_LIST": "9.0", "FLASHINFER_WORKSPACE_BASE": "/cache"})
    .add_local_dir(REPO, "/repo", ignore=["build", ".git", "results", "**/__pycache__"])
)

# Persists FetchContent downloads, the CMake build tree and JIT caches between runs.
cache = modal.Volume.from_name("paged-attention-cache", create_if_missing=True)
app = modal.App("paged-attention", image=image)


@app.function(gpu=GPU, volumes={"/cache": cache}, timeout=3600)
def run(cmd: str) -> str:
    # Copy the mounted sources: mounts carry stale mtimes, which would make
    # Ninja skip rebuilding against the cached build tree.
    src, build = "/work", "/cache/build-work"
    subprocess.run(["cp", "-r", "/repo", src], check=True)
    os.makedirs(build, exist_ok=True)
    subprocess.run(["cmake", "-S", src, "-B", build, "-G", "Ninja",
                    "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=native"],
                   check=True, stdout=subprocess.DEVNULL)
    b = subprocess.run(["cmake", "--build", build, "-j", "16"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if b.returncode != 0:
        errors = [l for l in b.stdout.splitlines() if "error" in l.lower() or "FAILED" in l]
        return "# BUILD FAILED\n" + "\n".join(errors[:60]) + "\n"
    os.symlink(build, f"{src}/build")
    env = dict(os.environ, TORCH_EXTENSIONS_DIR="/cache/torch_ext")
    proc = subprocess.run(cmd, shell=True, cwd=src, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    cache.commit()
    return f"{proc.stdout}\n# exit code {proc.returncode}\n"


@app.local_entrypoint()
def main(cmd: str = "./build/decode_tests", out: str = ""):
    text = run.remote(cmd)
    print(text)
    if out:
        path = REPO / out
        path.parent.mkdir(parents=True, exist_ok=True)
        # Keep CSV rows only; build noise and comments go to the console.
        rows = [l for l in text.splitlines() if l and not l.startswith("#") and "," in l]
        path.write_text("\n".join(rows) + "\n")
        print(f"wrote {path}")

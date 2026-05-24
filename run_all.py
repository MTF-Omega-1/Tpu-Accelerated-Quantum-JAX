"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        JAX Quantum Simulator — Master Research Runner                       ║
║        Run this ONE file to execute ALL experiments automatically.          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage (WSL2 GPU):
    source ~/jax_gpu_env/bin/activate
    cd /mnt/c/Users/mswuk/Desktop/qauntum\ machine\ learning
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    python3 run_all.py

Usage (Windows CPU):
    python run_all.py
"""

import os
import sys
import time
import traceback
import subprocess
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# ⚠  These MUST be set BEFORE importing JAX — they configure the XLA runtime.
# ─────────────────────────────────────────────────────────────────────────────

# 1. Stop JAX from pre-allocating 90% of VRAM on startup.
#    Without this, JAX grabs 3.6 GB immediately and leaves no room for workspaces.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# 2. Hard cap: allow JAX to use at most 90% of VRAM = 3.7 GB on a 4 GB card.
#    This leaves headroom for XLA workspace buffers.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")

# 3. Use the async CUDA allocator — avoids fragmentation, handles OOM gracefully.
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

# 5. Suppress XLA / TF C++ verbose logs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

# ─────────────────────────────────────────────────────────────────────────────
# Setup: ensure project root is on PYTHONPATH
# ─────────────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT, "results"),          exist_ok=True)
os.makedirs(os.path.join(ROOT, "examples", "plots"),exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Experiment Registry — add new experiments here
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENTS = [
    {
        "id":      1,
        "name":    "Quantum State Preparation (GHZ)",
        "module":  "examples.01_state_preparation",
        "func":    "run_state_prep",
        "desc":    "Learn a 3-qubit GHZ state using Adam optimizer + JAX autodiff.",
    },
    {
        "id":      2,
        "name":    "Variational Quantum Classifier (XOR)",
        "module":  "examples.02_vqc_classification",
        "func":    "run_vqc",
        "desc":    "Train a VQC on the non-linear XOR classification problem via jax.vmap.",
    },
    {
        "id":      3,
        "name":    "GPU VRAM Scaling Benchmark",
        "module":  "examples.03_benchmarks",
        "func":    "run_full_benchmark",
        "desc":    "Scale from 4→29 qubits, track VRAM usage, JIT speedup, throughput.",
    },
    {
        "id":      4,
        "name":    "VQE — H₂ Molecule Ground State",
        "module":  "examples.04_vqe_h2_molecule",
        "func":    "run_vqe",
        "desc":    "Find H₂ electronic ground state energy to chemical accuracy (<1.6 mHa).",
    },
    {
        "id":      5,
        "name":    "QAOA — MaxCut Optimization",
        "module":  "examples.05_qaoa_maxcut",
        "func":    "run_qaoa_study",
        "desc":    "Solve weighted MaxCut graph problem with QAOA depths p=1..5.",
    },
    {
        "id":      6,
        "name":    "Barren Plateau Research",
        "module":  "examples.06_barren_plateaus",
        "func":    "run_barren_plateau_study",
        "desc":    "Quantify exponential gradient vanishing vs circuit width and depth.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Pretty Printing Helpers
# ─────────────────────────────────────────────────────────────────────────────

WIDTH = 76

def banner():
    print()
    print("╔" + "═" * WIDTH + "╗")
    print("║" + "  JAX QUANTUM SIMULATOR — FULL RESEARCH SUITE".center(WIDTH) + "║")
    print("║" + "  Running all experiments automatically...".center(WIDTH)    + "║")
    print("╚" + "═" * WIDTH + "╝")
    print()

def section_header(exp_id, name, desc):
    print()
    print("┌" + "─" * WIDTH + "┐")
    title = f"  [{exp_id}/{len(EXPERIMENTS)}]  {name}"
    print("│" + title.ljust(WIDTH) + "│")
    desc_line = f"  {desc}"
    print("│" + desc_line.ljust(WIDTH) + "│")
    print("└" + "─" * WIDTH + "┘")
    print()

def success_footer(name, elapsed):
    print()
    print(f"  ✅  {name}  —  completed in {elapsed:.1f}s")
    print("  " + "─" * 50)

def error_footer(name, elapsed, exc):
    print()
    print(f"  ❌  {name}  —  FAILED after {elapsed:.1f}s")
    print(f"  Error: {type(exc).__name__}: {exc}")
    print("  " + "─" * 50)

def final_summary(log):
    print()
    print("╔" + "═" * WIDTH + "╗")
    print("║" + "  EXPERIMENT SUMMARY".center(WIDTH) + "║")
    print("╠" + "═" * WIDTH + "╣")
    total = len(log)
    passed = sum(1 for r in log if r["status"] == "OK")
    failed = total - passed
    for r in log:
        icon   = "✅" if r["status"] == "OK" else "❌"
        line   = f"  {icon}  [{r['id']}] {r['name']:<40}  {r['elapsed']:>7.1f}s  {r['status']}"
        print("║" + line.ljust(WIDTH) + "║")
    print("╠" + "═" * WIDTH + "╣")
    summary = f"  {passed}/{total} succeeded  |  {failed} failed  |  Total: {sum(r['elapsed'] for r in log):.1f}s"
    print("║" + summary.ljust(WIDTH) + "║")
    print("╚" + "═" * WIDTH + "╝")
    print()
    print("  📁 Results  → ./results/   (CSV + JSON)")
    print("  🖼  Plots    → ./examples/plots/  (PNG @ 180 DPI)")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Detect and print JAX backend info
# ─────────────────────────────────────────────────────────────────────────────

def print_environment():
    import jax
    backend = jax.default_backend()
    devices = jax.devices()
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    print(f"  {'Timestamp':<16}: {ts}")
    print(f"  {'JAX version':<16}: {jax.__version__}")
    print(f"  {'Backend':<16}: {backend.upper()}")
    print(f"  {'Devices':<16}: {devices}")

    if backend == "gpu":
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                 '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            gpu_info = result.stdout.strip()
            print(f"  {'GPU Info':<16}: {gpu_info}")
        except Exception:
            pass
    elif backend == "cpu":
        print()
        print("  ⚠️   WARNING: Running on CPU. For GPU acceleration:")
        print("       1. Use WSL2 with NVIDIA drivers installed")
        print("       2. Run: pip install \"jax[cuda12]\" \\")
        print("               -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
        print("       3. Verify: python3 -c \"import jax; print(jax.devices('gpu'))\"")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Dynamic module importer + function runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(exp: dict) -> dict:
    """Dynamically import and run a single experiment function."""
    import importlib.util

    mod_name        = "exp_" + exp["module"].replace(".", "_").replace("-", "_")
    abs_module_path = os.path.join(ROOT, exp["module"].replace(".", os.sep) + ".py")

    spec = importlib.util.spec_from_file_location(mod_name, abs_module_path)
    mod  = importlib.util.module_from_spec(spec)

    # ── Critical: inject ROOT into the module's sys.path BEFORE exec ─────────
    # Without this, 'from jax_qsim import ...' fails inside dynamically loaded
    # modules because they get a reference to the real sys object but the path
    # entry may not have been added yet when the module-level import runs.
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    # Also ensure PYTHONPATH env var is set (for any sub-processes spawned)
    ppath = os.environ.get("PYTHONPATH", "")
    if ROOT not in ppath.split(os.pathsep):
        os.environ["PYTHONPATH"] = ROOT + (os.pathsep + ppath if ppath else "")

    # Register in sys.modules so repeated imports work correctly
    sys.modules[mod_name] = mod

    # Execute the module (runs all module-level code including imports)
    spec.loader.exec_module(mod)

    func = getattr(mod, exp["func"])
    func()

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    banner()
    print_environment()

    log = []
    overall_start = time.perf_counter()

    for exp in EXPERIMENTS:
        section_header(exp["id"], exp["name"], exp["desc"])
        t0 = time.perf_counter()
        try:
            run_experiment(exp)
            elapsed = time.perf_counter() - t0
            success_footer(exp["name"], elapsed)
            log.append({**exp, "elapsed": elapsed, "status": "OK"})
        except Exception as e:
            elapsed = time.perf_counter() - t0
            error_footer(exp["name"], elapsed, e)
            traceback.print_exc()
            log.append({**exp, "elapsed": elapsed, "status": f"FAILED: {e}"})

    final_summary(log)

if __name__ == "__main__":
    main()

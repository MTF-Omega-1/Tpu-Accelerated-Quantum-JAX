#!/bin/bash
# ============================================================
#  run_gpu.sh - GPU-Accelerated JAX Quantum Simulator Launcher
#  Run this inside WSL2 to use your NVIDIA GeForce RTX 2050 GPU
# ============================================================

# Activate the GPU-enabled virtual environment
source ~/jax_gpu_env/bin/activate

# Set the project directory (adjust if needed)
cd /mnt/c/Users/mswuk/Desktop/qauntum\ machine\ learning

# Add project root to python path so jax_qsim is importable
export PYTHONPATH="$PYTHONPATH:/mnt/c/Users/mswuk/Desktop/qauntum machine learning"

# Verify GPU is being used by JAX
echo "==========================================="
echo "  JAX GPU Verification"
echo "==========================================="
python3 -c "
import jax
print('JAX version:', jax.__version__)
print('Devices:', jax.devices())
print('Default backend:', jax.default_backend())
d = jax.devices('gpu')
print('GPU devices:', d)
"

echo ""
echo "==========================================="
echo "  Choose an example to run:"
echo "  1) State Preparation (01_state_preparation.py)"
echo "  2) VQC Classifier     (02_vqc_classification.py)"
echo "  3) Benchmarks         (03_benchmarks.py)"
echo "  4) Run Test Suite     (pytest tests/)"
echo "==========================================="
echo ""

read -p "Enter choice [1-4]: " choice

case $choice in
    1) python3 examples/01_state_preparation.py ;;
    2) python3 examples/02_vqc_classification.py ;;
    3) python3 examples/03_benchmarks.py ;;
    4) python3 -m pytest tests/ -v ;;
    *) echo "Invalid choice." ;;
esac

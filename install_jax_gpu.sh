#!/bin/bash
# ============================================================
#  install_jax_gpu.sh - CUDA JAX GPU Installation for WSL2
# ============================================================
set -e

echo ""
echo "============================================="
echo "  JAX GPU Setup for NVIDIA GeForce RTX 2050 + WSL2"
echo "============================================="

# 1. Check GPU visibility
echo ""
echo "[1/5] Verifying GPU is visible in WSL2..."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# 2. Create virtual environment
echo ""
echo "[2/5] Setting up Python virtual environment at ~/jax_gpu_env..."
python3 -m venv ~/jax_gpu_env
source ~/jax_gpu_env/bin/activate
pip install --upgrade pip

# 3. Install CUDA-enabled jaxlib and jax
echo ""
echo "[3/5] Installing CUDA-enabled JAX (for CUDA 12.x)..."
pip install --upgrade "jax[cuda12]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 4. Install other dependencies
echo ""
echo "[4/5] Installing other dependencies (matplotlib, pytest)..."
pip install matplotlib pytest

# 5. Verify GPU backend
echo ""
echo "[5/5] Verifying JAX uses GPU backend..."
python3 -c "
import jax
print('JAX version:', jax.__version__)
print('All devices:', jax.devices())
print('GPU devices:', jax.devices('gpu'))
print('Default backend:', jax.default_backend())
import jax.numpy as jnp
x = jnp.ones((1000, 1000))
y = jnp.dot(x, x).block_until_ready()
print('GPU matmul SUCCESS - shape:', y.shape)
"

echo ""
echo "============================================="
echo "  Setup complete! To run examples:"
echo "  source ~/jax_gpu_env/bin/activate"
echo "  cd /mnt/c/Users/mswuk/Desktop/qauntum\ machine\ learning"
echo "  export PYTHONPATH=\$PYTHONPATH:\$(pwd)"
echo "  python3 examples/03_benchmarks.py"
echo "============================================="

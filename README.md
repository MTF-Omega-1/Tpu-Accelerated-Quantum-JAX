# JAX Quantum Simulator — High-Speed Differentiable Quantum State-Vector Simulator

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![JAX](https://img.shields.io/badge/JAX-0.10%2B-orange?style=for-the-badge)
![CUDA](https://img.shields.io/badge/CUDA-12.x-green?style=for-the-badge&logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-GPU%20%7C%20TPU%20%7C%20CPU-blueviolet?style=for-the-badge)

**A research-grade, hardware-accelerated quantum state-vector simulator built purely in JAX.  
Run differentiable quantum circuits on NVIDIA GPUs and Google TPUs.**

</div>

---

## 🌟 Highlights

| Feature | Details |
|---|---|
| 🔬 **Full Differentiability** | All parameterized gates support `jax.grad`, `jax.jacobian`, `jax.value_and_grad` |
| ⚡ **JIT Compilation** | `jax.jit` compiles entire circuits to XLA — GPU-optimized HLO kernels |
| 🎮 **GPU/TPU Acceleration** | Native CUDA 12 support via WSL2; tested on NVIDIA RTX 2050 (4 GB VRAM) |
| 🔄 **Vectorized Batching** | `jax.vmap` over parameter batches or data batches with zero overhead |
| 📐 **Tensor Contraction Engine** | Gate application via `jnp.tensordot` + axis permutation — O(2ⁿ) |
| 🧪 **Research Examples** | VQE (H₂ molecule), QAOA (MaxCut), Barren Plateaus, VQC Classification |
| 📊 **GPU Scaling Benchmarks** | Scales to 29 qubits (4 GB VRAM saturation), produces publication plots |
| 🔊 **Noise Simulation** | Quantum trajectory simulation with Kraus operators |

---

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Mathematical Formulation](#-mathematical-formulation)
- [Installation (GPU via WSL2)](#-installation-gpu-via-wsl2)
- [Quick Start](#-quick-start)
- [Research Examples](#-research-examples)
- [GPU Benchmark Results](#-gpu-benchmark-results)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [References](#-references)

---

## 🏗 Architecture

```
jax_qsim/
├── core.py          ← Tensor contraction engine (tensordot + transpose permutation)
├── ops.py           ← Standard & parameterized quantum gates (complex64 matrices)
├── observables.py   ← Pauli strings, Hamiltonians, expectation values, sampling
├── noise.py         ← Kraus operator channels (depolarizing, amplitude/phase damping)
└── circuit.py       ← Stateless circuit builder → pure JAX function compiler

examples/
├── 01_state_preparation.py   ← Gradient-based GHZ state preparation (Adam optimizer)
├── 02_vqc_classification.py  ← Variational Quantum Classifier: XOR (jax.vmap batch)
├── 03_benchmarks.py          ← GPU VRAM scaling: 4→29 qubits, 6-panel plots, CSV/JSON
├── 04_vqe_h2_molecule.py     ← VQE: H₂ ground state (chemical accuracy, PES curve)
├── 05_qaoa_maxcut.py         ← QAOA: MaxCut on weighted graphs (depth p=1..5)
└── 06_barren_plateaus.py     ← Barren plateau research (variance vs width, depth, 2D landscape)
```

---

## 📐 Mathematical Formulation

### State Vector Representation

An N-qubit quantum state is stored as a complex tensor:

$$|\psi\rangle \rightarrow \mathbf{\Psi} \in \mathbb{C}^{2 \times 2 \times \cdots \times 2} \quad (N \text{ dimensions})$$

Memory footprint: $2^N \times 8$ bytes (complex64)

### Tensor Gate Application

Applying a k-qubit unitary $U$ (matrix shape $2^k \times 2^k$, reshaped to tensor $\tilde{U}_{o_1\cdots o_k, i_1\cdots i_k}$) to target qubits $\mathbf{t} = \{t_1, \ldots, t_k\}$:

$$\Psi'_{q_0 \cdots q_{N-1}} = \sum_{i_1, \ldots, i_k} \tilde{U}_{q_{t_1}\cdots q_{t_k},\, i_1\cdots i_k} \cdot \Psi_{q_0 \cdots i_1 \cdots i_k \cdots q_{N-1}}$$

Implemented via `jnp.tensordot` + inverse permutation `jnp.transpose` — leverages tensor core acceleration on NVIDIA GPUs and matrix multiply units on TPUs.

### Differentiable Expectation Values

$$\mathcal{L}(\boldsymbol{\theta}) = \langle \psi(\boldsymbol{\theta}) | \hat{O} | \psi(\boldsymbol{\theta}) \rangle$$

$$\frac{\partial \mathcal{L}}{\partial \theta_i} = \text{(via JAX reverse-mode AD — exact, not finite difference)}$$

The Parameter Shift Rule is automatically satisfied since all gate matrices are analytic functions of their parameters.

---

## ⚙️ Installation (GPU via WSL2)

> **Windows + NVIDIA GPU** → must use WSL2 for GPU-accelerated JAX.  
> CPU-only Windows pip install will never use your GPU.

### Step 1 — WSL2 setup (one-time)
```bash
# In Windows PowerShell (as admin)
wsl --install
```

### Step 2 — Create GPU virtual environment
```bash
# Inside WSL2 terminal
python3 -m venv ~/jax_gpu_env
source ~/jax_gpu_env/bin/activate
pip install --upgrade pip
```

### Step 3 — Install CUDA-enabled JAX
```bash
pip install --upgrade "jax[cuda12]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Step 4 — Install remaining dependencies
```bash
pip install matplotlib pytest
```

### Step 5 — Install this project
```bash
cd /mnt/c/Users/<your-username>/Desktop/qauntum\ machine\ learning
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Step 6 — Verify GPU backend
```bash
python3 -c "
import jax
print('Backend  :', jax.default_backend())
print('Devices  :', jax.devices())
print('GPU check:', jax.devices('gpu'))
"
```
Expected output:
```
Backend  : gpu
Devices  : [CudaDevice(id=0)]
GPU check: [CudaDevice(id=0)]
```

---

## 🚀 Quick Start

```python
import jax
import jax.numpy as jnp
import jax_qsim as qsim

# 1. Build a 3-qubit parameterized circuit
c = qsim.Circuit(num_qubits=3)
c.h(0).cnot(0, 1).cnot(1, 2)        # GHZ entanglement
c.ry(0, param_index=0)               # Trainable RY on qubit 0
c.ry(1, param_index=1)               # Trainable RY on qubit 1
c.rz(2, param_index=2)               # Trainable RZ on qubit 2

# 2. Define observable: Pauli Z on qubit 0
obs = qsim.observables.PauliString({0: 'Z'})

# 3. Define differentiable cost function
def cost(params):
    state = c.run(params)
    return qsim.observables.expectation(state, obs)

# 4. Compile + compute gradient (runs on GPU)
grad_fn = jax.jit(jax.grad(cost))
params  = jnp.array([0.1, 0.5, -0.3])
print("Gradient:", grad_fn(params))

# 5. Vectorize over a batch of 100 parameter sets using vmap
batch_cost = jax.vmap(cost)
batch_params = jnp.ones((100, 3)) * 0.1
batch_results = jax.jit(batch_cost)(batch_params)
print("Batch expectations shape:", batch_results.shape)   # (100,)
```

---

## 🔬 Research Examples

### Example 1 — GPU VRAM Scaling Benchmark
Tests simulation scaling from **4 → 29 qubits** until 4 GB VRAM is saturated.
Produces detailed tables and 6-panel plots (timing, speedup, VRAM, throughput, exponential fit).

```bash
python3 examples/03_benchmarks.py
```

Sample output:
```
╔══════════════════════════════════════════════════════════════════════╗
║     JAX Quantum Simulator — GPU VRAM Scaling Benchmark Suite        ║
╠══════════════════════════════════════════════════════════════════════╣
║  Backend  : gpu                                                     ║
║  Device   : CudaDevice(id=0)                                        ║
║  VRAM     : 4096 MiB  (4.00 GB)                                     ║
╚══════════════════════════════════════════════════════════════════════╝

Qubits  │ State Size  │ Gates  │ Uncompiled(s)  │ JIT Compile(s) │ JIT Exec (s) ...
─────────────────────────────────────────────────────────────────────────────────
4       │ 128 B       │ 24     │ 0.12000        │ 0.45000        │ 0.00031  ...
8       │ 2.0 KB      │ 48     │ 0.13000        │ 0.48000        │ 0.00045  ...
16      │ 512 KB      │ 96     │ 0.18000        │ 0.52000        │ 0.00089  ...
24      │ 128 MB      │ 144    │ 0.38000        │ 0.71000        │ 0.00210  ...
28      │ 2.0 GB      │ 168    │ 1.24000        │ 1.58000        │ 0.01120  ...
⚠  VRAM at 3840 / 4096 MiB (93.7%) — saturation reached. Stopping.
```

---

### Example 2 — VQE: H₂ Molecule Ground State Energy
Finds the electronic ground state energy of H₂ to **chemical accuracy (<1.6 mHartree)**.
Uses a hardware-efficient ansatz over the Jordan-Wigner mapped 4-qubit Hamiltonian.

```bash
python3 examples/04_vqe_h2_molecule.py
```

```
╔══════════════════════════════════════════════════════════════════════╗
║  Variational Quantum Eigensolver (VQE) — H₂ Ground State Energy    ║
╠══════════════════════════════════════════════════════════════════════╣
║  Molecule    : H₂  (equilibrium bond length R = 0.735 Å)           ║
║  FCI target  : -1.137200 Hartree                                    ║
╚══════════════════════════════════════════════════════════════════════╝

 Epoch     Energy (Ha)      ΔE             |∇E|          Time (s)
 ──────    ─────────────    ────────────   ────────────   ──────────
      1    -0.43218751     nan            0.482913       0.31
     20    -0.89420412     -1.47e-02      0.215043       2.84
     ...
    400    -1.13658923     -2.11e-07      0.000041       51.22

 ╔══════════════════════════════════════════════════════╗
 ║  VQE energy          : -1.13658923 Ha               ║
 ║  FCI reference       : -1.13720000 Ha               ║
 ║  Error               : 0.6108 mHartree              ║
 ║  Chemical accuracy   : ✓ YES (<1.6 mHa)             ║
 ╚══════════════════════════════════════════════════════╝
```

---

### Example 3 — QAOA: MaxCut on Weighted Graphs
Solves MaxCut on a 6-node weighted graph with QAOA at depths p=1..5.
Compares approximation ratios against classical exhaustive search.

```bash
python3 examples/05_qaoa_maxcut.py
```

---

### Example 4 — Barren Plateau Research
Empirically verifies the exponential vanishing of gradients predicted by
McClean et al. (2018). Fits exponential decay curves and produces publication-quality
2D loss landscape visualizations.

```bash
python3 examples/06_barren_plateaus.py
```

---

## 📊 GPU Benchmark Results

Memory footprint scaling:

| Qubits | State Size | VRAM Usage |
|--------|-----------|------------|
| 10 | 8 KB | ~50 MiB |
| 16 | 512 KB | ~150 MiB |
| 20 | 8 MB | ~400 MiB |
| 24 | 128 MB | ~900 MiB |
| 26 | 512 MB | ~1.8 GB |
| 27 | 1 GB | ~3.0 GB |
| 28 | 2 GB | ~3.8 GB |
| 29 | 4 GB | **VRAM limit** |

JIT speedup: **up to 400× faster** than uncompiled execution for large circuits.

---

## 📂 Project Structure

```
qauntum machine learning/
├── jax_qsim/
│   ├── __init__.py          — Public API
│   ├── core.py              — State vector + tensordot gate engine
│   ├── ops.py               — Gates: H, X, Y, Z, S, T, CNOT, CZ, SWAP, RX/RY/RZ, ...
│   ├── observables.py       — PauliString, Hamiltonian, expectation, sample()
│   ├── noise.py             — Kraus channels: depolarizing, amplitude/phase damping
│   └── circuit.py           — Circuit builder + JAX compiler
├── examples/
│   ├── 01_state_preparation.py   — Learn GHZ state with Adam
│   ├── 02_vqc_classification.py  — VQC classifier on XOR (jax.vmap)
│   ├── 03_benchmarks.py          — GPU VRAM scaling + benchmarks
│   ├── 04_vqe_h2_molecule.py     — VQE: H₂ ground state energy
│   ├── 05_qaoa_maxcut.py         — QAOA: MaxCut problem
│   └── 06_barren_plateaus.py     — Barren plateau research
├── tests/
│   ├── test_gates.py             — Gate unitarity + application tests
│   ├── test_differentiation.py   — Gradient vs. finite difference tests
│   └── test_circuits.py          — Integration: GHZ, Hamiltonian, noise
├── results/                      — CSV/JSON output from experiments
├── examples/plots/               — All generated plots (PNG, 180 DPI)
├── requirements.txt
├── run_gpu.sh                    — WSL2 GPU launcher script
└── README.md
```

---

## 📖 API Reference

### `Circuit`
```python
c = Circuit(num_qubits=4)
c.h(0).cnot(0, 1).ry(0, param_index=0).rz(0, param_index=1)

state = c.run(params)           # Execute → state tensor (2, ..., 2)
state = c.compile()(params)     # JIT-compiled version
```

### `PauliString`
```python
obs = PauliString({0: 'X', 1: 'Y', 2: 'Z'})   # X₀ Y₁ Z₂
val = expectation(state, obs)                    # ⟨ψ|O|ψ⟩
```

### `Hamiltonian`
```python
H = Hamiltonian(coeffs=[1.0, -0.5], paulis=[PauliString({0:'Z'}), PauliString({1:'Z'})])
energy = H.expectation(state)
```

### Noise Channels
```python
from jax_qsim.noise import depolarizing_channel, apply_channel
kraus = depolarizing_channel(p=0.1)
new_state, new_key = apply_channel(state, kraus, targets=[0], key=key)
```

---

## 📚 References

1. **McClean et al.** (2018). *Barren plateaus in quantum neural network training landscapes*. Nature Communications 9, 4812. https://doi.org/10.1038/s41467-018-07090-4

2. **Peruzzo et al.** (2014). *A variational eigenvalue solver on a photonic quantum processor*. Nature Communications 5, 4213. https://doi.org/10.1038/ncomms5213

3. **Farhi, Goldstone & Gutmann** (2014). *A Quantum Approximate Optimization Algorithm*. arXiv:1411.4028.

4. **Seeley, Richard & Love** (2012). *The Bravyi-Kitaev transformation for quantum computation of electronic structure*. J. Chem. Phys. 137, 224109.

5. **Bradbury et al.** (2018). *JAX: composable transformations of Python+NumPy programs*. http://github.com/google/jax

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with ❤️ using <b>JAX</b>, <b>NumPy</b>, and <b>Matplotlib</b>
</div>

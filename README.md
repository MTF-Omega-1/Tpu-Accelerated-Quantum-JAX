# TPU-Accelerated Quantum JAX

> **Simulating 36-qubit quantum circuits at 549 GB scale. ~0.01ms per gate. 100% pure JAX.**
> Accelerated on NVIDIA GPUs and Google Cloud TPU v6e-64 / v5e clusters. Supported by the Google TPU Research Cloud (TRC) program.

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76b900?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![TPU](https://img.shields.io/badge/TPU-v6e--64_|_v5e--16-7b2fbe?style=for-the-badge&logo=google-cloud)](https://cloud.google.com/tpu)
[![Platform](https://img.shields.io/badge/Platform-GPU_|_TPU_|_CPU-5c6bc0?style=for-the-badge)](https://github.com/AshiteshSingh/Tpu-Accelerated-Quantum-JAX)
[![TRC Supported](https://img.shields.io/badge/Supported_by-TPU_Research_Cloud-4285F4?style=for-the-badge&logo=google-cloud)](https://sites.research.google/trc/)
[![Open In Colab](https://img.shields.io/badge/Open_In-Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/AshiteshSingh/Tpu-Accelerated-Quantum-JAX/blob/main/tpu/colab_tpu_v5e_1chip.ipynb)

<br/>
<img src="gpu/plots/quantum_header_animation.gif" width="280" height="280" alt="Bloch Sphere Dynamics">
<br/>

👉 **For the full research write-up, benchmarks, and detailed technical deep-dives, visit [ashitesh.me](https://ashitesh.me).**

</div>

---

A high-performance, research-grade quantum state-vector simulator built purely in JAX. Run differentiable, noise-resilient, and large-scale quantum circuits accelerated on local NVIDIA GPUs and multi-worker Google Cloud TPU clusters.

---

## ⚡ Key Features

- **100% Pure JAX:** Zero dependencies on heavy frameworks (Qiskit, Cirq, Pennylane). Compiled natively into a single monolithic XLA kernel for bare-metal execution speeds.
- **Multi-Device Sharding:** Scale up to 36 qubits (549 GB state-vector footprint) using distributed JAX `PositionalSharding` across a 64-chip Cloud TPU v6e mesh.
- **Reverse-Mode Auto-Differentiation:** Compute exact gradients in a single backward pass via `jax.grad` for fast training of variational algorithms (VQE, QAOA, QNNs).
- **Hardware-Level Optimizations:** Structured loop primitives (`jax.lax.fori_loop`) prevent XLA graph bloat, while `jax.checkpoint` (gradient rematerialization) keeps memory complexity at $\mathcal{O}(1)$.
- **Stochastic Noise Support:** Built-in Monte Carlo trajectory simulations for open systems and depolarizing NISQ gate noise.

---

## 🏗 Directory Layout

```
.
├── gpu/                     # GPU Modular Simulator & Research scripts
│   ├── jax_qsim/            # Core contraction engine (tensordot + transpose)
│   └── quantum_research/    # VQE, QAOA, GHZ state prep, noise trajectories
├── tpu/                     # TPU Scaling Suite (experiments and runners)
├── shors/                   # TPU-sharded Shor's Algorithm (33 qubits)
├── grover_simulation/       # Grover's Search (up to 36 qubits on 64 TPU chips)
├── tests/                   # Pytest verification suite
└── requirements.txt         # Core dependencies
```

---

## 🛠 Quick Start

### 1. GPU (Local WSL2 / Linux)
Ensure you have CUDA 12 installed, then set up the environment:
```bash
python3 -m venv venv && source venv/bin/activate
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install matplotlib pytest numpy
```
Verify the JAX device setup:
```bash
python3 -c "import jax; print('Backend:', jax.default_backend()); print('Devices:', jax.devices())"
```
Run the local GPU benchmarks:
```bash
python benchmarks/benchmark_27q.py
```

### 2. TPU (Google Cloud TPU v5e / v6e)
In your TPU VM cluster SSH session:
```bash
python3 -m venv tpu_env && source tpu_env/bin/activate
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install matplotlib numpy
```
Run the scaling suite:
```bash
python tpu/tpu_quantum_scale.py
```

---

## 📊 Performance Summary

| Environment | Hardware | Max Qubits | State-Vector Footprint | Gate Speed (10-q) |
|:---|:---|:---:|:---:|:---:|
| **Local GPU** | NVIDIA RTX 2050 (4 GB VRAM) | 29 | ~4.29 GB | ~0.01 ms |
| **TPU Mesh (v5e-16)** | 16x TPU v5e (256 GB aggregate HBM2e) | 33 | 64.00 GB | ~0.01 ms |
| **TPU Mesh (v6e-64)** | 64x TPU v6e (2.0 TB aggregate HBM3) | 36 | 549.76 GB | ~0.01 ms |

---

## 🙏 Acknowledgements

We are extremely grateful to the **TPU Research Cloud (TRC)** program by Google for providing access to Cloud TPU v6e and v5e VM clusters that enabled this scale of research.

---

## 📄 License

Licensed under the [Apache License 2.0](LICENSE).

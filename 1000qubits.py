#!/usr/bin/env python3
"""
================================================================================
  Advanced QML Research: Differentiable MPS on TPU v5e-16
  Target: 1000-Qubit VQE Implementation
================================================================================
"""

import os
import time
import warnings
import numpy as np  # Essential for standard array operations and plotting
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Suppress JAX warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="jax")

# Memory Optimization
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

# Initialize TPU Cluster
jax.distributed.initialize()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
TOTAL_QUBITS = 1000
NUM_GLOBAL_DEVICES = jax.device_count()
QUBITS_PER_CHIP = TOTAL_QUBITS // NUM_GLOBAL_DEVICES 

CHI = 64                    # Reduced to 64 to manage memory for 1000 qubits
EPOCHS = 40                 
LEARNING_RATE = 0.05
EPS = 1e-7

# ─────────────────────────────────────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────────────────────────────────────
Z_MAT = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)

def get_parametric_su4_gate(theta):
    h_zz = jnp.diag(jnp.array([1, -1, -1, 1], dtype=jnp.complex64))
    gate = jnp.cos(theta) * jnp.eye(4, dtype=jnp.complex64) - 1j * jnp.sin(theta) * h_zz
    return gate.reshape((2, 2, 2, 2))

@jax.pmap
def initialize_local_mps(global_dev_idx):
    key = jax.random.PRNGKey(global_dev_idx)
    noise = jax.random.normal(key, (QUBITS_PER_CHIP, CHI, 2, CHI)) * 1e-2
    tensors = noise.astype(jnp.complex64)
    tensors = tensors.at[:, 0, 0, 0].set(1.0 + 0.0j)
    norm = jnp.linalg.norm(tensors)
    return tensors / (norm + EPS)

def apply_local_layer(local_tensors, gate_u, layer_type="even"):
    start_idx = 0 if layer_type == "even" else 1
    entropies_list = []
    limit = QUBITS_PER_CHIP - 1
    
    for idx in range(start_idx, limit, 2):
        site1 = local_tensors[idx]
        site2 = local_tensors[idx + 1]
        
        fused = jnp.einsum("ijk,klm->ijlm", site1, site2)
        transformed = jnp.einsum("abcd,ibcj->iadj", gate_u, fused)
        
        mat = transformed.reshape((CHI * 2, 2 * CHI))
        u, s, vh = jnp.linalg.svd(mat, full_matrices=False)
        
        # Entropy & SVD stability
        s_sq = jnp.square(s)
        s_sq_norm = s_sq / (jnp.sum(s_sq) + EPS)
        entropy = -jnp.sum(s_sq_norm * jnp.log2(s_sq_norm + EPS))
        
        s_trunc = s[:CHI]
        s_trunc = s_trunc / (jnp.linalg.norm(s_trunc) + EPS)
        
        new_site1 = u[:, :CHI].reshape((CHI, 2, CHI))
        new_site2 = (jnp.diag(s_trunc) @ vh[:CHI, :]).reshape((CHI, 2, CHI))
        
        new_site1 = new_site1 / (jnp.linalg.norm(new_site1) + EPS)
        new_site2 = new_site2 / (jnp.linalg.norm(new_site2) + EPS)
        
        local_tensors = local_tensors.at[idx].set(new_site1)
        local_tensors = local_tensors.at[idx + 1].set(new_site2)
        entropies_list.append(entropy)

    return local_tensors, jnp.mean(jnp.stack(entropies_list))

def _vqe_grad_engine_impl(theta, local_mps):
    def evaluate_local_energy(t):
        gate_u = get_parametric_su4_gate(t)
        mps_even, ent_even = apply_local_layer(local_mps, gate_u, "even")
        mps_odd, ent_odd  = apply_local_layer(mps_even, gate_u, "odd")
        
        total_z = 0.0
        for idx in range(QUBITS_PER_CHIP):
            tensor = mps_odd[idx]
            rho_local = jnp.einsum("ijk,ilk->jl", tensor, jnp.conj(tensor))
            z_exp = jnp.real(jnp.trace(rho_local @ Z_MAT))
            total_z = total_z + z_exp
        return total_z, (mps_odd, (ent_even + ent_odd) / 2.0)

    (local_z, (new_local_mps, local_ent)), local_grad = jax.value_and_grad(evaluate_local_energy, argnums=0, has_aux=True)(theta)
    
    # Gradient Clipping (Real-part only to avoid Complex comparison error)
    local_grad = jnp.clip(jnp.real(local_grad), -1.0, 1.0).astype(jnp.complex64)
    
    global_z = jax.lax.psum(local_z, axis_name='dev') / TOTAL_QUBITS
    global_grad = jax.lax.psum(local_grad, axis_name='dev') / TOTAL_QUBITS
    global_ent = jax.lax.pmean(local_ent, axis_name='dev')
    
    return global_z, global_grad, new_local_mps, global_ent

vqe_grad_engine = jax.pmap(_vqe_grad_engine_impl, axis_name='dev', in_axes=(None, 0))

# ─────────────────────────────────────────────────────────────────────────────
# EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    global_device_indices = jnp.arange(jax.local_device_count()) + jax.process_index() * jax.local_device_count()
    mps_state = initialize_local_mps(global_device_indices)
    
    theta = jnp.array(0.85, dtype=jnp.complex64)
    metrics = {"energy": [], "grad_norm": [], "entropy": []}

    for epoch in range(EPOCHS):
        global_z, global_grad, updated_mps, global_ent = vqe_grad_engine(theta, mps_state)
        global_z.block_until_ready()
        
        energy = float(jnp.real(global_z[0]))
        grad_val = float(jnp.real(global_grad[0]))
        
        if np.isnan(energy): break
            
        theta = theta - LEARNING_RATE * grad_val
        mps_state = updated_mps 
        
        metrics["energy"].append(energy)
        metrics["grad_norm"].append(abs(grad_val))
        metrics["entropy"].append(float(jnp.real(global_ent[0])))
        
        if jax.process_index() == 0:
            print(f"Epoch {epoch:<3} | E: {energy:<9.5f} | Grad: {abs(grad_val):<8.5f}")

    if jax.process_index() == 0:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics["energy"], label="Energy")
        plt.title(f"Convergence: {TOTAL_QUBITS} Qubits")
        plt.savefig("vqe_1000q_convergence.png")
        print("✅ Graph saved: vqe_1000q_convergence.png")

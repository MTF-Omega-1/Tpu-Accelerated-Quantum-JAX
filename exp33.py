import os
import time
import functools
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------------
# 1. TPU INITIALIZATION & SHARDING CONFIGURATION
# -------------------------------------------------------------------------
print("Initializing TPU Environment...")
devices = jax.devices()
num_devices = len(devices)
assert num_devices == 4, f"Expected 4 TPU chips for v6e-4, found {num_devices}."

mesh = Mesh(mesh_utils.create_device_mesh((4,)), axis_names=('chips',))
state_sharding = NamedSharding(mesh, P('chips'))

NUM_QUBITS = 33
STATE_SIZE = 1 << NUM_QUBITS  
LOCAL_SIZE = STATE_SIZE // 4

print(f"Allocating 33-qubit state vector ({STATE_SIZE * 8 / 1e9:.2f} GB) via Host Streaming...")

# -------------------------------------------------------------------------
# 2. ZERO-OVERHEAD STREAMING FROM HOST TO CHIPS
# -------------------------------------------------------------------------
t_init_start = time.perf_counter()
host_buffer = np.zeros((LOCAL_SIZE,), dtype=np.complex64)
dev_arrays = []

print("Streaming Shard 0 to TPU Chip 0...")
host_buffer[0] = 1.0 + 0.0j
dev_arrays.append(jax.device_put(host_buffer, devices[0]))

host_buffer[0] = 0.0 + 0.0j
for i in range(1, 4):
    print(f"Streaming Shard {i} to TPU Chip {i}...")
    dev_arrays.append(jax.device_put(host_buffer, devices[i]))

del host_buffer

state = jax.make_array_from_single_device_arrays(
    shape=(STATE_SIZE,),
    sharding=state_sharding,
    arrays=dev_arrays
)
jax.block_until_ready(state)
print(f"State vector successfully loaded into TPU HBM pools. Time taken: {time.perf_counter() - t_init_start:.2f}s")

# -------------------------------------------------------------------------
# 3. RANK-3 & RANK-5 OPTIMIZED GATE OPERATIONS (FAST COMPILATION)
# -------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnums=2, donate_argnums=0)
def apply_1q_gate(state_vec, gate_matrix, target):
    # Flatten into 3 logical dimensions: left, target (2), right
    left_dim = 1 << target
    right_dim = 1 << (NUM_QUBITS - target - 1)
    
    tensor = state_vec.reshape((left_dim, 2, right_dim))
    tensor = jnp.einsum('ij,ajb->aib', gate_matrix, tensor)
    return tensor.reshape((-1,))

@functools.partial(jax.jit, static_argnums=(1, 2), donate_argnums=0)
def apply_cnot(state_vec, control, target):
    X_gate = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex64)
    
    if control < target:
        dim1 = 1 << control
        dim3 = 1 << (target - control - 1)
        dim5 = 1 << (NUM_QUBITS - target - 1)
        
        # Max Rank-5 tensor reshape
        tensor = state_vec.reshape((dim1, 2, dim3, 2, dim5))
        state_c0 = tensor[:, 0, :, :, :]
        state_c1 = tensor[:, 1, :, :, :]
        
        # Apply X to target qubit axis
        state_c1_flipped = jnp.einsum('ij,axjb->axib', X_gate, state_c1)
        
        # Reconstruct by stacking along the control axis (axis 1)
        combined = jnp.stack([state_c0, state_c1_flipped], axis=1)
        return combined.reshape((-1,))
    else:
        dim1 = 1 << target
        dim3 = 1 << (control - target - 1)
        dim5 = 1 << (NUM_QUBITS - control - 1)
        
        tensor = state_vec.reshape((dim1, 2, dim3, 2, dim5))
        state_c0 = tensor[:, :, :, 0, :]
        state_c1 = tensor[:, :, :, 1, :]
        
        # Apply X to target qubit axis
        state_c1_flipped = jnp.einsum('ij,ajxb->aixb', X_gate, state_c1)
        
        # Reconstruct by stacking along the control axis (axis 3)
        combined = jnp.stack([state_c0, state_c1_flipped], axis=3)
        return combined.reshape((-1,))

# -------------------------------------------------------------------------
# 4. BENCHMARKING RUN & PERFORMANCE MONITORING
# -------------------------------------------------------------------------
print("\nStarting Benchmark Circuit...")
Hadamard = jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=jnp.complex64) / jnp.sqrt(2.0)

qubit_latencies = []
gate_depth_times = []

print("Compiling XLA graph (Warm-up run)...")
tmp = apply_1q_gate(state, Hadamard, 2)
tmp = apply_cnot(tmp, 2, 5)
jax.block_until_ready(tmp)
del tmp
print("Compilation complete. Executing benchmark...")

start_total = time.perf_counter()

# Run 1-Qubit gate benchmark
for q in range(NUM_QUBITS):
    t0 = time.perf_counter()
    state = apply_1q_gate(state, Hadamard, q)
    jax.block_until_ready(state)
    t1 = time.perf_counter()
    
    latency = (t1 - t0) * 1000  
    qubit_latencies.append(latency)
    gate_depth_times.append(time.perf_counter() - start_total)
    print(f"Gate Depth {q+1:02d}: 1Q-Gate on Qubit {q:02d} | Execution Time: {latency:.2f} ms")

# Run entangling CNOT layers
cnot_pairs = [(i, (i + 7) % NUM_QUBITS) for i in range(10)]
for idx, (ctrl, tgt) in enumerate(cnot_pairs):
    t0 = time.perf_counter()
    state = apply_cnot(state, ctrl, tgt)
    jax.block_until_ready(state)
    t1 = time.perf_counter()
    
    latency = (t1 - t0) * 1000
    gate_depth_times.append(time.perf_counter() - start_total)
    print(f"Gate Depth {NUM_QUBITS + idx + 1:02d}: CNOT ctrl={ctrl:02d} tgt={tgt:02d} | Execution Time: {latency:.2f} ms")

end_total = time.perf_counter()
print(f"\nSimulation complete. Total circuit execution time: {end_total - start_total:.4f} seconds.")

# -------------------------------------------------------------------------
# 5. GRAPH GENERATION METRICS
# -------------------------------------------------------------------------
print("\nGenerating performance diagnostic plots...")
os.makedirs("metrics", exist_ok=True)

plt.figure(figsize=(10, 5))
plt.bar(range(NUM_QUBITS), qubit_latencies, color='royalblue', edgecolor='black', alpha=0.85)
plt.axvline(x=1.5, color='crimson', linestyle='--', linewidth=2, label='Distributed Interconnect Boundary')
plt.title("TPU v6e-4 Latency Profile by Qubit Index", fontsize=14, fontweight='bold')
plt.xlabel("Target Qubit Index", fontsize=12)
plt.ylabel("Execution Latency (ms)", fontsize=12)
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("metrics/qubit_latency_profile.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(gate_depth_times) + 1), gate_depth_times, marker='o', color='forestgreen', linewidth=2)
plt.title("Total Simulation Run Time vs. Gate Depth", fontsize=14, fontweight='bold')
plt.xlabel("Gate Execution Step Depth", fontsize=12)
plt.ylabel("Cumulative Elapsed Time (s)", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("metrics/runtime_scaling.png", dpi=300)
plt.close()

print("Performance graphs successfully generated and saved to the 'metrics/' folder.")

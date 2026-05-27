import os
import time
import functools
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------------
# 1. 2D TOPOLOGY MESH & SHARDING CONFIGURATION
# -------------------------------------------------------------------------
print("Initializing TPU Environment...")
devices = jax.devices()
num_devices = len(devices)
assert num_devices == 4, f"Expected 4 TPU chips for v6e-4, found {num_devices}."

# Establish a 2x2 Mesh matching the 4 physical TPU chips perfectly
mesh = Mesh(mesh_utils.create_device_mesh((2, 2)), axis_names=('q31', 'q30'))
state_sharding = NamedSharding(mesh, P('q31', 'q30', None))

NUM_QUBITS = 32
STATE_SIZE = 1 << NUM_QUBITS  

print(f"Allocating static 32-qubit state vector ({STATE_SIZE * 8 / 1e9:.2f} GB) across 2D mesh layout...")

# -------------------------------------------------------------------------
# 2. IN-PLACE SHARDED INITIALIZATION
# -------------------------------------------------------------------------
@jax.jit
def init_ground_state():
    state_vec = jnp.zeros((2, 2, 1 << 30), dtype=jnp.complex64)
    return state_vec.at[0, 0, 0].set(1.0 + 0.0j)

init_ground_state_sharded = jax.jit(init_ground_state, out_shardings=state_sharding)
state = init_ground_state_sharded()
jax.block_until_ready(state)
print("State vector successfully sharded across 2D TPU HBM pools.")

# -------------------------------------------------------------------------
# 3. CORRECTED TOPOLOGY-AWARE IN-PLACE GATE OPERATORS
# -------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnums=2, donate_argnums=0)
def apply_1q_gate(state_vec, gate_matrix, target):
    if target < 30:
        @functools.partial(shard_map, mesh=mesh, in_specs=P('q31', 'q30', None), out_specs=P('q31', 'q30', None))
        def local_1q(local_state):
            left = 1 << target
            right = 1 << (30 - target - 1)
            tensor = local_state.reshape((left, 2, right))
            tensor = jnp.einsum('ij,ajb->aib', gate_matrix, tensor)
            # FIX: Reshape back to rank-3 (1, 1, 1<<30) to match out_specs length
            return tensor.reshape((1, 1, 1 << 30))
        return local_1q(state_vec)
        
    elif target == 30:
        return jnp.einsum('ij,kjl->kil', gate_matrix, state_vec)
        
    elif target == 31:
        return jnp.einsum('ij,jkl->ikl', gate_matrix, state_vec)

@functools.partial(jax.jit, static_argnums=(1, 2), donate_argnums=0)
def apply_cnot(state_vec, control, target):
    cnot_tensor = jnp.array([
        [[[1, 0], [0, 0]], [[0, 1], [0, 0]]],
        [[[0, 0], [0, 1]], [[0, 0], [1, 0]]]
    ], dtype=jnp.complex64)
    
    # CASE 1: Both Qubits are Local (< 30)
    if control < 30 and target < 30:
        @functools.partial(shard_map, mesh=mesh, in_specs=P('q31', 'q30', None), out_specs=P('q31', 'q30', None))
        def local_cnot(local_state):
            low, high = min(control, target), max(control, target)
            dim1 = 1 << low
            dim3 = 1 << (high - low - 1)
            dim5 = 1 << (30 - high - 1)
            tensor = local_state.reshape((dim1, 2, dim3, 2, dim5))
            if control < target:
                res = jnp.einsum('CTct,acbtd->aCbTd', cnot_tensor, tensor)
            else:
                res = jnp.einsum('CTct,atbcd->aTbCd', cnot_tensor, tensor)
            # FIX: Reshape back to rank-3 (1, 1, 1<<30) to match out_specs length
            return res.reshape((1, 1, 1 << 30))
        return local_cnot(state_vec)
        
    # CASE 2: Global Control, Local Target
    elif control == 31 and target < 30:
        left = 1 << target
        right = 1 << (30 - target - 1)
        tensor = state_vec.reshape((2, 2, left, 2, right))
        res = jnp.einsum('CTct,cqatb->CqaTb', cnot_tensor, tensor)
        return res.reshape((2, 2, 1 << 30))
        
    elif control == 30 and target < 30:
        left = 1 << target
        right = 1 << (30 - target - 1)
        tensor = state_vec.reshape((2, 2, left, 2, right))
        res = jnp.einsum('CTct,qcatb->qCaTb', cnot_tensor, tensor)
        return res.reshape((2, 2, 1 << 30))
        
    # CASE 3: Local Control, Global Target
    elif control < 30 and target == 31:
        left = 1 << control
        right = 1 << (30 - control - 1)
        tensor = state_vec.reshape((2, 2, left, 2, right))
        res = jnp.einsum('CTct,tqacb->TqaCb', cnot_tensor, tensor)
        return res.reshape((2, 2, 1 << 30))
        
    elif control < 30 and target == 30:
        left = 1 << control
        right = 1 << (30 - control - 1)
        tensor = state_vec.reshape((2, 2, left, 2, right))
        res = jnp.einsum('CTct,qtacb->qTaCb', cnot_tensor, tensor)
        return res.reshape((2, 2, 1 << 30))
        
    # CASE 4: Both Qubits are Global (30 and 31)
    elif control == 31 and target == 30:
        return jnp.einsum('CTct,ctl->CTl', cnot_tensor, state_vec)
        
    elif control == 30 and target == 31:
        return jnp.einsum('CTct,tcl->TCl', cnot_tensor, state_vec)

# -------------------------------------------------------------------------
# 4. BENCHMARKING RUN & PERFORMANCE MONITORING
# -------------------------------------------------------------------------
print("\nStarting Benchmark Circuit...")
Hadamard = jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=jnp.complex64) / jnp.sqrt(2.0)

qubit_latencies = []
gate_depth_times = []

print("Compiling XLA graph (Warm-up run)...")
state = apply_1q_gate(state, Hadamard, 2)
state = apply_cnot(state, 2, 5)
state = apply_1q_gate(state, Hadamard, 30)
jax.block_until_ready(state)
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
plt.title("TPU v6e-4 Latency Profile by Qubit Index (32 Qubits)", fontsize=14, fontweight='bold')
plt.xlabel("Target Qubit Index", fontsize=12)
plt.ylabel("Execution Latency (ms)", fontsize=12)
plt.grid(axis='y', linestyle=':', alpha=0.6)
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

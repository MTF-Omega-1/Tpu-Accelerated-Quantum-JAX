import os
import time
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
import matplotlib.pyplot as plt
import numpy as np
print('Initializing TPU Environment...')
devices = jax.devices()
num_devices = len(devices)
assert num_devices == 4, f'Expected 4 TPU chips for v6e-4, found {num_devices}.'
mesh = Mesh(mesh_utils.create_device_mesh((4,)), axis_names=('chips',))
state_sharding = NamedSharding(mesh, P('chips', None))
NUM_QUBITS = 32
LOCAL_QUBITS = 30
STATE_SIZE = 1 << NUM_QUBITS
LOCAL_SIZE = 1 << LOCAL_QUBITS
total_gib = STATE_SIZE * np.dtype(np.complex64).itemsize / 1024 ** 3
per_chip_gib = total_gib / 4.0
print(f'Allocating exact 32-qubit state vector: {total_gib:.2f} GiB total ({per_chip_gib:.2f} GiB per chip across 4 TPU chips).')

def init_ground_state():
    state_vec = jnp.zeros((4, LOCAL_SIZE), dtype=jnp.complex64)
    return state_vec.at[0, 0].set(jnp.array(1.0 + 0j, dtype=jnp.complex64))
init_ground_state_sharded = jax.jit(init_ground_state, out_shardings=state_sharding)
state = init_ground_state_sharded()
jax.block_until_ready(state)
print('State vector successfully sharded across TPU HBM pools.')

def _make_1q_branch(t):
    if t < LOCAL_QUBITS:

        def branch(state_vec, gate_matrix, t=t):
            left = 1 << t
            right = 1 << LOCAL_QUBITS - t - 1
            tensor = state_vec.reshape((4, left, 2, right))
            tensor = jnp.einsum('ij,aljb->alib', gate_matrix, tensor)
            return tensor.reshape((4, LOCAL_SIZE))
        return branch
    if t == 30:

        def branch(state_vec, gate_matrix):
            tensor = state_vec.reshape((2, 2, LOCAL_SIZE))
            tensor = jnp.einsum('ij,akb->aib', gate_matrix, tensor)
            return tensor.reshape((4, LOCAL_SIZE))
        return branch

    def branch(state_vec, gate_matrix):
        tensor = state_vec.reshape((2, 2, LOCAL_SIZE))
        tensor = jnp.einsum('ij,kab->iab', gate_matrix, tensor)
        return tensor.reshape((4, LOCAL_SIZE))
    return branch
ONE_Q_BRANCHES = tuple((_make_1q_branch(t) for t in range(NUM_QUBITS)))

def _apply_1q_gate_impl(state_vec, gate_matrix, target):
    target = jnp.asarray(target, dtype=jnp.int32)
    target = jnp.clip(target, 0, NUM_QUBITS - 1)
    return lax.switch(target, ONE_Q_BRANCHES, state_vec, gate_matrix)
apply_1q_gate = jax.jit(_apply_1q_gate_impl, donate_argnums=0, out_shardings=state_sharding)
CNOT_TENSOR = jnp.array([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [0, 1]], [[0, 0], [1, 0]]]], dtype=jnp.complex64)

def _make_cnot_branch(control, target):

    def branch(state_vec):
        if control == target:
            return state_vec
        if control < LOCAL_QUBITS and target < LOCAL_QUBITS:
            low, high = (min(control, target), max(control, target))
            dim1 = 1 << low
            dim3 = 1 << high - low - 1
            dim5 = 1 << LOCAL_QUBITS - high - 1
            tensor = state_vec.reshape((4, dim1, 2, dim3, 2, dim5))
            if control < target:
                tensor = jnp.einsum('CTct,alcbtd->aCbTd', CNOT_TENSOR, tensor)
            else:
                tensor = jnp.einsum('TCtc,atlbcd->aTbCd', CNOT_TENSOR, tensor)
            return tensor.reshape((4, LOCAL_SIZE))
        if control >= LOCAL_QUBITS and target < LOCAL_QUBITS:
            left = 1 << target
            right = 1 << LOCAL_QUBITS - target - 1
            tensor = state_vec.reshape((2, 2, left, 2, right))
            if control == 30:
                tensor = jnp.einsum('CTct,gcltr->gClTr', CNOT_TENSOR, tensor)
            else:
                tensor = jnp.einsum('CTct,cgltr->CglTr', CNOT_TENSOR, tensor)
            return tensor.reshape((4, LOCAL_SIZE))
        if control < LOCAL_QUBITS and target >= LOCAL_QUBITS:
            left = 1 << control
            right = 1 << LOCAL_QUBITS - control - 1
            tensor = state_vec.reshape((2, 2, left, 2, right))
            if target == 30:
                tensor = jnp.einsum('CTct,gtlcr->gTlCr', CNOT_TENSOR, tensor)
            else:
                tensor = jnp.einsum('CTct,gltcr->TglCr', CNOT_TENSOR, tensor)
            return tensor.reshape((4, LOCAL_SIZE))
        tensor = state_vec.reshape((2, 2, LOCAL_SIZE))
        if control == 31 and target == 30:
            tensor = jnp.einsum('CTct,ctl->CTl', CNOT_TENSOR, tensor)
        else:
            tensor = jnp.einsum('CTct,tcl->TCl', CNOT_TENSOR, tensor)
        return tensor.reshape((4, LOCAL_SIZE))
    return branch
CNOT_BRANCHES = tuple((_make_cnot_branch(control, target) for control in range(NUM_QUBITS) for target in range(NUM_QUBITS)))

def _apply_cnot_impl(state_vec, control, target):
    control = jnp.asarray(control, dtype=jnp.int32)
    target = jnp.asarray(target, dtype=jnp.int32)
    control = jnp.clip(control, 0, NUM_QUBITS - 1)
    target = jnp.clip(target, 0, NUM_QUBITS - 1)
    case = control * NUM_QUBITS + target
    return lax.switch(case, CNOT_BRANCHES, state_vec)
apply_cnot = jax.jit(_apply_cnot_impl, donate_argnums=0, out_shardings=state_sharding)
print('\nStarting Benchmark Circuit...')
Hadamard = jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=jnp.complex64) / jnp.sqrt(jnp.array(2.0, dtype=jnp.complex64))
qubit_latencies = []
gate_depth_times = []
print('Compiling XLA graph (one-time warm-up)...')
state = apply_1q_gate(state, Hadamard, 2)
state = apply_cnot(state, 2, 5)
state = apply_1q_gate(state, Hadamard, 30)
state = apply_cnot(state, 30, 5)
jax.block_until_ready(state)
print('Compilation complete. Executing benchmark...')
start_total = time.perf_counter()
for q in range(NUM_QUBITS):
    t0 = time.perf_counter()
    state = apply_1q_gate(state, Hadamard, q)
    jax.block_until_ready(state)
    t1 = time.perf_counter()
    latency = (t1 - t0) * 1000
    qubit_latencies.append(latency)
    gate_depth_times.append(time.perf_counter() - start_total)
    print(f'Gate Depth {q + 1:02d}: 1Q-Gate on Qubit {q:02d} | Execution Time: {latency:.2f} ms')
cnot_pairs = [(i, (i + 7) % NUM_QUBITS) for i in range(10)]
for idx, (ctrl, tgt) in enumerate(cnot_pairs):
    t0 = time.perf_counter()
    state = apply_cnot(state, ctrl, tgt)
    jax.block_until_ready(state)
    t1 = time.perf_counter()
    latency = (t1 - t0) * 1000
    gate_depth_times.append(time.perf_counter() - start_total)
    print(f'Gate Depth {NUM_QUBITS + idx + 1:02d}: CNOT ctrl={ctrl:02d} tgt={tgt:02d} | Execution Time: {latency:.2f} ms')
end_total = time.perf_counter()
print(f'\nSimulation complete. Total circuit execution time: {end_total - start_total:.4f} seconds.')
print('\nGenerating performance diagnostic plots...')
os.makedirs('metrics', exist_ok=True)
plt.figure(figsize=(10, 5))
plt.bar(range(NUM_QUBITS), qubit_latencies, edgecolor='black', alpha=0.85)
plt.title('TPU v6e-4 Latency Profile by Qubit Index (32 Qubits)', fontsize=14, fontweight='bold')
plt.xlabel('Target Qubit Index', fontsize=12)
plt.ylabel('Execution Latency (ms)', fontsize=12)
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('metrics/qubit_latency_profile.png', dpi=300)
plt.close()
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(gate_depth_times) + 1), gate_depth_times, marker='o', linewidth=2)
plt.title('Total Simulation Run Time vs. Gate Depth', fontsize=14, fontweight='bold')
plt.xlabel('Gate Execution Step Depth', fontsize=12)
plt.ylabel('Cumulative Elapsed Time (s)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('metrics/runtime_scaling.png', dpi=300)
plt.close()
print("Performance graphs successfully generated and saved to the 'metrics/' folder.")
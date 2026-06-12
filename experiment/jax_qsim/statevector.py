import jax
import jax.numpy as jnp
from . import gates
from .observables import PauliString, Hamiltonian

def zero_state(num_qubits):
    state = jnp.zeros(2 ** num_qubits, dtype=jnp.complex64)
    state = state.at[0].set(1.0)
    return state.reshape((2,) * num_qubits)

def apply_gate(state, gate, target_qubits):
    n = state.ndim
    k = len(target_qubits)
    state_flat = state.reshape(-1)
    if k == 1:
        q = target_qubits[0]
        gate_2x2 = gate.reshape(2, 2)
        shape_3d = (1 << q, 2, 1 << n - q - 1)
        state_3d = state_flat.reshape(shape_3d)
        state_transposed = jnp.transpose(state_3d, (1, 0, 2))
        state_2d = state_transposed.reshape(2, -1)
        out_2d = gate_2x2 @ state_2d
        out_transposed = out_2d.reshape(2, 1 << q, 1 << n - q - 1)
        out_3d = jnp.transpose(out_transposed, (1, 0, 2))
        res_flat = out_3d.reshape(-1)
    elif k == 2:
        q1, q2 = target_qubits
        gate_4x4 = gate.reshape(4, 4)
        if q1 > q2:
            gate_tensor = gate_4x4.reshape(2, 2, 2, 2)
            gate_tensor = jnp.transpose(gate_tensor, (1, 0, 3, 2))
            gate_4x4 = gate_tensor.reshape(4, 4)
            left_q, right_q = (q2, q1)
        else:
            left_q, right_q = (q1, q2)
        shape_5d = (1 << left_q, 2, 1 << right_q - left_q - 1, 2, 1 << n - right_q - 1)
        state_5d = state_flat.reshape(shape_5d)
        state_transposed = jnp.transpose(state_5d, (1, 3, 0, 2, 4))
        state_2d = state_transposed.reshape(4, -1)
        out_2d = gate_4x4 @ state_2d
        out_transposed = out_2d.reshape(2, 2, 1 << left_q, 1 << right_q - left_q - 1, 1 << n - right_q - 1)
        out_5d = jnp.transpose(out_transposed, (2, 0, 3, 1, 4))
        res_flat = out_5d.reshape(-1)
    else:
        if gate.ndim != 2 * k:
            gate = gate.reshape((2,) * (2 * k))
        gate_axes = tuple(range(k, 2 * k))
        state_axes = tuple(target_qubits)
        out = jnp.tensordot(gate, state, axes=(gate_axes, state_axes))
        uncontracted = [i for i in range(n) if i not in target_qubits]
        dest = [None] * n
        for i, q in enumerate(target_qubits):
            dest[q] = i
        curr = k
        for q in uncontracted:
            dest[q] = curr
            curr += 1
        return jnp.transpose(out, dest)
    return res_flat.reshape((2,) * n)

def expectation_pauli_string(state, pauli_string):
    n = state.ndim
    phi = state
    for q, op in pauli_string.paulis.items():
        if op == 'X':
            phi = apply_gate(phi, gates.X(), [q])
        elif op == 'Y':
            phi = apply_gate(phi, gates.Y(), [q])
        elif op == 'Z':
            phi = apply_gate(phi, gates.Z(), [q])
    return jnp.real(jnp.vdot(state, phi))

def expectation_hamiltonian(state, hamiltonian):
    exp_val = 0.0
    for coeff, pauli_string in zip(hamiltonian.coeffs, hamiltonian.pauli_strings):
        exp_val += coeff * expectation_pauli_string(state, pauli_string)
    return exp_val

def sample(state, num_shots, key):
    n = state.ndim
    probs = jnp.abs(state.reshape(-1)) ** 2
    sampled_indices = jax.random.choice(key, 2 ** n, shape=(num_shots,), p=probs)
    powers = 2 ** jnp.arange(n - 1, -1, -1)
    bitstrings = sampled_indices[:, None] // powers[None, :] % 2
    return bitstrings

def measure(state, qubit, key):
    n = state.ndim
    probs = jnp.abs(state) ** 2
    axes = tuple((i for i in range(n) if i != qubit))
    marginal_prob = jnp.sum(probs, axis=axes)
    prob_0 = marginal_prob[0]
    r = jax.random.uniform(key)
    measured_bit = jnp.where(r < prob_0, 0, 1)
    collapsed = jnp.where(measured_bit == 0, state.at[tuple((slice(None) if i != qubit else 0 for i in range(n)))].get(), state.at[tuple((slice(None) if i != qubit else 1 for i in range(n)))].get())
    new_state = jnp.zeros_like(state)
    new_state = jnp.where(measured_bit == 0, new_state.at[tuple((slice(None) if i != qubit else 0 for i in range(n)))].set(collapsed), new_state.at[tuple((slice(None) if i != qubit else 1 for i in range(n)))].set(collapsed))
    norm = jnp.sqrt(jnp.sum(jnp.abs(new_state) ** 2))
    return (measured_bit, new_state / norm)

class Statevector:

    def __init__(self, num_qubits, data=None):
        self.num_qubits = num_qubits
        self.data = data if data is not None else zero_state(num_qubits)

    def apply_gate(self, gate, target_qubits):
        self.data = apply_gate(self.data, gate, target_qubits)
        return self

    def expectation(self, observable):
        if isinstance(observable, PauliString):
            return expectation_pauli_string(self.data, observable)
        elif isinstance(observable, Hamiltonian):
            return expectation_hamiltonian(self.data, observable)
        else:
            raise TypeError('Observable must be PauliString or Hamiltonian')

    def sample(self, num_shots, key):
        return sample(self.data, num_shots, key)

    def measure(self, qubit, key):
        bit, new_data = measure(self.data, qubit, key)
        self.data = new_data
        return bit
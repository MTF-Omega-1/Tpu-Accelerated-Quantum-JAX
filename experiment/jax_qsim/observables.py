import jax.numpy as jnp
from . import gates

class PauliString:

    def __init__(self, paulis=None):
        self.paulis = paulis if paulis is not None else {}

    def __repr__(self):
        if not self.paulis:
            return 'Identity'
        terms = [f'{op}_{q}' for q, op in sorted(self.paulis.items())]
        return ' * '.join(terms)

class Hamiltonian:

    def __init__(self, coeffs, pauli_strings):
        self.coeffs = jnp.array(coeffs, dtype=jnp.float32)
        self.pauli_strings = pauli_strings

    def __repr__(self):
        terms = []
        for c, p in zip(self.coeffs, self.pauli_strings):
            terms.append(f'{c:+.4f} * ({p})')
        return ' + '.join(terms)
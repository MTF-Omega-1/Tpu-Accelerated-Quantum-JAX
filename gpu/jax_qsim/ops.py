import jax
import jax.numpy as jnp
X = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex64)
Y = jnp.array([[0.0, -1j], [1j, 0.0]], dtype=jnp.complex64)
Z = jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex64)
H = 1.0 / jnp.sqrt(2.0) * jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=jnp.complex64)
I = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.complex64)
S = jnp.array([[1.0, 0.0], [0.0, 1j]], dtype=jnp.complex64)
T = jnp.array([[1.0, 0.0], [0.0, jnp.exp(1j * jnp.pi / 4.0)]], dtype=jnp.complex64)

def rx(theta: float) -> jnp.ndarray:
    cos = jnp.cos(theta / 2.0)
    sin = jnp.sin(theta / 2.0)
    return jnp.array([[cos, -1j * sin], [-1j * sin, cos]], dtype=jnp.complex64)

def ry(theta: float) -> jnp.ndarray:
    cos = jnp.cos(theta / 2.0)
    sin = jnp.sin(theta / 2.0)
    return jnp.array([[cos, -sin], [sin, cos]], dtype=jnp.complex64)

def rz(theta: float) -> jnp.ndarray:
    exp_neg = jnp.exp(-1j * theta / 2.0)
    exp_pos = jnp.exp(1j * theta / 2.0)
    return jnp.array([[exp_neg, 0.0], [0.0, exp_pos]], dtype=jnp.complex64)

def phase_shift(phi: float) -> jnp.ndarray:
    return jnp.array([[1.0, 0.0], [0.0, jnp.exp(1j * phi)]], dtype=jnp.complex64)

def controlled_gate(U: jnp.ndarray) -> jnp.ndarray:
    zero = jnp.zeros((2, 2), dtype=jnp.complex64)
    row1 = jnp.hstack([I, zero])
    row2 = jnp.hstack([zero, U])
    return jnp.vstack([row1, row2])
CNOT = controlled_gate(X)
CZ = controlled_gate(Z)
SWAP = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=jnp.complex64)

def crx(theta: float) -> jnp.ndarray:
    return controlled_gate(rx(theta))

def cry(theta: float) -> jnp.ndarray:
    return controlled_gate(ry(theta))

def crz(theta: float) -> jnp.ndarray:
    return controlled_gate(rz(theta))

def cphase(phi: float) -> jnp.ndarray:
    return controlled_gate(phase_shift(phi))
import jax.numpy as jnp

def H():
    return jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=jnp.complex64) / jnp.sqrt(2.0)

def X():
    return jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex64)

def Y():
    return jnp.array([[0.0, -1j], [1j, 0.0]], dtype=jnp.complex64)

def Z():
    return jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex64)

def S():
    return jnp.array([[1.0, 0.0], [0.0, 1j]], dtype=jnp.complex64)

def T():
    return jnp.array([[1.0, 0.0], [0.0, jnp.exp(1j * jnp.pi / 4.0)]], dtype=jnp.complex64)

def RX(theta):
    c = jnp.cos(theta / 2.0)
    s = -1j * jnp.sin(theta / 2.0)
    return jnp.array([[c, s], [s, c]], dtype=jnp.complex64)

def RY(theta):
    c = jnp.cos(theta / 2.0)
    s = jnp.sin(theta / 2.0)
    return jnp.array([[c, -s], [s, c]], dtype=jnp.complex64)

def RZ(theta):
    e = jnp.exp(-1j * theta / 2.0)
    return jnp.array([[e, 0.0], [0.0, jnp.conj(e)]], dtype=jnp.complex64)

def PhaseShift(theta):
    return jnp.array([[1.0, 0.0], [0.0, jnp.exp(1j * theta)]], dtype=jnp.complex64)

def CNOT():
    return jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]], dtype=jnp.complex64).reshape(2, 2, 2, 2)

def CZ():
    return jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, -1.0]], dtype=jnp.complex64).reshape(2, 2, 2, 2)

def SWAP():
    return jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=jnp.complex64).reshape(2, 2, 2, 2)

def Toffoli():
    mat = jnp.eye(8, dtype=jnp.complex64)
    mat = mat.at[6, 6].set(0.0)
    mat = mat.at[7, 7].set(0.0)
    mat = mat.at[6, 7].set(1.0)
    mat = mat.at[7, 6].set(1.0)
    return mat.reshape(2, 2, 2, 2, 2, 2)

def CRX(theta):
    rx_mat = RX(theta)
    mat = jnp.eye(4, dtype=jnp.complex64)
    mat = mat.at[2:4, 2:4].set(rx_mat)
    return mat.reshape(2, 2, 2, 2)

def CRY(theta):
    ry_mat = RY(theta)
    mat = jnp.eye(4, dtype=jnp.complex64)
    mat = mat.at[2:4, 2:4].set(ry_mat)
    return mat.reshape(2, 2, 2, 2)

def CRZ(theta):
    rz_mat = RZ(theta)
    mat = jnp.eye(4, dtype=jnp.complex64)
    mat = mat.at[2:4, 2:4].set(rz_mat)
    return mat.reshape(2, 2, 2, 2)

def CP(theta):
    mat = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, jnp.exp(1j * theta)]], dtype=jnp.complex64)
    return mat.reshape(2, 2, 2, 2)
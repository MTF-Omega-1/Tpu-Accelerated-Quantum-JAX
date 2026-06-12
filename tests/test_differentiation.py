import jax
import jax.numpy as jnp
from jax_qsim import Circuit
from jax_qsim.observables import PauliString, expectation

def test_single_parameter_gradient():
    c = Circuit(num_qubits=1)
    c.ry(0, param_index=0)
    obs = PauliString({0: 'Z'})

    def loss(params):
        state = c.run(params)
        return expectation(state, obs)
    grad_fn = jax.grad(loss)
    theta = 0.5
    params = jnp.array([theta])
    jax_grad = grad_fn(params)[0]
    analytical_grad = -jnp.sin(theta)
    eps = 0.0001
    val_plus = loss(jnp.array([theta + eps]))
    val_minus = loss(jnp.array([theta - eps]))
    fd_grad = (val_plus - val_minus) / (2.0 * eps)
    assert jnp.allclose(jax_grad, analytical_grad, atol=0.0001)
    assert jnp.allclose(jax_grad, fd_grad, atol=0.001)

def test_multi_parameter_gradient():
    c = Circuit(num_qubits=2)
    c.rx(0, param_index=0)
    c.ry(1, param_index=1)
    c.cnot(0, 1)
    obs = PauliString({0: 'Z', 1: 'Z'})

    def loss(params):
        state = c.run(params)
        return expectation(state, obs)
    grad_fn = jax.grad(loss)
    params = jnp.array([0.4, -0.7])
    jax_grads = grad_fn(params)
    eps = 0.0001
    fd_grads = []
    for i in range(len(params)):
        params_plus = params.at[i].add(eps)
        params_minus = params.at[i].add(-eps)
        val_plus = loss(params_plus)
        val_minus = loss(params_minus)
        fd_grads.append((val_plus - val_minus) / (2.0 * eps))
    fd_grads = jnp.array(fd_grads)
    assert jnp.allclose(jax_grads, fd_grads, atol=0.001)
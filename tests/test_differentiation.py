import jax
import jax.numpy as jnp
from jax_qsim import Circuit
from jax_qsim.observables import PauliString, expectation

def test_single_parameter_gradient():
    """Verify gradient of a simple 1-qubit RY rotation against analytical and numerical solutions."""
    # State: RY(theta) |0> = cos(theta/2)|0> + sin(theta/2)|1>
    # Expectation of Z: cos^2(theta/2) - sin^2(theta/2) = cos(theta)
    # Derivative w.r.t theta: -sin(theta)
    c = Circuit(num_qubits=1)
    c.ry(0, param_index=0)
    
    obs = PauliString({0: 'Z'})
    
    def loss(params):
        state = c.run(params)
        return expectation(state, obs)
        
    grad_fn = jax.grad(loss)
    
    # Test at theta = 0.5
    theta = 0.5
    params = jnp.array([theta])
    
    # 1. JAX gradient
    jax_grad = grad_fn(params)[0]
    
    # 2. Analytical gradient
    analytical_grad = -jnp.sin(theta)
    
    # 3. Finite-difference numerical gradient
    eps = 1e-4
    val_plus = loss(jnp.array([theta + eps]))
    val_minus = loss(jnp.array([theta - eps]))
    fd_grad = (val_plus - val_minus) / (2.0 * eps)
    
    assert jnp.allclose(jax_grad, analytical_grad, atol=1e-4)
    assert jnp.allclose(jax_grad, fd_grad, atol=1e-3)

def test_multi_parameter_gradient():
    """Verify gradients of a multi-qubit circuit with multiple parameterized gates."""
    # Circuit: RX(theta0) on qubit 0, RY(theta1) on qubit 1, CNOT(0, 1)
    c = Circuit(num_qubits=2)
    c.rx(0, param_index=0)
    c.ry(1, param_index=1)
    c.cnot(0, 1)
    
    obs = PauliString({0: 'Z', 1: 'Z'})
    
    def loss(params):
        state = c.run(params)
        return expectation(state, obs)
        
    grad_fn = jax.grad(loss)
    
    # Test at some random parameters
    params = jnp.array([0.4, -0.7])
    
    # JAX gradient
    jax_grads = grad_fn(params)
    
    # Finite differences
    eps = 1e-4
    fd_grads = []
    for i in range(len(params)):
        params_plus = params.at[i].add(eps)
        params_minus = params.at[i].add(-eps)
        val_plus = loss(params_plus)
        val_minus = loss(params_minus)
        fd_grads.append((val_plus - val_minus) / (2.0 * eps))
        
    fd_grads = jnp.array(fd_grads)
    
    assert jnp.allclose(jax_grads, fd_grads, atol=1e-3)

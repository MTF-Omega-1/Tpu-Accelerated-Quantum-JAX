import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jax_qsim import Circuit, state_vector_flat
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_state_prep():
    print('=' * 60)
    print('   Quantum State Preparation: Learning a 3-Qubit GHZ State')
    print('=' * 60)
    target_state = jnp.zeros(8, dtype=jnp.complex64)
    target_state = target_state.at[0].set(1.0 / jnp.sqrt(2.0))
    target_state = target_state.at[7].set(1.0 / jnp.sqrt(2.0))
    c = Circuit(num_qubits=3)
    c.rx(0, param_index=0)
    c.ry(1, param_index=1)
    c.rz(2, param_index=2)
    c.cnot(0, 1)
    c.cnot(1, 2)
    c.rx(0, param_index=3)
    c.ry(1, param_index=4)
    c.rz(2, param_index=5)
    c.cnot(0, 1)
    c.cnot(1, 2)
    c.rx(0, param_index=6)
    c.ry(1, param_index=7)
    c.rz(2, param_index=8)
    print(c)

    def loss_fn(params):
        state = c.run(params)
        flat = state_vector_flat(state)
        overlap = jnp.vdot(target_state, flat)
        fidelity = jnp.abs(overlap) ** 2
        return 1.0 - fidelity

    def adam_step(params, grads, m, v, t, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-08):
        t = t + 1
        m = beta1 * m + (1.0 - beta1) * grads
        v = beta2 * v + (1.0 - beta2) * grads ** 2
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        params = params - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        return (params, m, v, t)

    @jax.jit
    def train_step(params, m, v, t):
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        params, m, v, t = adam_step(params, grads, m, v, t)
        return (params, m, v, t, loss_val)
    key = jax.random.PRNGKey(42)
    params = jax.random.normal(key, shape=(c.num_params,)) * 0.1
    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)
    t = 0
    epochs = 100
    loss_history = []
    print('\nStarting optimization...')
    for epoch in range(1, epochs + 1):
        params, m, v, t, loss_val = train_step(params, m, v, t)
        loss_history.append(float(loss_val))
        if epoch == 1 or epoch % 10 == 0:
            fidelity = 1.0 - loss_val
            print(f'Epoch {epoch:3d} | Loss: {loss_val:.6f} | State Fidelity: {fidelity:.6%}')
    final_state = state_vector_flat(c.run(params))
    print('\nTarget State Vector: \n', target_state)
    print('Prepared State Vector: \n', jnp.round(final_state, 4))
    plt.figure(figsize=(10, 5), facecolor='#1e1e2e')
    ax = plt.subplot(1, 1, 1)
    ax.set_facecolor('#24273a')
    epochs_range = range(1, epochs + 1)
    fidelities = 1.0 - jnp.array(loss_history)
    plt.plot(epochs_range, loss_history, label='Loss (1 - Fidelity)', color='#f38ba8', linewidth=2.5)
    plt.plot(epochs_range, fidelities, label='Fidelity', color='#a6e3a1', linewidth=2.5)
    ax.tick_params(colors='#cdd6f4', labelsize=11)
    ax.xaxis.label.set_color('#cdd6f4')
    ax.yaxis.label.set_color('#cdd6f4')
    ax.title.set_color('#cdd6f4')
    plt.title('Convergence during Parameterized Quantum State Preparation', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, linestyle='--', color='#585b70', alpha=0.5)
    plt.legend(facecolor='#1e1e2e', edgecolor='#cba6f7', labelcolor='#cdd6f4')
    plot_dir = os.path.join(ROOT, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, '01_state_prep.png')
    plt.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'\n  🖼  Plot saved → {plot_path}')
if __name__ == '__main__':
    run_state_prep()
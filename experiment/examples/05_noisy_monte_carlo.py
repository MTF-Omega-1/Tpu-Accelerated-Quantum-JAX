import os
import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jax_qsim.circuit import Circuit
from jax_qsim import gates
import jax_qsim.statevector as sv
import jax_qsim.density_matrix as dm
os.makedirs('results', exist_ok=True)
state_plus_sv = jnp.array([1.0, 1.0], dtype=jnp.complex64) / jnp.sqrt(2.0)
state_plus_dm = jnp.outer(state_plus_sv, jnp.conj(state_plus_sv)).reshape(2, 2)
observable_x = sv.PauliString({0: 'X'})

def run_density_matrix_simulation(gamma):
    rho = state_plus_dm
    kraus = dm.phase_damping_kraus(gamma)
    rho_after = dm.apply_channel_1q(rho, kraus, 0)
    return dm.expectation_pauli_string(rho_after, observable_x)

def apply_channel_stochastic(state, kraus_ops, key, qubit):
    applied_states = []
    probs = []
    for K in kraus_ops:
        temp = sv.apply_gate(state, K, [qubit])
        applied_states.append(temp)
        probs.append(jnp.sum(jnp.abs(temp) ** 2))
    probs_jnp = jnp.stack(probs)
    probs_jnp = probs_jnp / jnp.sum(probs_jnp)
    idx = jax.random.choice(key, len(kraus_ops), p=probs_jnp)
    selected_state = jnp.stack(applied_states)[idx]
    norm = jnp.sqrt(jnp.sum(jnp.abs(selected_state) ** 2) + 1e-12)
    return selected_state / norm

def run_single_trajectory(key, gamma):
    state = state_plus_sv
    kraus = dm.phase_damping_kraus(gamma)
    state = apply_channel_stochastic(state, kraus, key, 0)
    phi = sv.apply_gate(state, gates.X(), [0])
    return jnp.real(jnp.vdot(state, phi))
run_trajectories_vectorized = jax.vmap(run_single_trajectory, in_axes=(0, None))

def run_experiment():
    print('=' * 80)
    print(' EXPERIMENT 5: Noisy Quantum Simulation - Monte Carlo vs Exact '.center(80, '='))
    print('=' * 80)
    print('Comparing expected <X> under Phase Damping (Dephasing) noise.')
    print('-' * 80)
    noise_rates = jnp.linspace(0.0, 0.99, 25)
    trajectory_counts = [10, 100, 1000]
    base_key = jax.random.PRNGKey(101)
    exact_analytical = [jnp.sqrt(1.0 - g) for g in noise_rates]
    exact_dm = [float(run_density_matrix_simulation(g)) for g in noise_rates]
    print(f'{'Damping Rate (gamma)':^22} | {'Analytical <X>':^18} | {'Exact DM <X>':^16} | {'Trajectory Runs'}')
    print('-' * 80)
    sample_gammas = [0.0, 0.2, 0.5, 0.8, 0.95]
    for g in sample_gammas:
        ana_val = jnp.sqrt(1.0 - g)
        dm_val = run_density_matrix_simulation(g)
        print(f'{g:^22.3f} | {ana_val:^18.5f} | {dm_val:^16.5f} | Completed')
    print('-' * 80)
    print('Computing parallelized stochastic trajectories...', flush=True)
    traj_results = {}
    t0 = time.time()
    for m in trajectory_counts:
        keys = jax.random.split(base_key, m)
        avg_x_vals = []
        for g in noise_rates:
            x_vals = run_trajectories_vectorized(keys, g)
            avg_x_vals.append(float(jnp.mean(x_vals)))
        traj_results[m] = avg_x_vals
        print(f'  Completed {m:^5d} parallelized trajectories in {time.time() - t0:.3f}s')
        t0 = time.time()
    print('=' * 80)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(11, 7), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    ax.plot(noise_rates, exact_analytical, color='#ffa657', lw=3.0, label='Exact Analytical Curve (sqrt(1-gamma))')
    ax.plot(noise_rates, exact_dm, color='#f78166', ls='--', lw=2.0, label='Exact Density Matrix Solver')
    colors = {10: '#ff7b72', 100: '#79c0ff', 1000: '#56d364'}
    for m in trajectory_counts:
        ax.scatter(noise_rates, traj_results[m], color=colors[m], s=35, alpha=0.85, edgecolor='#21262d', label=f'Monte Carlo Average ({m} runs)')
    ax.set_title('⚛  Noisy Quantum Channel — Stochastic Trajectories vs Exact Solver', fontsize=14, color='#e6edf3', fontweight='bold', pad=15)
    ax.set_xlabel('Dephasing Rate (gamma)', fontsize=12, color='#8b949e', labelpad=10)
    ax.set_ylabel('Expectation Value <X>', fontsize=12, color='#8b949e', labelpad=10)
    ax.grid(True, linestyle='--', color='#21262d', alpha=0.6)
    ax.tick_params(colors='#e6edf3', labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3', fontsize=11)
    plot_path = os.path.join('results', '05_noise_monte_carlo.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'Plot saved successfully to: {plot_path}')
    print('=' * 80)
if __name__ == '__main__':
    run_experiment()
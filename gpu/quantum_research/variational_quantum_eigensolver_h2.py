import os
import sys
import json
import time
from datetime import datetime
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jax_qsim.circuit import Circuit
from jax_qsim.observables import PauliString, Hamiltonian, expectation
H2_HAMILTONIAN_TERMS = [(-0.81054, {}), (0.1712, {0: 'Z'}), (-0.22278, {1: 'Z'}), (-0.22278, {2: 'Z'}), (0.1712, {3: 'Z'}), (0.12091, {0: 'Z', 1: 'Z'}), (0.16862, {0: 'Z', 2: 'Z'}), (0.17434, {1: 'Z', 2: 'Z'}), (0.04532, {0: 'Z', 3: 'Z'}), (0.16862, {1: 'Z', 3: 'Z'}), (0.12091, {2: 'Z', 3: 'Z'}), (0.04532, {0: 'X', 1: 'X', 2: 'Y', 3: 'Y'}), (-0.04532, {0: 'Y', 1: 'X', 2: 'X', 3: 'Y'}), (-0.04532, {0: 'X', 1: 'Y', 2: 'Y', 3: 'X'}), (0.04532, {0: 'Y', 1: 'Y', 2: 'X', 3: 'X'})]
FCI_ENERGY = -1.1372

def build_h2_hamiltonian() -> Hamiltonian:
    coeffs = [c for c, _ in H2_HAMILTONIAN_TERMS]
    paulis = [PauliString(term) for _, term in H2_HAMILTONIAN_TERMS]
    return Hamiltonian(coeffs, paulis)

def build_hea_circuit(num_layers: int=3) -> Circuit:
    c = Circuit(num_qubits=4)
    param_idx = 0
    c.x(2)
    c.x(3)
    for _ in range(num_layers):
        for q in range(4):
            c.ry(q, param_index=param_idx)
            param_idx += 1
            c.rz(q, param_index=param_idx)
            param_idx += 1
        for q in range(4):
            c.cnot(q, (q + 1) % 4)
    for q in range(4):
        c.ry(q, param_index=param_idx)
        param_idx += 1
        c.rz(q, param_index=param_idx)
        param_idx += 1
    return c

def adam_update(params, grads, m, v, t, lr=0.005, b1=0.9, b2=0.999, eps=1e-08):
    t = t + 1
    m = b1 * m + (1 - b1) * grads
    v = b2 * v + (1 - b2) * grads ** 2
    m_h = m / (1 - b1 ** t)
    v_h = v / (1 - b2 ** t)
    return (params - lr * m_h / (jnp.sqrt(v_h) + eps), m, v, t)

def run_vqe(num_layers: int=3, epochs: int=300, seed: int=42):
    print()
    print('╔══════════════════════════════════════════════════════════════════════╗')
    print('║  Variational Quantum Eigensolver (VQE) — H₂ Ground State Energy    ║')
    print('╠══════════════════════════════════════════════════════════════════════╣')
    print(f'║  Molecule    : H₂  (equilibrium bond length R = 0.735 Å)           ║')
    print(f'║  Basis set   : STO-3G  (Jordan-Wigner mapping, 4 qubits)           ║')
    print(f'║  FCI target  : {FCI_ENERGY:.6f} Hartree                              ║')
    print(f'║  Ansatz      : HEA {num_layers} layers  (UCCSD-inspired)               ║')
    print('╚══════════════════════════════════════════════════════════════════════╝')
    print()
    H = build_h2_hamiltonian()
    circuit = build_hea_circuit(num_layers=num_layers)
    print(f'  Circuit: {circuit}')
    print(f'  Hamiltonian terms: {len(H2_HAMILTONIAN_TERMS)}')
    print()

    def energy_fn(params):
        state = circuit.run(params)
        return H.expectation(state)
    value_and_grad = jax.jit(jax.value_and_grad(energy_fn))
    key = jax.random.PRNGKey(seed)
    params = jax.random.normal(key, shape=(circuit.num_params,)) * 0.05
    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)
    t = 0
    history = []
    t_start = time.perf_counter()
    print(f'  {'Epoch':>6}  {'Energy (Ha)':>14}  {'ΔE':>12}  {'|∇E|':>12}  {'Time (s)':>10}')
    print(f'  {'─' * 6}  {'─' * 14}  {'─' * 12}  {'─' * 12}  {'─' * 10}')
    prev_energy = None
    for epoch in range(1, epochs + 1):
        energy, grads = value_and_grad(params)
        params, m, v, t = adam_update(params, grads, m, v, t)
        e_val = float(energy)
        grad_norm = float(jnp.linalg.norm(grads))
        delta_e = e_val - prev_energy if prev_energy is not None else float('nan')
        elapsed = time.perf_counter() - t_start
        history.append({'epoch': epoch, 'energy': e_val, 'delta_e': delta_e, 'grad_norm': grad_norm, 'elapsed_s': elapsed})
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            marker = ' ✓' if abs(e_val - FCI_ENERGY) < 0.0016 else ''
            print(f'  {epoch:>6}  {e_val:>14.8f}  {delta_e:>+12.2e}  {grad_norm:>12.6f}  {elapsed:>10.2f}{marker}')
        prev_energy = e_val
    final_energy = history[-1]['energy']
    error_mhartree = abs(final_energy - FCI_ENERGY) * 1000
    print()
    print(f'  ╔══════════════════════════════════════════════╗')
    print(f'  ║  RESULTS                                     ║')
    print(f'  ╠══════════════════════════════════════════════╣')
    print(f'  ║  VQE energy          : {final_energy:+.8f} Ha     ║')
    print(f'  ║  FCI reference       : {FCI_ENERGY:+.8f} Ha     ║')
    print(f'  ║  Error               : {error_mhartree:.4f} mHartree     ║')
    print(f'  ║  Chemical accuracy   : {('✓ YES (<1.6 mHa)' if error_mhartree < 1.6 else f'✗ NO ({error_mhartree:.2f} mHa)')}       ║')
    print(f'  ╚══════════════════════════════════════════════╝')
    return (history, circuit)
PES_DATA = [(0.4, -0.8527), (0.5, -1.0284), (0.6, -1.0994), (0.7, -1.1279), (0.735, -1.1372), (0.8, -1.1378), (0.9, -1.1311), (1.0, -1.1186), (1.2, -1.0882), (1.5, -1.0374), (2.0, -0.9877), (2.5, -0.9694)]
PALETTE = {'bg': '#0d1117', 'panel': '#161b22', 'border': '#30363d', 'text': '#e6edf3', 'subtext': '#8b949e', 'accent1': '#58a6ff', 'accent2': '#3fb950', 'accent3': '#f78166', 'accent4': '#d2a8ff', 'accent5': '#ffa657', 'grid': '#21262d'}

def apply_theme(fig, axes):
    fig.patch.set_facecolor(PALETTE['bg'])
    for ax in axes if hasattr(axes, '__iter__') else [axes]:
        ax.set_facecolor(PALETTE['panel'])
        ax.tick_params(colors=PALETTE['text'], labelsize=10)
        ax.xaxis.label.set_color(PALETTE['text'])
        ax.yaxis.label.set_color(PALETTE['text'])
        ax.title.set_color(PALETTE['text'])
        for sp in ax.spines.values():
            sp.set_edgecolor(PALETTE['border'])
        ax.grid(True, color=PALETTE['grid'], linestyle='--', alpha=0.6, linewidth=0.7)

def plot_vqe_results(history, timestamp):
    os.makedirs('gpu/plots', exist_ok=True)
    os.makedirs('gpu/results', exist_ok=True)
    epochs = [h['epoch'] for h in history]
    energies = [h['energy'] for h in history]
    gnorms = [h['grad_norm'] for h in history]
    delta_e = [abs(h['delta_e']) if h['delta_e'] == h['delta_e'] else np.nan for h in history]
    fig = plt.figure(figsize=(16, 11), facecolor=PALETTE['bg'])
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35, left=0.08, right=0.97, top=0.91, bottom=0.07)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(epochs, energies, '-', color=PALETTE['accent1'], lw=2, label='VQE Energy')
    ax0.axhline(FCI_ENERGY, color=PALETTE['accent3'], ls='--', lw=1.5, label=f'FCI Reference ({FCI_ENERGY:.4f} Ha)')
    ax0.axhspan(FCI_ENERGY - 0.0016, FCI_ENERGY + 0.0016, color=PALETTE['accent2'], alpha=0.1, label='Chemical accuracy band')
    ax0.set_xlabel('Epoch')
    ax0.set_ylabel('Energy (Hartree)')
    ax0.set_title('⚛  VQE Energy Convergence — H₂ Ground State')
    ax0.legend(facecolor=PALETTE['panel'], edgecolor=PALETTE['border'], labelcolor=PALETTE['text'], fontsize=9)
    apply_theme(fig, ax0)
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.semilogy(epochs, gnorms, '-', color=PALETTE['accent4'], lw=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('|∇E|  (Hartree / rad) [log]')
    ax1.set_title('📉  Gradient Norm Decay')
    apply_theme(fig, ax1)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.semilogy(epochs[1:], delta_e[1:], '-', color=PALETTE['accent5'], lw=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('|ΔE| (Hartree) [log]')
    ax2.set_title('🔍  Energy Change per Step')
    apply_theme(fig, ax2)
    ax3 = fig.add_subplot(gs[1, 1])
    r_fci, e_fci = zip(*PES_DATA)
    ax3.plot(r_fci, e_fci, 'o-', color=PALETTE['accent2'], lw=2, ms=6, label='FCI / STO-3G')
    ax3.axvline(0.735, color=PALETTE['accent3'], ls=':', lw=1.5, label='Equilibrium R = 0.735 Å')
    ax3.scatter([0.735], [FCI_ENERGY], color=PALETTE['accent3'], s=100, zorder=5)
    vqe_e = history[-1]['energy']
    ax3.scatter([0.735], [vqe_e], color=PALETTE['accent1'], s=120, zorder=6, marker='*', label=f'VQE result ({vqe_e:.5f} Ha)')
    ax3.set_xlabel('Bond Length  R (Å)')
    ax3.set_ylabel('Energy (Hartree)')
    ax3.set_title('📊  H₂ Potential Energy Surface (STO-3G)')
    ax3.legend(facecolor=PALETTE['panel'], edgecolor=PALETTE['border'], labelcolor=PALETTE['text'], fontsize=9)
    apply_theme(fig, ax3)
    fig.suptitle(f'VQE — H₂ Molecule Ground State Energy  │  JAX Quantum Simulator\nHardware-Efficient Ansatz  │  {jax.default_backend().upper()}  │  {timestamp}', color=PALETTE['text'], fontsize=13, fontweight='bold', y=0.97)
    plot_path = f'gpu/plots/vqe_{timestamp}.png'
    plt.savefig(plot_path, dpi=180, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f'\n  🖼  VQE plot saved → {plot_path}')
    json_path = f'gpu/results/vqe_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump({'fci_energy': FCI_ENERGY, 'history': history}, f, indent=2)
    print(f'  📄 VQE history saved → {json_path}')
_run_vqe_impl = run_vqe

def run_vqe():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    history, circuit = _run_vqe_impl(num_layers=4, epochs=400)
    plot_vqe_results(history, ts)
if __name__ == '__main__':
    run_vqe()
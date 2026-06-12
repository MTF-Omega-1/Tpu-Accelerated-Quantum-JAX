import os
import sys
import gc
import time
import subprocess
import csv
import json
from datetime import datetime
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.90')
os.environ.setdefault('TF_GPU_ALLOCATOR', 'cuda_malloc_async')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jax_qsim.core import zero_state, apply_gate
from jax_qsim import ops

def get_vram_mib() -> float:
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        lines = result.stdout.strip().split('\n')
        return float(lines[0].strip())
    except Exception:
        return 0.0

def get_vram_total_mib() -> float:
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        return float(result.stdout.strip().split('\n')[0].strip())
    except Exception:
        return 4096.0

def state_size_bytes(n: int) -> int:
    return 2 ** n * 8

def format_bytes(b: int) -> str:
    if b < 1024:
        return f'{b} B'
    elif b < 1 << 20:
        return f'{b / 1024:.1f} KB'
    elif b < 1 << 30:
        return f'{b / (1 << 20):.1f} MB'
    else:
        return f'{b / (1 << 30):.2f} GB'

def build_benchmark_circuit(n: int):
    from jax_qsim.circuit import Circuit
    num_layers = 2
    c = Circuit(num_qubits=n)
    param_idx = 0
    for _ in range(num_layers):
        for q in range(n):
            c.h(q)
        for q in range(n - 1):
            c.cnot(q, q + 1)
        for q in range(n):
            c.ry(q, param_index=param_idx)
            param_idx += 1
        for q in range(n):
            c.rz(q, param_index=param_idx)
            param_idx += 1
    return c
VRAM_HARD_LIMIT_MIB = 3788

def _free_gpu_memory():
    gc.collect()
    try:
        jax.clear_caches()
    except Exception:
        pass

def _vram_headroom_ok(state_bytes: int, vram_total_mib: float) -> bool:
    needed_mib = state_bytes * 2.5 / (1 << 20)
    current_mib = get_vram_mib()
    if current_mib + needed_mib > VRAM_HARD_LIMIT_MIB:
        return False
    return True

def benchmark_n_qubits(n: int, backend: str, num_repeats: int=5):
    sb = state_size_bytes(n)
    vram_total = get_vram_total_mib()
    if backend == 'gpu' and (not _vram_headroom_ok(sb, vram_total)):
        return 'OOM'
    circuit = build_benchmark_circuit(n)
    params = jnp.ones(circuit.num_params, dtype=jnp.float32)
    try:
        vram_before = get_vram_mib()
        t0 = time.perf_counter()
        state_raw = circuit.run(params)
        state_raw.block_until_ready()
        t_uncompiled = time.perf_counter() - t0
        vram_after = get_vram_mib()
        vram_used_mib = max(0, vram_after - vram_before)
    except Exception:
        _free_gpu_memory()
        return 'OOM'
    try:
        jit_run = jax.jit(circuit.run)
        t0 = time.perf_counter()
        state_jit = jit_run(params)
        state_jit.block_until_ready()
        t_jit_compile = time.perf_counter() - t0
    except Exception:
        _free_gpu_memory()
        return 'OOM'
    jit_times = []
    for _ in range(num_repeats):
        try:
            t0 = time.perf_counter()
            state_jit = jit_run(params)
            state_jit.block_until_ready()
            jit_times.append(time.perf_counter() - t0)
        except Exception:
            break
    _free_gpu_memory()
    if not jit_times:
        return 'OOM'
    t_jit_mean = float(np.mean(jit_times))
    t_jit_std = float(np.std(jit_times))
    speedup = t_uncompiled / t_jit_mean if t_jit_mean > 0 else 0
    total_gates = len(circuit.gates)
    gates_per_second = total_gates / t_jit_mean if t_jit_mean > 0 else 0
    return {'n_qubits': n, 'backend': backend, 'state_size_bytes': sb, 'state_size_str': format_bytes(sb), 'num_gates': total_gates, 'num_params': circuit.num_params, 't_uncompiled_s': t_uncompiled, 't_jit_compile_s': t_jit_compile, 't_jit_mean_s': t_jit_mean, 't_jit_std_s': t_jit_std, 'speedup_x': speedup, 'gates_per_second': gates_per_second, 'vram_used_mib': vram_used_mib, 'vram_total_mib': vram_total}

def run_full_benchmark():
    backend = jax.default_backend()
    device_str = str(jax.devices()[0])
    vram_total = get_vram_total_mib()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('gpu/results', exist_ok=True)
    os.makedirs('gpu/plots', exist_ok=True)
    print()
    print('╔══════════════════════════════════════════════════════════════════════╗')
    print('║     JAX Quantum Simulator — GPU VRAM Scaling Benchmark Suite        ║')
    print('╠══════════════════════════════════════════════════════════════════════╣')
    print(f'║  Backend  : {backend:<57} ║')
    print(f'║  Device   : {device_str:<57} ║')
    print(f'║  VRAM     : {vram_total:.0f} MiB  ({vram_total / 1024:.2f} GB){' ' * 38} ║')
    print(f'║  Time     : {timestamp:<57} ║')
    print('╚══════════════════════════════════════════════════════════════════════╝')
    print()
    print('  Warming up JAX XLA runtime...', end='', flush=True)
    _ = jax.jit(lambda x: jnp.dot(x, x))(jnp.ones((64, 64))).block_until_ready()
    print(' done.\n')
    HDR = ('Qubits', 'State Size', 'Gates', 'Uncompiled(s)', 'JIT Compile(s)', 'JIT Exec (s)', '±Std', 'Speedup', 'VRAM Used', 'Throughput')
    col_widths = (7, 11, 7, 14, 15, 13, 8, 9, 11, 18)
    sep = '─' * sum(col_widths) + '─' * (len(col_widths) * 3 - 1)

    def fmt_row(*vals):
        return ' │ '.join((str(v).ljust(w) for v, w in zip(vals, col_widths)))
    print('  ' + sep)
    print('  ' + fmt_row(*HDR))
    print('  ' + sep)
    results = []
    qubit_range = list(range(4, 30))
    for n in qubit_range:
        sb_mib = state_size_bytes(n) / (1 << 20)
        if sb_mib > VRAM_HARD_LIMIT_MIB:
            print(f'  │  n={n:2d}  │  State vector ({format_bytes(state_size_bytes(n))}) exceeds {VRAM_HARD_LIMIT_MIB} MiB cap. Stopping.')
            break
        r = benchmark_n_qubits(n, backend)
        if r == 'OOM':
            vram_now = get_vram_mib()
            print(f'  │  n={n:2d}  │  ⛔ OOM — VRAM {vram_now:.0f}/{vram_total:.0f} MiB  ({vram_now / vram_total * 100:.0f}%) — 3.7 GB cap reached. Stopping.')
            break
        if r is None:
            print(f'  │  n={n:2d}  │  Unexpected error — skipping.  │')
            continue
        results.append(r)
        throughput_str = f'{r['gates_per_second']:.1f} gates/s' if r['gates_per_second'] < 1000000.0 else f'{r['gates_per_second'] / 1000000.0:.2f}M gates/s'
        vram_str = f'{r['vram_used_mib']:.0f} MiB' if r['vram_used_mib'] > 0 else 'GPU (delta N/A)'
        row = fmt_row(n, r['state_size_str'], r['num_gates'], f'{r['t_uncompiled_s']:.5f}', f'{r['t_jit_compile_s']:.5f}', f'{r['t_jit_mean_s']:.5f}', f'±{r['t_jit_std_s']:.5f}', f'{r['speedup_x']:.1f}×', vram_str, throughput_str)
        print('  ' + row)
        vram_now = get_vram_mib()
        if backend == 'gpu' and vram_now >= VRAM_HARD_LIMIT_MIB:
            print(f'\n  ✅ VRAM at {vram_now:.0f} / {vram_total:.0f} MiB ({vram_now / vram_total * 100:.1f}%) — 3.7 GB target reached. Stopping safely.')
            break
    print('  ' + sep)
    print(f'\n  Total qubit configurations benchmarked: {len(results)}')
    print(f'  Peak qubit count achieved: {results[-1]['n_qubits']} qubits ({results[-1]['state_size_str']} state vector)\n')
    csv_path = f'gpu/results/benchmark_{timestamp}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f'  📄 CSV results saved → {csv_path}')
    json_path = f'gpu/results/benchmark_{timestamp}.json'
    meta = {'timestamp': timestamp, 'backend': backend, 'device': device_str, 'vram_total_mib': vram_total, 'results': results}
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  📄 JSON results saved → {json_path}')
    plot_benchmark_results(results, timestamp, backend, device_str)
    return results
PALETTE = {'bg': '#0d1117', 'panel': '#161b22', 'border': '#30363d', 'text': '#e6edf3', 'subtext': '#8b949e', 'accent1': '#58a6ff', 'accent2': '#3fb950', 'accent3': '#f78166', 'accent4': '#d2a8ff', 'accent5': '#ffa657', 'grid': '#21262d'}

def apply_theme(fig, axes):
    fig.patch.set_facecolor(PALETTE['bg'])
    for ax in axes if hasattr(axes, '__iter__') else [axes]:
        ax.set_facecolor(PALETTE['panel'])
        ax.tick_params(colors=PALETTE['text'], labelsize=10)
        ax.xaxis.label.set_color(PALETTE['text'])
        ax.yaxis.label.set_color(PALETTE['text'])
        ax.title.set_color(PALETTE['text'])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE['border'])
        ax.grid(True, color=PALETTE['grid'], linestyle='--', alpha=0.6, linewidth=0.7)

def plot_benchmark_results(results, timestamp, backend, device_str):
    ns = [r['n_qubits'] for r in results]
    t_unc = [r['t_uncompiled_s'] for r in results]
    t_jit_c = [r['t_jit_compile_s'] for r in results]
    t_jit = [r['t_jit_mean_s'] for r in results]
    t_std = [r['t_jit_std_s'] for r in results]
    speedups = [r['speedup_x'] for r in results]
    sizes_mb = [r['state_size_bytes'] / (1 << 20) for r in results]
    vrams = [r['vram_used_mib'] for r in results]
    throughputs = [r['gates_per_second'] for r in results]
    fig = plt.figure(figsize=(18, 14), facecolor=PALETTE['bg'])
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35, left=0.07, right=0.97, top=0.92, bottom=0.06)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.semilogy(ns, t_unc, 'o--', color=PALETTE['accent3'], lw=2, label='Uncompiled', ms=6)
    ax0.semilogy(ns, t_jit_c, 's--', color=PALETTE['accent5'], lw=2, label='JIT (compile+exec)', ms=6)
    ax0.fill_between(ns, [m - s for m, s in zip(t_jit, t_std)], [m + s for m, s in zip(t_jit, t_std)], color=PALETTE['accent1'], alpha=0.2)
    ax0.semilogy(ns, t_jit, 'd-', color=PALETTE['accent1'], lw=2.5, label='JIT exec (mean ± σ)', ms=7)
    ax0.set_xlabel('Number of Qubits')
    ax0.set_ylabel('Execution Time (s) [log]')
    ax0.set_title('⏱  Execution Time Scaling')
    ax0.legend(facecolor=PALETTE['panel'], edgecolor=PALETTE['border'], labelcolor=PALETTE['text'], fontsize=9)
    ax0.set_xticks(ns)
    apply_theme(fig, ax0)
    ax1 = fig.add_subplot(gs[0, 1])
    bars = ax1.bar(ns, speedups, color=PALETTE['accent2'], alpha=0.85, edgecolor=PALETTE['border'], linewidth=0.8)
    for bar, sp in zip(bars, speedups):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f'{sp:.1f}×', ha='center', va='bottom', color=PALETTE['text'], fontsize=8)
    ax1.set_xlabel('Number of Qubits')
    ax1.set_ylabel('Speedup  (Uncompiled / JIT)')
    ax1.set_title('🚀  JIT Compilation Speedup over Eager Execution')
    ax1.set_xticks(ns)
    apply_theme(fig, ax1)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.semilogy(ns, sizes_mb, 'o-', color=PALETTE['accent4'], lw=2.5, ms=7)
    if max(sizes_mb) > 0:
        ax2.axhline(y=4096, color=PALETTE['accent3'], ls='--', lw=1.5, label='4 GB VRAM limit (GeForce RTX 2050)')
        ax2.axhline(y=100, color=PALETTE['accent5'], ls=':', lw=1.2, label='100 MB marker')
        ax2.legend(facecolor=PALETTE['panel'], edgecolor=PALETTE['border'], labelcolor=PALETTE['text'], fontsize=9)
    ax2.set_xlabel('Number of Qubits')
    ax2.set_ylabel('State-Vector Size (MiB) [log]')
    ax2.set_title('💾  Memory Footprint vs Qubit Count  (2ⁿ × 8 bytes)')
    ax2.set_xticks(ns)
    apply_theme(fig, ax2)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(ns, [t / 1000000.0 for t in throughputs], 's-', color=PALETTE['accent5'], lw=2.5, ms=7)
    ax3.set_xlabel('Number of Qubits')
    ax3.set_ylabel('Gate Throughput  (Mgates / s)')
    ax3.set_title('⚡  Gate Throughput on GPU (JIT compiled)')
    ax3.set_xticks(ns)
    apply_theme(fig, ax3)
    ax4 = fig.add_subplot(gs[2, 0])
    vram_vals = [v if v > 0 else 0 for v in vrams]
    ax4.bar(ns, vram_vals, color=PALETTE['accent1'], alpha=0.8, edgecolor=PALETTE['border'], linewidth=0.8)
    ax4.axhline(y=4096, color=PALETTE['accent3'], ls='--', lw=1.5, label='4 GB VRAM ceiling')
    ax4.set_xlabel('Number of Qubits')
    ax4.set_ylabel('VRAM Delta Used (MiB)')
    ax4.set_title('🎮  VRAM Consumption on NVIDIA GeForce RTX 2050')
    ax4.legend(facecolor=PALETTE['panel'], edgecolor=PALETTE['border'], labelcolor=PALETTE['text'], fontsize=9)
    ax4.set_xticks(ns)
    apply_theme(fig, ax4)
    ax5 = fig.add_subplot(gs[2, 1])
    log_t = np.log2(np.array(t_jit) + 1e-12)
    coeffs = np.polyfit(ns, log_t, 1)
    fit_line = np.poly1d(coeffs)
    ns_fine = np.linspace(min(ns), max(ns), 200)
    ax5.scatter(ns, t_jit, color=PALETTE['accent1'], s=55, zorder=5, label='JIT exec data')
    ax5.plot(ns_fine, 2 ** fit_line(ns_fine), '-', color=PALETTE['accent3'], lw=2.5, label=f'Exp. fit: ×2^({coeffs[0]:.3f}n)')
    ax5.set_yscale('log')
    ax5.set_xlabel('Number of Qubits')
    ax5.set_ylabel('Execution Time (s) [log]')
    ax5.set_title(f'📈  Exponential Scaling Law  (slope ≈ {coeffs[0]:.3f} bits/qubit)')
    ax5.legend(facecolor=PALETTE['panel'], edgecolor=PALETTE['border'], labelcolor=PALETTE['text'], fontsize=9)
    ax5.set_xticks(ns)
    apply_theme(fig, ax5)
    fig.suptitle(f'JAX Quantum Simulator — GPU Scaling Benchmark  │  {backend.upper()}  │  {device_str}\nGenerated: {timestamp}', color=PALETTE['text'], fontsize=14, fontweight='bold', y=0.98)
    plot_path = f'gpu/plots/benchmark_{timestamp}.png'
    plt.savefig(plot_path, dpi=180, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f'  🖼  Multi-panel benchmark plot saved → {plot_path}')
if __name__ == '__main__':
    run_full_benchmark()
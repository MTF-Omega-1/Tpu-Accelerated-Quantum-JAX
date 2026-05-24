#!/usr/bin/env python3
"""
================================================================================
  JAX TPU v5lite-16 Quantum Scaling Suite (Headless Benchmark & Visualizer)
  Pure JAX High-Performance Sharded Quantum Simulator with Memory Safeguards
  Saves CSV, JSON, and High-DPI Plot Dashboards directly on the TPU SSH VM
================================================================================
"""
import os
import sys
import time
import math
import csv
import json
from datetime import datetime
import numpy as np

# Configure headless plotting BEFORE importing matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Configure JAX to allocate memory dynamically
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async" 

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

# ─────────────────────────────────────────────────────────────────────────────
# Theme & Palette Definition
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":        "#0d1117",
    "panel":     "#161b22",
    "border":    "#30363d",
    "text":      "#e6edf3",
    "subtext":   "#8b949e",
    "accent1":   "#58a6ff",  # TPU Blue
    "accent2":   "#3fb950",  # Success Green
    "accent3":   "#f78166",  # Safety Coral
    "accent4":   "#d2a8ff",  # Deep Purple
    "accent5":   "#ffa657",  # Warning Gold
    "grid":      "#21262d",
}

def apply_theme(fig, axes):
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor(PALETTE["panel"])
        ax.tick_params(colors=PALETTE["text"], labelsize=10)
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])
        ax.grid(True, color=PALETTE["grid"], linestyle='--', alpha=0.6, linewidth=0.7)

def format_bytes(size_bytes):
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def get_tpu_hbm_used_mib():
    """Queries JAX native device memory stats for actual HBM consumption in MiB."""
    try:
        stats = jax.devices()[0].memory_stats()
        if stats is not None and "bytes_in_use" in stats:
            return stats["bytes_in_use"] / (1024 * 1024)
    except Exception:
        pass
    return 0.0

def print_header(title):
    width = 80
    print("\n" + "═" * width)
    print(f" {title.center(width - 2)} ")
    print("═" * width)

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    os.makedirs("examples/plots", exist_ok=True)

    # ==========================================================================
    # 1. HARDWARE DETECTION & CONFIGURATION
    # ==========================================================================
    print_header("JAX HARDWARE DETECTION")
    backend = jax.default_backend()
    devices = jax.devices()
    num_devices = len(devices)
    device_str = f"{devices[0].device_kind} cluster"
    
    print(f"  JAX Backend          : {backend.upper()}")
    print(f"  Detected Devices     : {num_devices}")
    for idx, d in enumerate(devices):
        print(f"    - Device {idx:2d}        : {d}")

    # TPU v5e/v5lite has 16 GB of HBM per chip.
    MEM_PER_DEVICE_GB = 16.0
    RESERVED_GB_PER_DEVICE = 4.0  # Headroom in GB to leave completely free
    
    total_raw_mem_gb = num_devices * MEM_PER_DEVICE_GB
    total_reserved_gb = num_devices * RESERVED_GB_PER_DEVICE
    usable_mem_gb = total_raw_mem_gb - total_reserved_gb
    
    print("\n  Memory Limits Configuration:")
    print(f"    - Raw HBM per Device: {MEM_PER_DEVICE_GB:.1f} GB")
    print(f"    - Reserved Headroom : {RESERVED_GB_PER_DEVICE:.1f} GB per device (for OS/other works)")
    print(f"    - Usable TPU Memory : {usable_mem_gb:.1f} GB (Total across {num_devices} devices)")
    
    # ==========================================================================
    # 2. SHARDING SETUP (MESH & NAMED SHARDING)
    # ==========================================================================
    is_power_of_2 = (num_devices & (num_devices - 1) == 0) and num_devices > 0
    k_shards = int(math.log2(num_devices)) if is_power_of_2 else 0
    
    if is_power_of_2 and num_devices > 1:
        mesh_shape = (2,) * k_shards
        mesh_devices = mesh_utils.create_device_mesh(mesh_shape)
        mesh_axis_names = [f"axis_{i}" for i in range(k_shards)]
        device_mesh = Mesh(mesh_devices, mesh_axis_names)
        print(f"  Multi-Device Mesh    : Enabled ({' × '.join(map(str, mesh_shape))} grid across {num_devices} devices)")
    else:
        device_mesh = None
        print("  Multi-Device Mesh    : Disabled (Single device or non-power-of-2 device count)")

    # ==========================================================================
    # 3. PURE JAX QUANTUM SIMULATOR ENGINE (JIT-COMPILABLE)
    # ==========================================================================
    # Define unitary gate matrices
    H_matrix = jnp.array([[1.0, 1.0], [1.0, -1.0]]) / jnp.sqrt(2.0)
    
    def rx_gate(theta):
        c = jnp.cos(theta / 2.0)
        s = -1j * jnp.sin(theta / 2.0)
        return jnp.array([[c, s], [s, c]])
        
    def ry_gate(theta):
        c = jnp.cos(theta / 2.0)
        s = jnp.sin(theta / 2.0)
        return jnp.array([[c, -s], [s, c]])

    def rz_gate(theta):
        val = jnp.exp(-1j * theta / 2.0)
        return jnp.array([[val, 0.0], [0.0, jnp.conj(val)]])

    # 1-Qubit Gate Application
    def apply_gate(state, gate, target, num_qubits):
        out = jnp.tensordot(gate, state, axes=((1,), (target,)))
        dest_axes = list(range(1, num_qubits))
        dest_axes.insert(target, 0)
        return jnp.transpose(out, dest_axes)

    # 2-Qubit Gate Application (CNOT)
    cnot_gate = jnp.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype=jnp.complex64).reshape(2, 2, 2, 2)

    def apply_cnot(state, control, target, num_qubits):
        out = jnp.tensordot(cnot_gate, state, axes=((2, 3), (control, target)))
        remaining_axes = [i for i in range(num_qubits) if i != control and i != target]
        dest = [0] * num_qubits
        dest[control] = 0
        dest[target] = 1
        rem_idx = 0
        for i in range(num_qubits):
            if i != control and i != target:
                dest[i] = rem_idx + 2
                rem_idx += 1
        return jnp.transpose(out, dest)

    # Differentiable Cost Circuit
    def execute_circuit(params, num_qubits):
        state = jnp.zeros((2,) * num_qubits, dtype=jnp.complex64)
        state = state.at[(0,) * num_qubits].set(1.0)
        for i in range(num_qubits):
            state = apply_gate(state, H_matrix, i, num_qubits)
        for i in range(num_qubits):
            state = apply_gate(state, rx_gate(params[i]), i, num_qubits)
            state = apply_gate(state, ry_gate(params[i + num_qubits]), i, num_qubits)
            state = apply_gate(state, rz_gate(params[i + 2 * num_qubits]), i, num_qubits)
        for i in range(num_qubits - 1):
            state = apply_cnot(state, i, i + 1, num_qubits)
        state = apply_cnot(state, num_qubits - 1, 0, num_qubits)
        
        probs = jnp.abs(state) ** 2
        sum_axes = tuple(range(1, num_qubits))
        marginal = jnp.sum(probs, axis=sum_axes)
        return marginal[0] - marginal[1]

    # ==========================================================================
    # 4. QUBIT SCALING BENCHMARK WITH HEADROOM SAFEGUARDS
    # ==========================================================================
    print_header("STARTING SCALING BENCHMARK LOOP")
    
    start_qubits = 10
    max_qubits = 40
    
    # Headers for the table
    HDR = ("Qubits", "State Size", "Gates", "Uncompiled(s)", "JIT Compile(s)", "JIT Exec (s)", "Speedup", "HBM Used", "Throughput")
    col_widths = (7, 11, 7, 14, 15, 13, 9, 11, 18)
    sep = "─" * sum(col_widths) + "─" * (len(col_widths) * 3 - 1)
    
    def fmt_row(*vals):
        return " │ ".join(str(v).ljust(w) for v, w in zip(vals, col_widths))
        
    print("  " + sep)
    print("  " + fmt_row(*HDR))
    print("  " + sep)
    
    results = []
    
    for n in range(start_qubits, max_qubits + 1):
        sb_bytes = (2 ** n) * 8
        sb_gb = sb_bytes / (1024 ** 3)
        num_gates = 1 + n * 4 + n  # Hadarmards, Rotations, CNOTs
        num_params = 3 * n
        
        if sb_gb > usable_mem_gb:
            print("  " + sep)
            print(f"  │  n={n:2d}  │  State vector ({format_bytes(sb_bytes)}) exceeds "
                  f"{usable_mem_gb:.2f} GB memory safety cap. Stopping.")
            print("  " + sep)
            break
            
        # Determine parameter layout and device mesh sharding mapping
        params = jnp.ones((num_params,), dtype=jnp.float32) * 0.5
        
        if device_mesh is not None and n >= k_shards:
            partition_spec = P(*[f"axis_{i}" for i in range(k_shards)], *([None] * (n - k_shards)))
            sharding = NamedSharding(device_mesh, partition_spec)
            
            @jax.jit
            def sharded_cost(p):
                sharded_p = jax.device_put(p, NamedSharding(device_mesh, P(None)))
                return execute_circuit(sharded_p, n)
        else:
            @jax.jit
            def sharded_cost(p):
                return execute_circuit(p, n)
                
        grad_fn = jax.jit(jax.grad(sharded_cost))
        
        # ── Eager (Uncompiled) execution benchmark (Skipped above 25 qubits to avoid freezing) ──
        if n <= 25:
            try:
                t0 = time.perf_counter()
                # Run gradient eagerly by disabling JIT compilation via jax.disable_jit
                with jax.disable_jit():
                    eager_grads = jax.grad(execute_circuit)(params, n)
                    eager_grads.block_until_ready()
                t_uncompiled = time.perf_counter() - t0
            except Exception:
                t_uncompiled = np.nan
        else:
            t_uncompiled = np.nan

        # ── First JIT execution (Compile + Run) ──
        hbm_before = get_tpu_hbm_used_mib()
        try:
            t0 = time.perf_counter()
            grads = grad_fn(params)
            grads.block_until_ready()
            t_jit_compile = time.perf_counter() - t0
            hbm_after = get_tpu_hbm_used_mib()
            hbm_used_mib = max(0.0, hbm_after - hbm_before)
        except Exception as e:
            print(f"  │  n={n:2d}  │  Execution failed: {e}. Stopping benchmark.")
            break

        # ── Subsequent JIT runs (Pure compiled execution) ──
        jit_times = []
        num_repeats = 5
        for _ in range(num_repeats):
            t0 = time.perf_counter()
            grads = grad_fn(params)
            grads.block_until_ready()
            jit_times.append(time.perf_counter() - t0)
            
        t_jit_mean = float(np.mean(jit_times))
        t_jit_std = float(np.std(jit_times))
        
        # Calculate scaling stats
        speedup = t_uncompiled / t_jit_mean if (t_jit_mean > 0 and not np.isnan(t_uncompiled)) else 0.0
        gates_per_second = num_gates / t_jit_mean if t_jit_mean > 0 else 0.0
        
        # String representations
        speedup_str = f"{speedup:.1f}×" if speedup > 0 else "N/A"
        eager_str = f"{t_uncompiled:.5f}" if not np.isnan(t_uncompiled) else "Skipped"
        hbm_str = f"{hbm_used_mib:.1f} MiB" if hbm_used_mib > 0 else "N/A"
        throughput_str = f"{gates_per_second/1e6:.2f}M gates/s" if gates_per_second >= 1e6 else f"{gates_per_second:.1f} gates/s"
        
        r = {
            "n_qubits": n,
            "backend": backend,
            "device": device_str,
            "state_size_bytes": sb_bytes,
            "state_size_str": format_bytes(sb_bytes),
            "num_gates": num_gates,
            "num_params": num_params,
            "t_uncompiled_s": t_uncompiled if not np.isnan(t_uncompiled) else 0.0,
            "t_jit_compile_s": t_jit_compile,
            "t_jit_mean_s": t_jit_mean,
            "t_jit_std_s": t_jit_std,
            "speedup_x": speedup,
            "gates_per_second": gates_per_second,
            "hbm_used_mib": hbm_used_mib,
            "hbm_total_gb": total_raw_mem_gb
        }
        results.append(r)
        
        row = fmt_row(
            n,
            r["state_size_str"],
            r["num_gates"],
            eager_str,
            f"{t_jit_compile:.5f}",
            f"{t_jit_mean:.5f}",
            speedup_str,
            hbm_str,
            throughput_str
        )
        print("  " + row)

    print("  " + sep)
    print(f"\n  Total qubit configurations benchmarked: {len(results)}")
    print(f"  Peak qubit count achieved: {results[-1]['n_qubits']} qubits "
          f"({results[-1]['state_size_str']} state vector)\n")

    # ==========================================================================
    # 5. SAVE CSV & JSON RESULTS
    # ==========================================================================
    csv_path = f"results/tpu_benchmark_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  📄 CSV results saved → {csv_path}")

    json_path = f"results/tpu_benchmark_{timestamp}.json"
    meta = {
        "timestamp": timestamp,
        "backend": backend,
        "device": device_str,
        "num_devices": num_devices,
        "total_raw_mem_gb": total_raw_mem_gb,
        "usable_mem_gb": usable_mem_gb,
        "results": results
    }
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  📄 JSON results saved → {json_path}")

    # ==========================================================================
    # 6. GENERATE HEADLESS PUBLICATION-QUALITY PLOTS
    # ==========================================================================
    ns = [r["n_qubits"] for r in results]
    t_unc = [r["t_uncompiled_s"] for r in results]
    t_jit_c = [r["t_jit_compile_s"] for r in results]
    t_jit = [r["t_jit_mean_s"] for r in results]
    t_std = [r["t_jit_std_s"] for r in results]
    speedups = [r["speedup_x"] for r in results]
    sizes_mb = [r["state_size_bytes"] / (1 << 20) for r in results]
    vrams = [r["hbm_used_mib"] for r in results]
    throughputs = [r["gates_per_second"] for r in results]

    fig = plt.figure(figsize=(18, 14), facecolor=PALETTE["bg"])
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35,
                            left=0.07, right=0.97, top=0.92, bottom=0.06)

    # ── (0,0) Execution time scaling (log) ──
    ax0 = fig.add_subplot(gs[0, 0])
    valid_unc_idx = [i for i, t in enumerate(t_unc) if t > 0]
    if valid_unc_idx:
        ax0.semilogy([ns[i] for i in valid_unc_idx], [t_unc[i] for i in valid_unc_idx],
                     'o--', color=PALETTE["accent3"], lw=2, label='Uncompiled (Eager)', ms=6)
    ax0.semilogy(ns, t_jit_c, 's--', color=PALETTE["accent5"], lw=2, label='JIT (compile+exec)', ms=6)
    ax0.fill_between(ns,
                     [m - s for m, s in zip(t_jit, t_std)],
                     [m + s for m, s in zip(t_jit, t_std)],
                     color=PALETTE["accent1"], alpha=0.2)
    ax0.semilogy(ns, t_jit, 'd-', color=PALETTE["accent1"], lw=2.5, label='JIT exec (mean ± σ)', ms=7)
    ax0.set_xlabel("Number of Qubits")
    ax0.set_ylabel("Execution Time (s) [log]")
    ax0.set_title("⏱  Execution Time Scaling")
    ax0.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
               labelcolor=PALETTE["text"], fontsize=9)
    ax0.set_xticks(ns)
    apply_theme(fig, ax0)

    # ── (0,1) JIT Speedup ──
    ax1 = fig.add_subplot(gs[0, 1])
    valid_speedups = [sp for sp in speedups if sp > 0]
    valid_ns = [ns[i] for i, sp in enumerate(speedups) if sp > 0]
    if valid_speedups:
        bars = ax1.bar(valid_ns, valid_speedups, color=PALETTE["accent2"], alpha=0.85,
                       edgecolor=PALETTE["border"], linewidth=0.8)
        for bar, sp in zip(bars, valid_speedups):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f'{sp:.1f}×', ha='center', va='bottom',
                     color=PALETTE["text"], fontsize=8)
    ax1.set_xlabel("Number of Qubits")
    ax1.set_ylabel("Speedup  (Uncompiled / JIT)")
    ax1.set_title("🚀  JIT Compilation Speedup over Eager Execution")
    ax1.set_xticks(ns)
    apply_theme(fig, ax1)

    # ── (1,0) State-vector memory footprint ──
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.semilogy(ns, sizes_mb, 'o-', color=PALETTE["accent4"], lw=2.5, ms=7)
    # Add horizontal markers representing TPU HBM capacity per chip (16 GB)
    ax2.axhline(y=16384, color=PALETTE["accent3"], ls='--', lw=1.5,
                label='Single TPU chip HBM ceiling (16 GB)')
    ax2.axhline(y=usable_mem_gb * 1024, color=PALETTE["accent5"], ls=':', lw=1.5,
                label=f'Safety HBM cap ({usable_mem_gb:.0f} GB)')
    ax2.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
               labelcolor=PALETTE["text"], fontsize=9)
    ax2.set_xlabel("Number of Qubits")
    ax2.set_ylabel("State-Vector Size (MiB) [log]")
    ax2.set_title("💾  Memory Footprint vs Qubit Count (2ⁿ × 8 bytes)")
    ax2.set_xticks(ns)
    apply_theme(fig, ax2)

    # ── (1,1) Gate Throughput ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(ns, [t/1e6 for t in throughputs], 's-', color=PALETTE["accent5"], lw=2.5, ms=7)
    ax3.set_xlabel("Number of Qubits")
    ax3.set_ylabel("Gate Throughput (Mgates / s)")
    ax3.set_title("⚡  Gate Throughput on TPU (JIT compiled)")
    ax3.set_xticks(ns)
    apply_theme(fig, ax3)

    # ── (2,0) Live HBM allocation delta ──
    ax4 = fig.add_subplot(gs[2, 0])
    hbm_vals = [v if v > 0 else 0.0 for v in vrams]
    ax4.bar(ns, hbm_vals, color=PALETTE["accent1"], alpha=0.8,
            edgecolor=PALETTE["border"], linewidth=0.8)
    ax4.set_xlabel("Number of Qubits")
    ax4.set_ylabel("TPU HBM Allocated Delta (MiB)")
    ax4.set_title(f"🎮  HBM Consumption Delta per Scale step ({num_devices} device mesh)")
    ax4.set_xticks(ns)
    apply_theme(fig, ax4)

    # ── (2,1) Exponential fit annotation ──
    ax5 = fig.add_subplot(gs[2, 1])
    log_t = np.log2(np.array(t_jit) + 1e-12)
    coeffs = np.polyfit(ns, log_t, 1)
    fit_line = np.poly1d(coeffs)
    ns_fine = np.linspace(min(ns), max(ns), 200)
    ax5.scatter(ns, t_jit, color=PALETTE["accent1"], s=55, zorder=5, label='JIT exec data')
    ax5.plot(ns_fine, 2**fit_line(ns_fine), '-', color=PALETTE["accent3"],
             lw=2.5, label=f'Exp. fit: ×2^({coeffs[0]:.3f}n)')
    ax5.set_yscale('log')
    ax5.set_xlabel("Number of Qubits")
    ax5.set_ylabel("Execution Time (s) [log]")
    ax5.set_title(f"📈  Exponential Scaling Law (slope ≈ {coeffs[0]:.3f} bits/qubit)")
    ax5.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
               labelcolor=PALETTE["text"], fontsize=9)
    ax5.set_xticks(ns)
    apply_theme(fig, ax5)

    # ── Super-title ──
    fig.suptitle(
        f"JAX TPU Scaling Benchmark Dashboard  │  {backend.upper()}  │  {device_str}\n"
        f"Generated: {timestamp} (v5litepod-16 safety constraint enabled)",
        color=PALETTE["text"], fontsize=14, fontweight='bold', y=0.98
    )

    plot_path = f"examples/plots/tpu_benchmark_{timestamp}.png"
    plt.savefig(plot_path, dpi=180, bbox_inches='tight', facecolor=PALETTE["bg"])
    plt.close()
    print(f"  🖼  Multi-panel benchmark plot saved → {plot_path}")
    print_header("SCALING COMPLETE")

if __name__ == "__main__":
    main()

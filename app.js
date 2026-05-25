/*
 * JAX Quantum Research Suite — Interactive JS Controller & Simulator Engine
 */

document.addEventListener('DOMContentLoaded', () => {
  // Initialize Background Animation
  initCanvasAnimation();

  // Initialize Interactive Directory Tree
  initDirectoryTree();

  // Initialize Interactive Framework Comparison
  initFrameworkCompare();

  // Initialize Interactive 3-Qubit Quantum Simulator
  initQuantumSimulator();

  // Initialize Interactive 3D Bloch Sphere
  init3DBlochSphere();

  // Initialize Clipboard Copying
  initClipboardCopier();

  // Initialize Lightbox Visual Gallery
  initLightboxGallery();

  // Highlight current nav links on scroll
  initScrollSpy();

  // Initialize Scroll-Reveal animations
  initScrollReveal();

  // Initialize 3D Interactive Card Tilt
  init3DCardTilt();
});

/* ==========================================
   1. Quantum Canvas Waveform Animation
   ========================================== */
function initCanvasAnimation() {
  const canvas = document.getElementById('canvas-bg');
  if (!canvas) return;
  
  const ctx = canvas.getContext('2d');
  let width = canvas.width = window.innerWidth;
  let height = canvas.height = window.innerHeight;

  window.addEventListener('resize', () => {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  });

  const particles = [];
  const particleCount = 35; // increased density for better interaction
  const maxDistance = 220;

  // Create random quantum particle nodes
  for (let i = 0; i < particleCount; i++) {
    particles.push({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.45,
      vy: (Math.random() - 0.5) * 0.45,
      radius: Math.random() * 3.2 + 1.2, // slightly larger particles for parallax visibility
      phase: Math.random() * Math.PI * 2,
      phaseSpeed: Math.random() * 0.02 + 0.005
    });
  }

  let mouse = { x: null, y: null, active: false };
  window.addEventListener('mousemove', (e) => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
    mouse.active = true;
  });
  window.addEventListener('mouseleave', () => {
    mouse.active = false;
  });

  // Track scrolling speeds for downward parallax drift
  let lastScrollY = window.scrollY;
  let scrollVelocity = 0;
  window.addEventListener('scroll', () => {
    const currentScrollY = window.scrollY;
    scrollVelocity = currentScrollY - lastScrollY;
    lastScrollY = currentScrollY;
  });

  // Shockwave blast on mouse click! Pushes particles away
  window.addEventListener('click', (e) => {
    const blastX = e.clientX;
    const blastY = e.clientY;
    
    particles.forEach(p => {
      const dx = p.x - blastX;
      const dy = p.y - blastY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      if (dist < 320) {
        const force = (320 - dist) * 0.12;
        const angle = Math.atan2(dy, dx);
        p.vx += Math.cos(angle) * force * 0.5;
        p.vy += Math.sin(angle) * force * 0.5;
      }
    });
  });

  function animate() {
    ctx.clearRect(0, 0, width, height);

    // Apply damping scroll velocity
    scrollVelocity *= 0.94; 
    const verticalDrift = scrollVelocity * 0.38;

    // Draw background wave interference fields
    const time = Date.now() * 0.0008;
    ctx.strokeStyle = 'rgba(42, 161, 152, 0.02)';
    ctx.lineWidth = 1;
    
    // Wave 1
    ctx.beginPath();
    for (let x = 0; x < width; x += 15) {
      const y = height * 0.45 + Math.sin(x * 0.0035 + time) * 70 + Math.cos(x * 0.0015 - time * 0.5) * 35;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Wave 2
    ctx.strokeStyle = 'rgba(108, 113, 196, 0.015)';
    ctx.beginPath();
    for (let x = 0; x < width; x += 15) {
      const y = height * 0.55 + Math.sin(x * 0.0025 - time * 0.8) * 85 + Math.cos(x * 0.004 + time * 0.4) * 25;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Animate and draw particle nodes
    particles.forEach(p => {
      // Natural slow drift
      p.x += p.vx;
      p.y += p.vy;

      // Scroll-driven downward drift with 3D parallax scaling
      p.y += verticalDrift * (p.radius * 0.35 + 0.55);

      // Dampen velocity increases from click shockwaves back to normal
      p.vx *= 0.98;
      p.vy *= 0.98;
      
      p.phase += p.phaseSpeed;

      // Wrap-around screen bounds with padding
      if (p.x < -10) p.x = width + 10;
      if (p.x > width + 10) p.x = -10;
      if (p.y < -10) p.y = height + 10;
      if (p.y > height + 10) p.y = -10;

      // Pulsing radius (mimics quantum state uncertainty)
      const currentRadius = p.radius + Math.sin(p.phase) * 0.6;

      ctx.fillStyle = 'rgba(7, 54, 66, 0.09)';
      ctx.beginPath();
      ctx.arc(p.x, p.y, currentRadius, 0, Math.PI * 2);
      ctx.fill();

      // Interconnect nodes inside threshold distance
      for (let j = 0; j < particles.length; j++) {
        const p2 = particles[j];
        if (p === p2) continue;
        const dx = p.x - p2.x;
        const dy = p.y - p2.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < maxDistance) {
          const alpha = (1 - dist / maxDistance) * 0.095;
          ctx.strokeStyle = `rgba(0, 43, 54, ${alpha})`;
          ctx.lineWidth = 0.55;
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
        }
      }

      // 3D Push & Connected net threads to cursor mouse pointer
      if (mouse.active) {
        const mdx = mouse.x - p.x;
        const mdy = mouse.y - p.y;
        const mdist = Math.sqrt(mdx * mdx + mdy * mdy);

        // Repel force if mouse is very close
        if (mdist < 130) {
          const force = (130 - mdist) * 0.035;
          const angle = Math.atan2(mdy, mdx);
          p.x -= Math.cos(angle) * force;
          p.y -= Math.sin(angle) * force;
        }

        // Draw glowing threads from pointer to nodes
        if (mdist < 160) {
          const alpha = (1 - mdist / 160) * 0.16;
          ctx.strokeStyle = `rgba(38, 139, 210, ${alpha})`;
          ctx.lineWidth = 0.75;
          ctx.beginPath();
          ctx.moveTo(mouse.x, mouse.y);
          ctx.lineTo(p.x, p.y);
          ctx.stroke();
        }
      }
    });

    requestAnimationFrame(animate);
  }

  animate();
}

/* ==========================================
   2. Interactive Directory Explorer
   ========================================== */
const directoryData = {
  "root": {
    type: "folder",
    desc: "Main Workspace Directory. Contains acceleration branches for CUDA GPUs and Cloud TPU meshes.",
    files: ["README.md", "requirements.txt", "run_all.py", "install_jax_gpu.sh"]
  },
  "gpu": {
    type: "folder",
    desc: "Modular GPU implementation division. Optimized for quick iterative testing, research-level auto-differentiation, and standard gates.",
    files: ["run_gpu.sh"]
  },
  "gpu/jax_qsim": {
    type: "folder",
    desc: "Primary quantum simulator library implemented in pure JAX tensor linear algebra ops.",
    files: ["core.py", "ops.py", "observables.py", "noise.py"]
  },
  "gpu/quantum_research": {
    type: "folder",
    desc: "8 physics-focused execution scripts. Exploring VQE, QAOA, Barren Plateaus, state preparations, and stochastic Lindbladian MC trajectories.",
    files: ["variational_quantum_classifier_xor.py", "variational_quantum_eigensolver_h2.py", "barren_plateau_gradient_vanishing.py", "quantum_noise_simulation_monte_carlo.py"]
  },
  "tpu": {
    type: "folder",
    desc: "Distributed scale TPU division. Utilizes compiler optimizations and mesh positional sharding to cross hardware limits.",
    files: ["tpu_quantum_scale.py", "run_tpu.sh"]
  },
  "grover_simulation": {
    type: "folder",
    desc: "High-dimensional Grover search benchmarks and Matrix Product State (MPS) tensor network approximation limits.",
    files: ["20qubits.py", "30qubits.py", "36qubits.py", "fullstatevector20qubits.py"]
  },
  "README.md": {
    type: "file",
    desc: "Primary documentation. Contains exhaustive theoretical frameworks, comparative matrix tables, mathematical equations, and benchmarks."
  },
  "requirements.txt": {
    type: "file",
    desc: "Python library dependency requirements. Pins JAX, CUDA references, Pytest, and Matplotlib."
  },
  "install_jax_gpu.sh": {
    type: "file",
    desc: "Shell helper automation script. Direct installations of matching CUDA toolkit JAX packages."
  },
  "run_all.py": {
    type: "file",
    desc: "GPU suite controller. Sequentially executes all modular quantum_research scripts and updates stored results."
  },
  "gpu/run_gpu.sh": {
    type: "file",
    desc: "Interactive GPU shell menu. Quickly synchronize, verify WSL2 environments, and launch designated quantum experiments."
  },
  "gpu/jax_qsim/core.py": {
    type: "file",
    desc: "Core state contraction engine. Uses highly optimized `jnp.tensordot` and transpose operations to model gate applies."
  },
  "gpu/jax_qsim/ops.py": {
    type: "file",
    desc: "Standard unitary and gate operators. Contains matrix transformations for H, X, Y, Z, CNOT, RX, RY, and RZ gates."
  },
  "gpu/jax_qsim/observables.py": {
    type: "file",
    desc: "Quantum measurement operations. Translates Pauli-string observables, evaluates exact expectation values, and runs vector-spaced state samplings."
  },
  "gpu/jax_qsim/noise.py": {
    type: "file",
    desc: "Stochastic Lindblad noise channel helper. Implements depolarization, amplitude damping, and customized jump probabilities."
  },
  "tpu/tpu_quantum_scale.py": {
    type: "file",
    desc: "Distributed Scale Engine VM runner. Merges 8 large scale sharded experiments under compiler optimizations like lax.fori_loop."
  },
  "tpu/run_tpu.sh": {
    type: "file",
    desc: "Multi-worker automated TPU synchronization and run controller. Pulls high resolution plots and packs output CSV logs."
  },
  "grover_simulation/36qubits.py": {
    type: "file",
    desc: "Extreme scale Grover search simulator. Executes 205,887 oracle-diffusion matrix iterations using PositionalSharding over 64 chips."
  }
};

function initDirectoryTree() {
  const treeNodes = document.querySelectorAll('.tree-node');
  const detailTitle = document.getElementById('file-title');
  const detailSubtitle = document.getElementById('file-subtitle');
  const detailBody = document.getElementById('file-desc');

  if (!detailTitle || !detailBody) return;

  treeNodes.forEach(node => {
    const label = node.querySelector('.node-label');
    if (!label) return;

    label.addEventListener('click', (e) => {
      e.stopPropagation();

      // Set active indicator
      document.querySelectorAll('.node-label').forEach(l => l.classList.remove('active'));
      label.classList.add('active');

      const path = label.getAttribute('data-path');
      const details = directoryData[path];

      if (details) {
        detailTitle.textContent = path.split('/').pop();
        detailSubtitle.textContent = details.type === 'folder' ? 'DIRECTORY COMPONENT' : 'FILE COMPONENT';
        detailBody.innerHTML = `<p>${details.desc}</p>${details.files ? `<strong>Contents:</strong><ul style="margin-left: 1.5rem; margin-top: 0.5rem;">${details.files.map(f => `<li>${f}</li>`).join('')}</ul>` : ''}`;
      }

      // Handle folder expand
      const children = node.querySelector('.tree-children');
      const folderIcon = label.querySelector('.folder-icon i');
      if (children) {
        children.classList.toggle('open');
        if (children.classList.contains('open')) {
          folderIcon.className = 'fas fa-folder-open';
        } else {
          folderIcon.className = 'fas fa-folder';
        }
      }
    });
  });

  // Trigger initial click on root directory
  const rootLabel = document.querySelector('.tree-node.tree-root > .node-label');
  if (rootLabel) rootLabel.click();
}

/* ==========================================
   3. Interactive Framework Comparison Cards
   ========================================== */
const compareData = {
  jax: {
    title: "🟢 JAX (This Simulator)",
    tpu: "Full XLA support natively",
    sharding: "Native PositionalSharding (64-chip mesh)",
    jit: "Native jax.jit, 100% caching speedups",
    autodiff: "Reverse-mode jax.grad in a single pass",
    loops: "jax.lax.fori_loop (O(1) graph size)",
    speed: "0.008ms (10-qubit gate, cached)",
    bullets: [
      "No classical framework overhead. Compiles directly into hardware HLO.",
      "Vectorized circuit batching is free via native <code>jax.vmap</code> wrapper.",
      "Integrates with gradient memory rematerialization to support massive sizes."
    ],
    pros: "Exceptional speed, scale, native multi-device sharding, reverse-mode derivatives.",
    cons: "Requires understanding pure-functional programming principles."
  },
  pennylane: {
    title: "🔵 PennyLane (Xanadu)",
    tpu: "No native TPU backend support",
    sharding: "Not supported",
    jit: "Device-dependent, high overhead",
    autodiff: "Parameter-shift rule (2 evaluations per param)",
    loops: "Not supported (Unrolled loops blow up compiler)",
    speed: "1.5ms per gate evaluation",
    bullets: [
      "Rich library layer, but relies on heavy OOP wrappers and CPU-bound DAG dispatch.",
      "Each variational step requires double the evaluations, causing memory walls.",
      "Struggles at 20+ qubits on distributed TPU nodes."
    ],
    pros: "Vast algorithm and hardware integration library, easy prototyping.",
    cons: "Massive speed limitations and OOM compile bounds for deep circuits at scale."
  },
  qiskit: {
    title: "🟡 Qiskit (IBM)",
    tpu: "Zero TPU integration support",
    sharding: "Not supported (requires IBM-cloud runtime)",
    jit: "Not supported (C++ compiled Aer simulator is separate)",
    autodiff: "None natively (uses finite difference)",
    loops: "Not supported",
    speed: "0.3ms (standard CPU Aer)",
    bullets: [
      "Imperative state management (QuantumCircuit objects) cannot compose with JIT.",
      "No auto-differentiation, highly complex backpropagation setups.",
      "Designed for IBM hardware dispatch, not differentiable machine learning research."
    ],
    pros: "Industry standard for programming physical superconducting quantum hardware.",
    cons: "Opaque simulations, lacks multi-device gradient training pipelines."
  },
  tensorflow: {
    title: "🔴 TensorFlow Quantum",
    tpu: "Only classical post-processing on TPU",
    sharding: "Not supported",
    jit: "TF Graph compile, but Cirq simulation is CPU-bound",
    autodiff: "TF GradTape, but quantum part is non-differentiable",
    loops: "Not supported",
    speed: "20ms–45ms per gradient step",
    bullets: [
      "Wraps Cirq simulations which execute strictly CPU-bound. Low TPU utility.",
      "Massive graph compilation times during deep random circuit unrolls.",
      "Stateful paradigm limits distributed sharding and parameter safety."
    ],
    pros: "Native Keras integration for small-scale hybrid structures.",
    cons: "Discontinued active expansion, lacks native XLA device gate acceleration."
  },
  pytorch: {
    title: "🟠 PyTorch",
    tpu: "Requires high-latency torch_xla bridge",
    sharding: "Not supported natively",
    jit: "torch.compile Dynamo (partial loop fusion)",
    autodiff: "Autograd backpropagation engine",
    loops: "Not supported natively",
    speed: "12ms per gradient step",
    bullets: [
      "Excellent classical ML engine, but lacks native quantum linear contractions.",
      "torch_xla bridge introduces massive data transfer latencies.",
      "Lacks unified functional loops equivalent to jax.lax.fori_loop."
    ],
    pros: "Very active classical deep learning library support.",
    cons: "Suboptimal accelerator compilation paths, memory overhead scales linearly."
  },
  numpy: {
    title: "🔵 NumPy",
    tpu: "No hardware acceleration support",
    sharding: "Not supported",
    jit: "Not supported",
    autodiff: "None (Finite difference approximations only)",
    loops: "Not supported",
    speed: "0.8ms (Single CPU core)",
    bullets: [
      "Purely classical CPU computation limit. All threads are serialized.",
      "A 25-qubit contraction takes minutes compared to JAX's milliseconds.",
      "No analytical backprop gradients, extremely expensive scale evaluations."
    ],
    pros: "Zero dependencies, clean matrix manipulation baseline.",
    cons: "CPU-only memory walls make simulations above 24 qubits physically impossible."
  }
};

function initFrameworkCompare() {
  const tabs = document.querySelectorAll('.compare-tab');
  
  const titleEl = document.getElementById('comp-title');
  const tpuVal = document.getElementById('comp-tpu');
  const shardingVal = document.getElementById('comp-sharding');
  const jitVal = document.getElementById('comp-jit');
  const autodiffVal = document.getElementById('comp-autodiff');
  const loopsVal = document.getElementById('comp-loops');
  const speedVal = document.getElementById('comp-speed');
  const bulletsEl = document.getElementById('comp-bullets');
  const prosEl = document.getElementById('comp-pros');
  const consEl = document.getElementById('comp-cons');

  if (!titleEl) return;

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      const fw = tab.getAttribute('data-fw');
      const data = compareData[fw];

      if (data) {
        // Update Title & Core Specs
        titleEl.textContent = data.title;
        tpuVal.textContent = data.tpu;
        shardingVal.textContent = data.sharding;
        jitVal.textContent = data.jit;
        autodiffVal.textContent = data.autodiff;
        loopsVal.textContent = data.loops;
        speedVal.textContent = data.speed;

        // Apply Yes/No styling color classes
        [tpuVal, shardingVal, jitVal, autodiffVal, loopsVal].forEach(el => {
          el.className = 'compare-score-val';
          const txt = el.textContent.toLowerCase();
          if (txt.includes('full') || txt.includes('native') || txt.includes('yes')) {
            el.classList.add('yes');
          } else if (txt.includes('no') || txt.includes('not')) {
            el.classList.add('no');
          } else {
            el.classList.add('partial');
          }
        });

        // Update bullets and Pros/Cons
        bulletsEl.innerHTML = data.bullets.map(b => `<li>${b}</li>`).join('');
        prosEl.innerHTML = `<strong>Pros:</strong> ${data.pros}`;
        consEl.innerHTML = `<strong>Cons:</strong> ${data.cons}`;
      }
    });
  });

  // Click initial JAX tab
  const jaxTab = document.querySelector('.compare-tab[data-fw="jax"]');
  if (jaxTab) jaxTab.click();
}

/* ==========================================
   4. Live 3-Qubit Quantum Simulator Widget
   ========================================== */
function initQuantumSimulator() {
  // Slots on our circuit wires. Index 0: Qubit 0, Index 1: Qubit 1, Index 2: Qubit 2
  // We allow up to 4 gates on each wire
  const gateGrid = [
    [null, null, null, null],
    [null, null, null, null],
    [null, null, null, null]
  ];

  let selectedGate = 'h'; // Default gate selection

  const slots = document.querySelectorAll('.circuit-gate-slot');
  const gateButtons = document.querySelectorAll('.btn-gate');
  const resetButton = document.querySelector('.btn-reset');

  if (!slots.length) return;

  // Gate type selection
  gateButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      gateButtons.forEach(b => b.style.backgroundColor = 'var(--bg-primary)');
      
      // Style active button
      let colorClass = '';
      if (btn.classList.contains('h')) {
        selectedGate = 'h';
        btn.style.backgroundColor = 'rgba(38, 139, 210, 0.15)';
      } else if (btn.classList.contains('x')) {
        selectedGate = 'x';
        btn.style.backgroundColor = 'rgba(203, 75, 22, 0.15)';
      } else if (btn.classList.contains('cnot')) {
        selectedGate = 'cnot';
        btn.style.backgroundColor = 'rgba(108, 113, 196, 0.15)';
      }
    });
  });

  // Set initial selected gate button color
  const initialBtn = document.querySelector('.btn-gate.h');
  if (initialBtn) initialBtn.style.backgroundColor = 'rgba(38, 139, 210, 0.15)';

  // Handle slot clicking (Toggle gate in slot)
  slots.forEach(slot => {
    slot.addEventListener('click', () => {
      const qubit = parseInt(slot.getAttribute('data-qubit'));
      const stage = parseInt(slot.getAttribute('data-stage'));

      if (gateGrid[qubit][stage] === null) {
        // Place selected gate
        gateGrid[qubit][stage] = selectedGate;
        slot.textContent = selectedGate.toUpperCase();
        slot.className = `circuit-gate-slot filled ${selectedGate}`;
      } else {
        // Remove gate
        gateGrid[qubit][stage] = null;
        slot.textContent = '';
        slot.className = 'circuit-gate-slot';
      }

      runSimulation(gateGrid);
    });
  });

  // Reset circuit
  if (resetButton) {
    resetButton.addEventListener('click', () => {
      for (let q = 0; q < 3; q++) {
        for (let s = 0; s < 4; s++) {
          gateGrid[q][s] = null;
        }
      }
      slots.forEach(slot => {
        slot.textContent = '';
        slot.className = 'circuit-gate-slot';
      });
      runSimulation(gateGrid);
    });
  }

  // Initial simulation pass (displays all probability on |000>)
  runSimulation(gateGrid);
}

// 3-Qubit quantum mathematics engine
function runSimulation(grid) {
  // 8 dimensional state vector (real coefficients only as H, X, CNOT don't produce complex phases here)
  let state = new Float32Array(8);
  state[0] = 1.0; // Start at |000>

  // Execute stages chronologically (columns 0 to 3)
  for (let s = 0; s < 4; s++) {
    // Collect gates at this stage
    for (let q = 0; q < 3; q++) {
      const gate = grid[q][s];
      if (!gate) continue;

      if (gate === 'h') {
        state = applyHadamard(state, q);
      } else if (gate === 'x') {
        state = applyPauliX(state, q);
      } else if (gate === 'cnot') {
        // For CNOT, we need to know the control and target.
        // Let's assume:
        // - If qubit 0 CNOT: qubit 0 is control, qubit 1 is target.
        // - If qubit 1 CNOT: qubit 1 is control, qubit 2 is target.
        // - If qubit 2 CNOT: qubit 2 is control, qubit 0 is target.
        const control = q;
        const target = (q + 1) % 3;
        state = applyCNOT(state, control, target);
      }
    }
  }

  // Calculate probabilities
  const probabilities = Array.from(state).map(c => c * c);

  // Render probabilities to bar charts
  for (let i = 0; i < 8; i++) {
    const label = i.toString(2).padStart(3, '0');
    const bar = document.getElementById(`bar-${label}`);
    const valText = document.getElementById(`val-${label}`);
    if (bar && valText) {
      const pct = (probabilities[i] * 100).toFixed(1);
      bar.style.width = `${pct}%`;
      valText.textContent = `${pct}%`;
      
      // Color bars based on dominant state
      if (probabilities[i] > 0.05) {
        bar.className = 'chart-bar active';
      } else {
        bar.className = 'chart-bar';
      }
    }
  }

  // --- Calculate Bloch Vector for Qubit 0 ---
  // Find effective wave amplitude and phase sign for subspace |0>_0 and |1>_0
  let norm0 = 0;
  let dominantVal0 = 0;
  for (let i = 0; i < 4; i++) {
    norm0 += state[i] * state[i];
    if (Math.abs(state[i]) > Math.abs(dominantVal0)) {
      dominantVal0 = state[i];
    }
  }
  const C0 = Math.sign(dominantVal0) * Math.sqrt(norm0);

  let norm1 = 0;
  let dominantVal1 = 0;
  for (let i = 4; i < 8; i++) {
    norm1 += state[i] * state[i];
    if (Math.abs(state[i]) > Math.abs(dominantVal1)) {
      dominantVal1 = state[i];
    }
  }
  const C1 = Math.sign(dominantVal1) * Math.sqrt(norm1);

  // Calculate Bloch Sphere coordinates
  const bx = 2 * C0 * C1;
  const by = 0.0; // Purely real amplitude space
  const bz = C0 * C0 - C1 * C1;

  // Trigger 3D Bloch sphere vector update
  updateBlochSphereVector(bx, by, bz);
}

// Linear algebra gate transformations
function applyHadamard(state, qubit) {
  const nextState = new Float32Array(8);
  const invSqrt2 = 1 / Math.sqrt(2);
  const mask = 1 << (2 - qubit); // Bit position matching qubit index

  for (let i = 0; i < 8; i++) {
    const bit = (i & mask) !== 0;
    const partner = i ^ mask;

    if (!bit) {
      nextState[i] = (state[i] + state[partner]) * invSqrt2;
    } else {
      nextState[i] = (state[partner] - state[i]) * invSqrt2;
    }
  }
  return nextState;
}

function applyPauliX(state, qubit) {
  const nextState = new Float32Array(8);
  const mask = 1 << (2 - qubit);

  for (let i = 0; i < 8; i++) {
    const partner = i ^ mask;
    nextState[i] = state[partner];
  }
  return nextState;
}

function applyCNOT(state, control, target) {
  const nextState = new Float32Array(8);
  const ctrlMask = 1 << (2 - control);
  const tgtMask = 1 << (2 - target);

  for (let i = 0; i < 8; i++) {
    const ctrlActive = (i & ctrlMask) !== 0;
    if (ctrlActive) {
      const partner = i ^ tgtMask;
      nextState[i] = state[partner];
    } else {
      nextState[i] = state[i];
    }
  }
  return nextState;
}

/* ==========================================
   5. Clipboard Copy Utility
   ========================================== */
function initClipboardCopier() {
  const buttons = document.querySelectorAll('.code-btn-copy');

  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      const container = btn.closest('.code-container');
      const codeElement = container.querySelector('code');
      if (!codeElement) return;

      // Extract original code text (stripping visual code element spans if necessary)
      const textToCopy = codeElement.innerText;

      navigator.clipboard.writeText(textToCopy).then(() => {
        const originalText = btn.innerHTML;
        btn.innerHTML = '<i class="fas fa-check" style="color: var(--solar-green);"></i> Copied!';
        btn.style.borderColor = 'var(--solar-green)';
        
        setTimeout(() => {
          btn.innerHTML = originalText;
          btn.style.borderColor = 'rgba(253, 246, 227, 0.15)';
        }, 1800);
      }).catch(err => {
        console.error('Failed to copy text: ', err);
      });
    });
  });
}

/* ==========================================
   6. Visual Gallery Filters & Lightbox Slider
   ========================================== */
function initLightboxGallery() {
  const filterBtns = document.querySelectorAll('.gallery-filter-btn');
  const cards = document.querySelectorAll('.gallery-card');
  const wrappers = document.querySelectorAll('.gallery-img-wrapper');
  
  const lightbox = document.createElement('div');
  lightbox.className = 'lightbox';
  lightbox.innerHTML = `
    <span class="lightbox-close">&times;</span>
    <img class="lightbox-content" src="" alt="Expanded View">
  `;
  document.body.appendChild(lightbox);

  const lightboxImg = lightbox.querySelector('.lightbox-content');
  const lightboxClose = lightbox.querySelector('.lightbox-close');

  // Filter functionality
  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const filter = btn.getAttribute('data-filter');

      cards.forEach(card => {
        const category = card.getAttribute('data-category');
        if (filter === 'all' || category === filter) {
          card.style.display = 'flex';
        } else {
          card.style.display = 'none';
        }
      });
    });
  });

  // Expand image on click (Lightbox)
  wrappers.forEach(wrapper => {
    wrapper.addEventListener('click', () => {
      const img = wrapper.querySelector('img');
      if (!img) return;

      lightboxImg.src = img.src;
      lightboxImg.alt = img.alt;
      lightbox.style.display = 'flex';
    });
  });

  // Close lightbox
  lightbox.addEventListener('click', (e) => {
    if (e.target !== lightboxImg) {
      lightbox.style.display = 'none';
    }
  });

  lightboxClose.addEventListener('click', () => {
    lightbox.style.display = 'none';
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && lightbox.style.display === 'flex') {
      lightbox.style.display = 'none';
    }
  });
}

/* ==========================================
   7. Navigation ScrollSpy Toggles
   ========================================== */
function initScrollSpy() {
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-links .nav-item');

  window.addEventListener('scroll', () => {
    let currentId = '';
    const scrollPos = window.scrollY + 120; // Offset for triggers

    sections.forEach(sec => {
      const top = sec.offsetTop;
      const height = sec.offsetHeight;
      if (scrollPos >= top && scrollPos < top + height) {
        currentId = sec.getAttribute('id');
      }
    });

    if (currentId) {
      navLinks.forEach(link => {
        link.classList.remove('active');
        const anchor = link.querySelector('a');
        if (anchor && anchor.getAttribute('href') === `#${currentId}`) {
          link.classList.add('active');
        }
      });
    }
  });
}

/* ==========================================
   8. Scroll-Reveal IntersectionObserver Hook
   ========================================== */
function initScrollReveal() {
  const revealElements = document.querySelectorAll('.reveal');
  
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('active');
        observer.unobserve(entry.target); // Reveal once
      }
    });
  }, {
    threshold: 0.08,
    rootMargin: '0px 0px -40px 0px'
  });

  revealElements.forEach((el, index) => {
    // Add minor staggered delays dynamically for children within grids
    const isGridChild = el.closest('.stats-grid') || el.closest('.gallery-grid');
    if (isGridChild) {
      const children = Array.from(isGridChild.children);
      const childIndex = children.indexOf(el);
      if (childIndex !== -1) {
        el.classList.add(`delay-${(childIndex % 4) * 100}`);
      }
    }
    observer.observe(el);
  });
}

/* ==========================================
   9. 3D Card Interactive Tilt Effect
   ========================================== */
function init3DCardTilt() {
  const cards = document.querySelectorAll('[data-tilt]');

  cards.forEach(card => {
    card.addEventListener('mousemove', (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      const xc = rect.width / 2;
      const yc = rect.height / 2;
      
      // Calculate rotation angles (max 7 degrees tilt for smooth performance)
      const rotateY = ((x - xc) / xc) * 7;
      const rotateX = -((y - yc) / yc) * 7;
      
      card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.015, 1.015, 1.015)`;
    });

    card.addEventListener('mouseleave', () => {
      card.style.transform = `perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)`;
    });
  });
}

/* ==========================================
   10. Interactive 3D Bloch Sphere Canvas Engine
   ========================================== */
let blochPitch = 0.45;
let blochYaw = 0.75;
let blochVector = { x: 1.0, y: 0.0, z: 0.0 }; // Start at state |0> (z=1) but projected
let blochCanvas = null;
let blochCtx = null;

function init3DBlochSphere() {
  blochCanvas = document.getElementById('canvas-3d-bloch');
  if (!blochCanvas) return;

  blochCtx = blochCanvas.getContext('2d');
  
  let isDragging = false;
  let prevMouseX = 0;
  let prevMouseY = 0;

  // Mouse Drag Interactions
  blochCanvas.addEventListener('mousedown', (e) => {
    isDragging = true;
    prevMouseX = e.clientX;
    prevMouseY = e.clientY;
    blochCanvas.style.cursor = 'grabbing';
  });

  window.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    const deltaX = e.clientX - prevMouseX;
    const deltaY = e.clientY - prevMouseY;
    
    blochYaw += deltaX * 0.007;
    blochPitch -= deltaY * 0.007;
    
    // Clamp pitch to avoid gimbal lock/complete inversion
    blochPitch = Math.max(-Math.PI/2 + 0.02, Math.min(Math.PI/2 - 0.02, blochPitch));

    prevMouseX = e.clientX;
    prevMouseY = e.clientY;

    drawBlochSphere();
  });

  window.addEventListener('mouseup', () => {
    if (isDragging) {
      isDragging = false;
      blochCanvas.style.cursor = 'grab';
    }
  });

  // Touch Support
  blochCanvas.addEventListener('touchstart', (e) => {
    if (e.touches.length === 1) {
      isDragging = true;
      prevMouseX = e.touches[0].clientX;
      prevMouseY = e.touches[0].clientY;
    }
  });

  window.addEventListener('touchmove', (e) => {
    if (!isDragging || e.touches.length !== 1) return;
    const deltaX = e.touches[0].clientX - prevMouseX;
    const deltaY = e.touches[0].clientY - prevMouseY;

    blochYaw += deltaX * 0.007;
    blochPitch -= deltaY * 0.007;
    blochPitch = Math.max(-Math.PI/2 + 0.02, Math.min(Math.PI/2 - 0.02, blochPitch));

    prevMouseX = e.touches[0].clientX;
    prevMouseY = e.touches[0].clientY;
    drawBlochSphere();
  });

  window.addEventListener('touchcancel', () => { isDragging = false; });
  window.addEventListener('touchend', () => { isDragging = false; });

  // slow idle spin rotation
  let lastTime = Date.now();
  function autoRotate() {
    if (!isDragging) {
      const now = Date.now();
      const dt = (now - lastTime) * 0.001;
      blochYaw += dt * 0.06; // Slow rotate
      drawBlochSphere();
    }
    lastTime = Date.now();
    requestAnimationFrame(autoRotate);
  }

  drawBlochSphere();
  autoRotate();
}

function updateBlochSphereVector(x, y, z) {
  const targetX = x;
  const targetY = y;
  const targetZ = z;

  let steps = 0;
  const maxSteps = 24;
  
  function stepAnimation() {
    if (steps >= maxSteps) {
      blochVector.x = targetX;
      blochVector.y = targetY;
      blochVector.z = targetZ;
      drawBlochSphere();
      return;
    }
    // Smooth linear interpolation curves
    blochVector.x += (targetX - blochVector.x) * 0.16;
    blochVector.y += (targetY - blochVector.y) * 0.16;
    blochVector.z += (targetZ - blochVector.z) * 0.16;
    drawBlochSphere();
    steps++;
    requestAnimationFrame(stepAnimation);
  }
  
  stepAnimation();
}

function drawBlochSphere() {
  if (!blochCanvas || !blochCtx) return;

  const w = blochCanvas.width;
  const h = blochCanvas.height;
  blochCtx.clearRect(0, 0, w, h);

  const cx = w / 2;
  const cy = h / 2;
  const r = 85; // Sphere wireframe projection radius

  // 3D coordinate system projection formula
  function project(x, y, z) {
    // 1. Yaw rotation (around Z-Axis)
    const x1 = x * Math.cos(blochYaw) - y * Math.sin(blochYaw);
    const y1 = x * Math.sin(blochYaw) + y * Math.cos(blochYaw);
    const z1 = z;

    // 2. Pitch rotation (around X-Axis)
    const x2 = x1;
    const y2 = y1 * Math.cos(blochPitch) - z1 * Math.sin(blochPitch);
    const z2 = y1 * Math.sin(blochPitch) + z1 * Math.cos(blochPitch);

    return {
      x: cx + x2 * r,
      y: cy - z2 * r,
      zDepth: y2
    };
  }

  // Draw background circle shell
  blochCtx.strokeStyle = 'rgba(0, 43, 54, 0.08)';
  blochCtx.lineWidth = 1;
  blochCtx.beginPath();
  blochCtx.arc(cx, cy, r, 0, Math.PI * 2);
  blochCtx.stroke();

  // Draw wireframe latitude equator circle (Z = 0)
  blochCtx.strokeStyle = 'rgba(0, 43, 54, 0.15)';
  blochCtx.beginPath();
  for (let theta = 0; theta <= Math.PI * 2 + 0.06; theta += 0.06) {
    const pt = project(Math.cos(theta), Math.sin(theta), 0);
    if (theta === 0) blochCtx.moveTo(pt.x, pt.y);
    else blochCtx.lineTo(pt.x, pt.y);
  }
  blochCtx.stroke();

  // Draw X-Z Meridian circle (Y = 0)
  blochCtx.strokeStyle = 'rgba(0, 43, 54, 0.07)';
  blochCtx.beginPath();
  for (let theta = 0; theta <= Math.PI * 2 + 0.06; theta += 0.06) {
    const pt = project(Math.cos(theta), 0, Math.sin(theta));
    if (theta === 0) blochCtx.moveTo(pt.x, pt.y);
    else blochCtx.lineTo(pt.x, pt.y);
  }
  blochCtx.stroke();

  // Draw Y-Z Meridian circle (X = 0)
  blochCtx.beginPath();
  for (let theta = 0; theta <= Math.PI * 2 + 0.06; theta += 0.06) {
    const pt = project(0, Math.cos(theta), Math.sin(theta));
    if (theta === 0) blochCtx.moveTo(pt.x, pt.y);
    else blochCtx.lineTo(pt.x, pt.y);
  }
  blochCtx.stroke();

  // Calculate projected coordinate poles
  const o = project(0, 0, 0);
  const px = project(1.25, 0, 0);
  const py = project(0, 1.25, 0);
  const pz = project(0, 0, 1.25);
  const nx = project(-1.25, 0, 0);
  const ny = project(0, -1.25, 0);
  const nz = project(0, 0, -1.25);

  // X-Axis (Red/Orange)
  blochCtx.strokeStyle = 'rgba(203, 75, 22, 0.45)';
  blochCtx.lineWidth = 1;
  blochCtx.beginPath();
  blochCtx.moveTo(nx.x, nx.y);
  blochCtx.lineTo(px.x, px.y);
  blochCtx.stroke();
  
  // Y-Axis (Green)
  blochCtx.strokeStyle = 'rgba(133, 153, 0, 0.45)';
  blochCtx.beginPath();
  blochCtx.moveTo(ny.x, ny.y);
  blochCtx.lineTo(py.x, py.y);
  blochCtx.stroke();

  // Z-Axis (Blue/Purple)
  blochCtx.strokeStyle = 'rgba(38, 139, 210, 0.55)';
  blochCtx.lineWidth = 1.5;
  blochCtx.beginPath();
  blochCtx.moveTo(nz.x, nz.y);
  blochCtx.lineTo(pz.x, pz.y);
  blochCtx.stroke();

  // Write coordinate indicators
  blochCtx.font = 'bold 9px var(--font-display)';
  
  blochCtx.fillStyle = 'var(--solar-orange)';
  blochCtx.fillText('+x', px.x + 4, px.y + 3);
  
  blochCtx.fillStyle = 'var(--solar-green)';
  blochCtx.fillText('+y', py.x + 4, py.y + 3);

  blochCtx.fillStyle = 'var(--solar-blue)';
  blochCtx.fillText('|0⟩', pz.x - 5, pz.y - 7);
  blochCtx.fillText('|1⟩', nz.x - 5, nz.y + 11);

  // Draw State Vector Arrow pointer
  const vec = project(blochVector.x, blochVector.y, blochVector.z);
  
  // Base pivot point
  blochCtx.fillStyle = 'rgba(42, 161, 152, 0.35)';
  blochCtx.beginPath();
  blochCtx.arc(o.x, o.y, 3.5, 0, Math.PI * 2);
  blochCtx.fill();

  // Vector Line (glowing state vector)
  blochCtx.strokeStyle = 'var(--solar-blue)';
  blochCtx.lineWidth = 3.5;
  blochCtx.shadowColor = 'var(--solar-blue)';
  blochCtx.shadowBlur = 6;
  blochCtx.beginPath();
  blochCtx.moveTo(o.x, o.y);
  blochCtx.lineTo(vec.x, vec.y);
  blochCtx.stroke();
  
  blochCtx.shadowBlur = 0; // reset shadow

  // Vector Pointer tip
  blochCtx.fillStyle = 'var(--solar-blue)';
  blochCtx.beginPath();
  blochCtx.arc(vec.x, vec.y, 5, 0, Math.PI * 2);
  blochCtx.fill();

  // Write state symbol |ψ⟩
  blochCtx.fillStyle = '#000000';
  blochCtx.font = 'bold 11px var(--font-display)';
  blochCtx.fillText('|ψ⟩', vec.x + 6, vec.y - 4);
}


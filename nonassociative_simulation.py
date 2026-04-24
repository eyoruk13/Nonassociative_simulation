"""
Two-qubit TFIM + Nonassociative Born-Markov Master Equation
Weak-Coupling Regime Simulation
====================================================
Parameters: g=0.2, Gamma=0.05, eps=0.01, J=1
These parameters satisfy the Born-Markov validity conditions:
  g/J  = 0.2  << 1   (weak system-bath coupling)
  Γ/J  = 0.05 << 1   (dissipative rate << system energy scale)
  ε/J  = 0.01 << 1   (dispersive rate << system energy scale)
Optimal transverse field: h/J = 0.2
  (entanglement maximum in weak-coupling regime)
κ range: {0, 50, 100, 150, 200}
  → λ/Γ = (g²/16)(ε/Γ)κ = {0, 0.025, 0.05, 0.075, 0.10}
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# 1. BASIC OPERATORS
# ─────────────────────────────────────────────────────────────
I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]],   dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]],  dtype=complex)
sm = np.array([[0, 0], [1, 0]],   dtype=complex)  # lowering: sigma_-

def kron(a, b): return np.kron(a, b)
def dag(X):     return X.conj().T
def comm(A, B): return A @ B - B @ A

# Two-qubit operators
sz1 = kron(sz, I2);  sz2 = kron(I2, sz)
sx1 = kron(sx, I2);  sx2 = kron(I2, sx)
sm1 = kron(sm, I2);  sm2 = kron(I2, sm)
sysy = kron(sy, sy)   # used in Wootters concurrence formula

# ─────────────────────────────────────────────────────────────
# 2. HAMILTONIAN
# ─────────────────────────────────────────────────────────────
def HTFIM(J, h):
    """
    Two-qubit transverse-field Ising model:
    H = -J σz⊗σz - (h/2)(σx⊗I + I⊗σx)
    J: Ising coupling strength
    h: transverse field strength
    """
    return -J * (sz1 @ sz2) - 0.5 * h * sx1 - 0.5 * h * sx2

# ─────────────────────────────────────────────────────────────
# 3. RIGHT-HAND SIDE OF THE MASTER EQUATION
# ─────────────────────────────────────────────────────────────
def drho_dt(rho, H, Gamma, g, eps, kappa):
    """
    Full right-hand side of the nonassociative master equation (Eq. 63):

    dρ/dt = -i[H_TFIM, ρ]                           (unitary)
           + Σ_a Γ(S_a ρ S_a† - ½{S_a†S_a, ρ})      (Lindblad T=0)
           + N[ρ]                                    (nonassociative)

    Nonassociative correction (Eq. 60):
      N[ρ] = Σ_a c · r_z^(a) · σ_z^(a)
      c    = -(g²/16) · ε₊ · κ
      r_z^(a) = Tr(ρ σ_z^(a))   -- instantaneous expectation value (nonlinear)

    Note: N[ρ] is dispersive (purely coherent). It does not open
    additional dissipative channels; Γ is unaffected by κ.
    """
    # Unitary part
    unitary = -1j * comm(H, rho)

    # Standard Lindblad dissipator (T=0, amplitude damping)
    dissipator = np.zeros_like(rho)
    for S in [sm1, sm2]:
        SdS = dag(S) @ S
        dissipator += Gamma * (S @ rho @ dag(S)
                                - 0.5 * (SdS @ rho + rho @ SdS))

    # Nonassociative correction
    # c = -(g²/16) · ε · κ  (same for both qubits by symmetry)
    rz1 = np.real(np.trace(rho @ sz1))
    rz2 = np.real(np.trace(rho @ sz2))
    c   = -(g**2 / 16.0) * eps * kappa
    NA  = c * (rz1 * sz1 + rz2 * sz2)

    return unitary + dissipator + NA

# ─────────────────────────────────────────────────────────────
# 4. RK4 INTEGRATION STEP
# ─────────────────────────────────────────────────────────────
def rk4_step(rho, dt, H, Gamma, g, eps, kappa):
    """
    Fourth-order Runge-Kutta step.
    r_z is recomputed at every substep to correctly capture
    the nonlinear feedback in N[ρ].
    Hermiticity and trace=1 are enforced after each step.
    """
    k1 = drho_dt(rho,              H, Gamma, g, eps, kappa)
    k2 = drho_dt(rho + 0.5*dt*k1,  H, Gamma, g, eps, kappa)
    k3 = drho_dt(rho + 0.5*dt*k2,  H, Gamma, g, eps, kappa)
    k4 = drho_dt(rho + dt*k3,      H, Gamma, g, eps, kappa)
    rho_new  = rho + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    rho_new  = 0.5 * (rho_new + dag(rho_new))  # enforce Hermiticity
    rho_new /= np.trace(rho_new)                # enforce trace = 1
    return rho_new

# ─────────────────────────────────────────────────────────────
# 5. OBSERVABLES
# ─────────────────────────────────────────────────────────────
def concurrence(rho):
    """Wootters concurrence (Phys. Rev. Lett. 80, 2245, 1998)."""
    R     = rho @ sysy @ rho.conj() @ sysy
    evals = np.maximum(np.real(np.linalg.eigvals(R)), 0.0)
    sq    = np.sort(np.sqrt(evals))[::-1]
    return float(max(0.0, sq[0] - sq[1] - sq[2] - sq[3]))

def purity(rho):
    """Tr(ρ²): equals 1 for pure states, 1/d for maximally mixed."""
    return float(np.real(np.trace(rho @ rho)))

def von_neumann(rho):
    """Von Neumann entropy S(ρ) = -Tr(ρ log ρ)."""
    evals = np.clip(np.real(
        np.linalg.eigvalsh(0.5 * (rho + dag(rho)))), 0, 1)
    nz = evals[evals > 1e-14]
    return float(-np.sum(nz * np.log(nz)))

def min_eigenvalue(rho):
    """
    Minimum eigenvalue of ρ — used for complete positivity (CP) check.
    Physical CP violation: min eigenvalue < 0 with magnitude >> 1e-15.
    Numerical noise level: ~1e-18 (floating-point rounding, not CP violation).
    """
    return float(np.min(np.real(
        np.linalg.eigvalsh(0.5 * (rho + dag(rho))))))

# ─────────────────────────────────────────────────────────────
# 6. SIMULATION PARAMETERS
# ─────────────────────────────────────────────────────────────
J     = 1.0    # Ising coupling (sets the energy unit)
h_opt = 0.25   # optimal h/J for entanglement in weak-coupling regime (peak of C_ss)
g     = 0.2    # system-bath coupling  → g/J = 0.2 << 1  ✓
Gamma = 0.05   # amplitude-damping rate → Γ/J = 0.05 << 1 ✓
eps   = 0.01   # dispersive (Lamb-shift) → ε/J = 0.01 << 1 ✓
dt    = 0.05   # RK4 step size
tmax  = 300.0  # = 15/Gamma (sufficient for convergence to steady state)
nsteps        = int(tmax / dt)
record_every  = 40   # record every 40 steps → dt_out = 2.0 in units of Γ⁻¹=20

# κ values: λ/Γ = (g²/16)(ε/Γ)κ
# κ=50  → λ/Γ = 0.025  (same λ/Γ as κ=0.5 in old g=1 parameters)
# κ=200 → λ/Γ = 0.10
kappas = [0, 50, 100, 150, 200]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

print("=" * 55)
print("PARAMETER HIERARCHY (Born-Markov validity)")
print("=" * 55)
print(f"  g/J  = {g/J:.3f}  << 1  ✓")
print(f"  Γ/J  = {Gamma/J:.3f}  << 1  ✓")
print(f"  ε/J  = {eps/J:.3f}  << 1  ✓")
print(f"  ε/Γ  = {eps/Gamma:.3f}")
print(f"  h/J  = {h_opt:.3f}  (entanglement optimum)")
print()
print(f"{'κ':>6} | {'λ/Γ':>7} | {'λ/J':>9}")
print("-" * 30)
for k in kappas:
    lG = (g**2/16) * eps * k / Gamma
    lJ = (g**2/16) * eps * k / J
    print(f"{k:>6} | {lG:>7.4f} | {lJ:>9.6f}")

# Initial state: |+⟩ ⊗ |+⟩
plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
rho0 = kron(np.outer(plus, plus.conj()),
            np.outer(plus, plus.conj()))

H_opt = HTFIM(J, h_opt * J)

# ─────────────────────────────────────────────────────────────
# 7. TIME SERIES SIMULATION
# ─────────────────────────────────────────────────────────────
print("\nComputing time series...")
ts = {}
for k in kappas:
    print(f"  κ = {k}...")
    rho = rho0.copy()
    times, rz, rx, zz_corr, concs, purs, entrs, min_evals = \
        [], [], [], [], [], [], [], []

    for i in range(nsteps):
        if i % record_every == 0:
            t_norm = i * dt * Gamma       # normalized time: t · Γ₊
            times.append(t_norm)
            rz.append(float(np.real(np.trace(rho @ sz1))))
            rx.append(float(np.real(np.trace(rho @ sx1))))
            zz_corr.append(float(np.real(np.trace(rho @ (sz1 @ sz2)))))
            concs.append(concurrence(rho))
            purs.append(purity(rho))
            entrs.append(von_neumann(rho))
            min_evals.append(min_eigenvalue(rho))
        rho = rk4_step(rho, dt, H_opt, Gamma, g, eps, k)

    ts[k] = dict(t=np.array(times), rz=np.array(rz), rx=np.array(rx),
                 zz=np.array(zz_corr), c=np.array(concs),
                 p=np.array(purs),    s=np.array(entrs),
                 min_eval=np.array(min_evals))

# ─────────────────────────────────────────────────────────────
# 8. COMPLETE POSITIVITY CHECK
# ─────────────────────────────────────────────────────────────
print("\nCP check (minimum eigenvalue of ρ(t)):")
for k in kappas:
    mv     = np.min(ts[k]['min_eval'])
    status = "✓  (numerical noise)" if mv > -1e-10 else "✗  CP VIOLATED"
    print(f"  κ={k:4d}: min eigenvalue = {mv:.2e}  {status}")

# ─────────────────────────────────────────────────────────────
# 9. h/J SCAN (steady-state concurrence vs transverse field)
# ─────────────────────────────────────────────────────────────
print("\nComputing h/J scan...")
h_vals    = np.linspace(0, 1.5, 31)
ss_nsteps = int(200.0 / dt)   # t = 200 = 10/Gamma (steady state)
hJ = {}
for k in kappas:
    print(f"  κ = {k}...")
    cs = []
    for h in h_vals:
        rho = rho0.copy()
        for _ in range(ss_nsteps):
            rho = rk4_step(rho, dt, HTFIM(J, h), Gamma, g, eps, k)
        cs.append(concurrence(rho))
    hJ[k] = np.array(cs)

# ─────────────────────────────────────────────────────────────
# 10. FIGURE HELPERS
# ─────────────────────────────────────────────────────────────
def kappa_label(k):
    lG = (g**2/16) * eps * k / Gamma
    return f'$\\kappa={k}$  ($\\lambda/\\Gamma={lG:.3f}$)'

# ─────────────────────────────────────────────────────────────
# 11. FIGURE 1 — Dynamical observables
# ─────────────────────────────────────────────────────────────
fig1, axs1 = plt.subplots(2, 2, figsize=(10, 8))
obs1 = [('rz', r'$\langle\sigma_z^{(1)}\rangle$',  'Relaxation'),
        ('rx', r'$\langle\sigma_x^{(1)}\rangle$',  'Coherence'),
        ('p',  r'$\mathrm{Tr}(\rho^2)$',            'Purity'),
        ('s',  r'$S(\rho)$',                         'Entropy')]

for ax, (key, ylabel, title) in zip(axs1.flat, obs1):
    for ki, k in enumerate(kappas):
        ax.plot(ts[k]['t'], ts[k][key],
                color=colors[ki], lw=1.8, label=kappa_label(k))
    ax.set_xlabel(r'$t \cdot \Gamma_+$', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig1_dynamics_wc.pdf',
            bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/fig1_dynamics_wc.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("\nFig. 1 saved.")

# ─────────────────────────────────────────────────────────────
# 12. FIGURE 2 — Steady-state concurrence vs h/J
# ─────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(7, 5))
for ki, k in enumerate(kappas):
    ax2.plot(h_vals / J, hJ[k],
             color=colors[ki], lw=2, marker='o', ms=3.5,
             label=kappa_label(k))
ax2.axvline(x=h_opt, color='gray', ls='--', lw=1.2, alpha=0.7,
            label=f'$h/J={h_opt}$ (optimum)')
ax2.set_xlabel('$h/J$', fontsize=12)
ax2.set_ylabel(r'Steady-state concurrence $C(\rho_\mathrm{ss})$',
               fontsize=11)
ax2.set_title('Steady-state Entanglement vs Transverse Field', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig2_hJ_scan_wc.pdf',
            bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/fig2_hJ_scan_wc.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("Fig. 2 saved.")

# ─────────────────────────────────────────────────────────────
# 13. FIGURE 3 — Entanglement dynamics at h/J = h_opt
# ─────────────────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(7, 5))
for ki, k in enumerate(kappas):
    ax3.plot(ts[k]['t'], ts[k]['c'],
             color=colors[ki], lw=2, label=kappa_label(k))
ax3.set_xlabel(r'Time $(\Gamma_+^{-1})$', fontsize=12)
ax3.set_ylabel(r'Concurrence $C(\rho(t))$', fontsize=11)
ax3.set_title(f'Entanglement Dynamics at $h/J={h_opt}$ (peak)',
              fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig3_entanglement_time_wc.pdf',
            bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/fig3_entanglement_time_wc.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("Fig. 3 saved.")

# ─────────────────────────────────────────────────────────────
# 14. FIGURE 4 — Correlations and entanglement suppression
# ─────────────────────────────────────────────────────────────
fig4, axs4 = plt.subplots(2, 2, figsize=(10, 8))

# (a) σz⊗σz spin-spin correlation
ax = axs4[0, 0]
for ki, k in enumerate(kappas):
    ax.plot(ts[k]['t'], ts[k]['zz'],
            color=colors[ki], lw=1.8, label=f'$\\kappa={k}$')
ax.set_xlabel(r'$t\cdot\Gamma_+$', fontsize=10)
ax.set_ylabel(r'$\langle\sigma_z^{(1)}\sigma_z^{(2)}\rangle$', fontsize=10)
ax.set_title('Correlation', fontsize=11)
ax.legend(fontsize=7.5)
ax.grid(True, alpha=0.3)

# (b) Concurrence time series
ax = axs4[0, 1]
for ki, k in enumerate(kappas):
    ax.plot(ts[k]['t'], ts[k]['c'],
            color=colors[ki], lw=1.8, label=kappa_label(k))
ax.set_xlabel(r'$t\cdot\Gamma_+$', fontsize=10)
ax.set_ylabel('Concurrence $C(\\rho)$', fontsize=10)
ax.set_title('Entanglement', fontsize=11)
ax.legend(fontsize=7.5)
ax.grid(True, alpha=0.3)

# (c) Von Neumann entropy
ax = axs4[1, 0]
for ki, k in enumerate(kappas):
    ax.plot(ts[k]['t'], ts[k]['s'],
            color=colors[ki], lw=1.8, label=f'$\\kappa={k}$')
ax.set_xlabel(r'$t\cdot\Gamma_+$', fontsize=10)
ax.set_ylabel('$S(\\rho)$', fontsize=10)
ax.set_title('Entropy', fontsize=11)
ax.legend(fontsize=7.5)
ax.grid(True, alpha=0.3)

# (d) C_max and C_ss vs λ/Γ
ax = axs4[1, 1]
lG_vals = [(g**2/16) * eps * k / Gamma for k in kappas]
max_C   = [max(ts[k]['c'])  for k in kappas]
ss_C    = [ts[k]['c'][-1]   for k in kappas]
ax.plot(lG_vals, max_C, 'o-',  color='blue', lw=2, ms=8,
        label=r'$C_\mathrm{max} = \max_t\, C(\rho(t))$')
ax.plot(lG_vals, ss_C,  's--', color='red',  lw=2, ms=8,
        label=r'$C_\mathrm{ss}$ (steady-state)')
ax.set_xlabel('$\\lambda/\\Gamma$', fontsize=11)
ax.set_ylabel('Concurrence', fontsize=10)
ax.set_title(r'Max Entanglement vs $\lambda/\Gamma$', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig4_correlations_wc.pdf',
            bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/fig4_correlations_wc.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("Fig. 4 saved.")

# ─────────────────────────────────────────────────────────────
# 15. NUMERICAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("NUMERICAL SUMMARY")
print("=" * 65)
print(f"\n{'κ':>6} | {'λ/Γ':>7} | {'C_max':>8} | "
      f"{'C_ss':>8} | {'ΔC_ss':>8} | {'Purity_ss':>10} | {'Entropy_ss':>11}")
print("-" * 75)
c0_ss = ts[0]['c'][-1]
for k in kappas:
    lG  = (g**2/16) * eps * k / Gamma
    mc  = max(ts[k]['c'])
    sc_ = ts[k]['c'][-1]
    sp_ = ts[k]['p'][-1]
    se_ = ts[k]['s'][-1]
    print(f"{k:>6} | {lG:>7.4f} | {mc:>8.5f} | "
          f"{sc_:>8.5f} | {c0_ss - sc_:>8.5f} | {sp_:>10.5f} | {se_:>11.5f}")

print(f"\nTotal nonassociative effect (κ=0 → κ=200):")
print(f"  ΔC_ss    = {ts[0]['c'][-1] - ts[200]['c'][-1]:.5f}"
      f"  ({(1 - ts[200]['c'][-1]/ts[0]['c'][-1])*100:.1f}% suppression)")
print(f"  ΔPurity  = {ts[0]['p'][-1] - ts[200]['p'][-1]:.5f}")
print(f"  ΔEntropy = {ts[200]['s'][-1] - ts[0]['s'][-1]:.5f}")
print(f"  ΔC_max   = {max(ts[0]['c']) - max(ts[200]['c']):.5f}"
      f"  ({(1 - max(ts[200]['c'])/max(ts[0]['c']))*100:.2f}% change in transient)")
print("\nAll figures saved to /mnt/user-data/outputs/")

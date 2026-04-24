#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-qubit TFIM + Lindblad (T=0) + Non-Associative (NA) Simulation
-----------------------------------------------------------------
Generates all single-quantity plots and two composite 4-panel figures:

FIGURE 1: Dynamical quantities (⟨σz¹⟩, ⟨σx¹⟩, Purity, Entropy)
FIGURE 2: Correlation and Entanglement (⟨σz¹σz²⟩, Concurrence, Entropy, Max Concurrence vs κ)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------- Linear algebra helpers ----------
def dag(X): return X.conj().T
def comm(A, B): return A @ B - B @ A

# ---------- Single-qubit Pauli matrices ----------
I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
sp = np.array([[0, 1], [0, 0]], dtype=complex)
sm = np.array([[0, 0], [1, 0]], dtype=complex)
def kron(a,b): return np.kron(a,b)

# ---------- Two-qubit operators ----------
sx1, sy1, sz1 = kron(sx, I2), kron(sy, I2), kron(sz, I2)
sx2, sy2, sz2 = kron(I2, sx), kron(I2, sy), kron(I2, sz)
sm1, sm2 = kron(sm, I2), kron(I2, sm)

# ---------- Hamiltonian ----------
def HTFIM(J, h1, h2):
    return -J*(sz1@sz2) - 0.5*h1*sx1 - 0.5*h2*sx2

# ---------- Lindblad dissipator (T=0) ----------
def lindblad_T0(rho, S_list, Gammas):
    L = np.zeros_like(rho, dtype=complex)
    for S,G in zip(S_list,Gammas):
        SdS = dag(S)@S
        L += G*(S@rho@dag(S) - 0.5*(SdS@rho + rho@SdS))
    return L

# ---------- Non-associative correction ----------
def NA_term(rho, g_list, eps_plus_list, kappa_list):
    rz1 = np.real_if_close(np.trace(rho@sz1))
    rz2 = np.real_if_close(np.trace(rho@sz2))
    c1 = -(g_list[0]**2/16)*eps_plus_list[0]*kappa_list[0]
    c2 = -(g_list[1]**2/16)*eps_plus_list[1]*kappa_list[1]
    return (c1*rz1)*sz1 + (c2*rz2)*sz2

# ---------- Full RHS ----------
def drho_dt(rho,H,S_list,Gammas,g_list,eps_plus_list,kappa_list):
    unitary = -1j*comm(H,rho)
    dissip = lindblad_T0(rho,S_list,Gammas)
    N = NA_term(rho,g_list,eps_plus_list,kappa_list)
    return unitary + dissip + N

# ---------- Metrics ----------
def von_neumann_entropy(rho):
    evals = np.linalg.eigvalsh((rho+dag(rho))/2)
    evals = np.clip(np.real_if_close(evals),0,1)
    nz = evals[evals>1e-14]
    return float(-np.sum(nz*np.log(nz)))

def purity(rho): return float(np.real_if_close(np.trace(rho@rho)))

sy_sy = kron(sy,sy)
def concurrence(rho):
    R = rho @ sy_sy @ rho.conj() @ sy_sy
    evals = np.sqrt(np.real_if_close(np.sort(np.real(np.linalg.eigvals(R))))[::-1])
    return float(max(0.0, evals[0]-evals[1]-evals[2]-evals[3]))

# ---------- Integrator ----------
def propagate_rho(rho0,H,S_list,Gammas,g_list,eps_plus_list,kappa_list,t_grid):
    rho = rho0.copy()
    out = {key:np.zeros_like(t_grid,float) for key in
           ["t","rz1","rx1","zz","entropy","purity","concurrence"]}
    out["t"]=t_grid
    for i,t in enumerate(t_grid):
        out["rz1"][i]=np.real(np.trace(rho@sz1))
        out["rx1"][i]=np.real(np.trace(rho@sx1))
        out["zz"][i]=np.real(np.trace(rho@(sz1@sz2)))
        out["entropy"][i]=von_neumann_entropy(rho)
        out["purity"][i]=purity(rho)
        out["concurrence"][i]=concurrence(rho)
        if i==len(t_grid)-1: break
        dt=t_grid[i+1]-t_grid[i]
        k1=drho_dt(rho,H,S_list,Gammas,g_list,eps_plus_list,kappa_list)
        k2=drho_dt(rho+0.5*dt*k1,H,S_list,Gammas,g_list,eps_plus_list,kappa_list)
        k3=drho_dt(rho+0.5*dt*k2,H,S_list,Gammas,g_list,eps_plus_list,kappa_list)
        k4=drho_dt(rho+dt*k3,H,S_list,Gammas,g_list,eps_plus_list,kappa_list)
        rho += (dt/6)*(k1+2*k2+2*k3+k4)
        rho=0.5*(rho+dag(rho)); rho/=np.trace(rho)
    return out

# ---------- Parameters ----------
J=h1=h2=1.0
Gamma1=Gamma2=1.0
Gammas=[Gamma1,Gamma2]
g_list=[1.0,1.0]
eps_list=[0.2,0.2]
S_list=[sm1,sm2]
kappa_sweep=[0.0,0.5,1.0,1.5,2.0]

plus=(1/np.sqrt(2))*np.array([1,1],complex)
rho_plus=np.outer(plus,plus.conj())
rho0=np.kron(rho_plus,rho_plus)

t_max,dt=10.0,0.01
t_grid=np.arange(0,t_max+dt,dt)

# ---------- Run simulations ----------
results={}
for kappa in kappa_sweep:
    kappas=[kappa,kappa]
    results[kappa]=propagate_rho(rho0,HTFIM(J,h1,h2),S_list,Gammas,g_list,eps_list,kappas,t_grid)

# ---------- Plot helper ----------
def line(ax,t,y_dict,ylabel,title):
    for k,y in y_dict.items(): ax.plot(t,y,label=f"κ={k}")
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)

# ---------- FIGURE 1: Dynamical quantities ----------
fig1, axs = plt.subplots(2,2,figsize=(8,6))
line(axs[0,0], t_grid, {k:results[k]["rz1"] for k in kappa_sweep}, "⟨σz¹⟩", "Relaxation")
line(axs[0,1], t_grid, {k:results[k]["rx1"] for k in kappa_sweep}, "⟨σx¹⟩", "Coherence")
line(axs[1,0], t_grid, {k:results[k]["purity"] for k in kappa_sweep}, "Tr(ρ²)", "Purity")
line(axs[1,1], t_grid, {k:results[k]["entropy"] for k in kappa_sweep}, "S(ρ)", "Entropy")
fig1.tight_layout()
fig1.savefig("figure1_dynamics.png", dpi=400)

# ---------- FIGURE 2: Correlation & Entanglement ----------
# Compute max concurrence for each κ
maxC = [max(results[k]["concurrence"]) for k in kappa_sweep]

fig2, axs = plt.subplots(2,2,figsize=(8,6))
line(axs[0,0], t_grid, {k:results[k]["zz"] for k in kappa_sweep}, "⟨σz¹σz²⟩", "Correlation")
line(axs[0,1], t_grid, {k:results[k]["concurrence"] for k in kappa_sweep}, "Concurrence", "Entanglement")
line(axs[1,0], t_grid, {k:results[k]["entropy"] for k in kappa_sweep}, "S(ρ)", "Entropy")
axs[1,1].plot(kappa_sweep, maxC, "o-", color="C3")
axs[1,1].set_xlabel("κ")
axs[1,1].set_ylabel("Max Concurrence")
axs[1,1].set_title("Max Entanglement vs κ")
fig2.tight_layout()
fig2.savefig("figure2_correlation_entanglement.png", dpi=400)

# ---------- Save CSVs ----------
for kappa in kappa_sweep:
    df=pd.DataFrame({
        "t":results[kappa]["t"],
        "rz1":results[kappa]["rz1"],
        "rx1":results[kappa]["rx1"],
        "zz":results[kappa]["zz"],
        "entropy":results[kappa]["entropy"],
        "purity":results[kappa]["purity"],
        "concurrence":results[kappa]["concurrence"],
    })
    df.to_csv(f"results_kappa_{kappa:.1f}.csv",index=False)

print("Simulation complete. Plots and CSV files saved in current directory.")

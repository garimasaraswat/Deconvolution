import torch
import numpy as np
import matplotlib.pyplot as plt

# Constants
kB = 8.617e-5  # eV/K

# Tip LDOS (Dynes broadened BCS form)
def rho_tip(E, amplitude=1.0, delta=0.002, gamma=0.0001):
    denom = torch.sqrt(E**2 + 2j * E * gamma - delta**2)
    result = amplitude * torch.sign(E) * torch.abs(torch.real(E / denom))
    return result

# Fermi function derivative
def dfermi_dE(E, T):
    kT = kB * T
    exp_term = torch.exp(E / kT)
    return exp_term / (kT * (1 + exp_term)**2)

# Random sample LDOS generator
def random_ldos(E, smooth=True):
    raw = torch.rand(len(E)) * 2  # Random LDOS [0, 2]
    if smooth:
        window = torch.hamming_window(15, periodic=False)
        raw = torch.nn.functional.conv1d(raw.view(1, 1, -1), window.view(1, 1, -1), padding=7)[0, 0]
        raw = raw[:len(E)]  # Trim
    return torch.relu(raw)

# Simulate dI/dV spectrum
def simulate_dIdV(E, V, T, rho_S, tip_params):
    dIdV = []
    rho_T_full = rho_tip(E[:, None] - V[None, :], *tip_params)  # (E, V)
    dfdE = dfermi_dE(E[:, None], T)  # (E, 1)
    integrand = rho_T_full * rho_S[:, None] * dfdE
    dIdV = torch.trapz(integrand, E, dim=0)
    return dIdV

# Example usage
if __name__ == '__main__':
    T = 4.2  # K
    E_vals = torch.linspace(-0.05, 0.05, 1000)
    V_vals = torch.linspace(-0.03, 0.03, 200)
    tip_params = [1.0, 0.002, 0.0001]  # amplitude, delta, gamma

    # Generate a random sample LDOS
    sample_ldos = random_ldos(E_vals, smooth=True)

    # Simulate dI/dV
    dIdV_sim = simulate_dIdV(E_vals, V_vals, T, sample_ldos, tip_params)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(V_vals, dIdV_sim.numpy(), label="Simulated dI/dV")
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("dI/dV (a.u.)")
    plt.title("Simulated STS Spectrum with Superconducting Tip")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Optional: also show LDOS
    plt.figure()
    plt.plot(E_vals, sample_ldos.numpy(), label="Sample LDOS")
    plt.xlabel("Energy (eV)")
    plt.ylabel("LDOS")
    plt.title("Random Sample LDOS")
    plt.grid(True)
    plt.legend()
    plt.show()

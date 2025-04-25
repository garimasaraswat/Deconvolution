import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- Constants ---
e = 1.602e-19  # charge of electron, but we’ll keep all energy units in eV
kB = 8.617e-5  # eV/K

# --- Physical Models ---
def rho_tip(E, params):
    amplitude, delta, gamma = params
    complex_denominator = torch.sqrt(E**2 + 2j * E * gamma - delta**2)
    real_part = torch.real(E / complex_denominator)
    return amplitude * torch.sign(E) * torch.abs(real_part)

def fermi_dirac(E, eV, T, mu=0.0):
    kT = kB * T
    return 1.0 / (torch.exp((E - eV - mu) / kT) + 1.0)

def d_fermi_dirac_dV(E, eV, T, mu=0.0):
    kT = kB * T
    exponent = (E - eV - mu) / kT
    exp_val = torch.exp(exponent)
    return exp_val / (kT * (1 + exp_val) ** 2)

def trapz_torch(y, x):
    return torch.trapz(y, x)

# --- Neural Network ---
class LDOSNet(nn.Module):
    def __init__(self, num_locations, num_E):
        super(LDOSNet, self).__init__()
        self.num_locations = num_locations
        self.num_E = num_E
        self.tip_param_net = nn.Sequential(
            nn.Linear(100, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        self.sample_ldos_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(100, 64),
                nn.ReLU(),
                nn.Linear(64, num_E),
                nn.ReLU()
            ) for _ in range(num_locations)
        ])

    def forward(self, dIdV_spectra):
        tip_params = self.tip_param_net(dIdV_spectra[0])
        sample_ldos = [self.sample_ldos_nets[i](dIdV_spectra[i]) for i in range(self.num_locations)]
        return tip_params, sample_ldos

# --- Physics-Informed Loss Function ---
def physics_loss(tip_params, sample_ldos, V_all, dIdV_exp_all, E_vals, T):
    loss = 0.0
    for i in range(len(V_all)):
        V = V_all[i]
        dIdV_exp = dIdV_exp_all[i]
        rho_S = sample_ldos[i]
        model_vals = []
        for v in V:
            E_shifted = E_vals - v
            f_diff = fermi_dirac(E_shifted, v, T) - fermi_dirac(E_vals, 0.0, T)
            df_dV = d_fermi_dirac_dV(E_shifted, v, T)
            rho_T = rho_tip(E_shifted, tip_params)

            integrand1 = f_diff * rho_T * rho_S
            integrand2 = df_dV * rho_T * rho_S

            dIdV_val = trapz_torch(integrand1 + integrand2, E_vals)
            model_vals.append(dIdV_val)

        dIdV_model = torch.stack(model_vals)
        loss += nn.MSELoss()(dIdV_model, dIdV_exp)
    return loss

# --- Optional: Normalize dI/dV spectra ---
def normalize_spectrum(spectrum):
    return (spectrum - spectrum.mean()) / spectrum.std()

# --- Main ---
if __name__ == '__main__':
    torch.manual_seed(42)

    # --- Parameters ---
    T = 4.2  # Kelvin
    E_min, E_max = -0.05, 0.05  # eV
    num_E = 100
    num_locations = 2
    num_V = 50

    # Energy and voltage axes
    E_vals_np = np.linspace(E_min, E_max, num_E)
    E_vals = torch.tensor(E_vals_np, dtype=torch.float32)
    V_points = np.linspace(-0.03, 0.03, num_V)
    V_all = [torch.tensor(V_points, dtype=torch.float32) for _ in range(num_locations)]

    # Simulated experimental data (replace with your real data)
    dIdV_exp_all = [torch.randn(num_V) for _ in range(num_locations)]
    dIdV_exp_all = [normalize_spectrum(s) for s in dIdV_exp_all]

    # --- Model ---
    model = LDOSNet(num_locations, num_E)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 100

    # --- Training ---
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        tip_params, sample_ldos = model(dIdV_exp_all)
        loss = physics_loss(tip_params, sample_ldos, V_all, dIdV_exp_all, E_vals, T)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    # --- Evaluation ---
    model.eval()
    with torch.no_grad():
        fitted_tip_params, fitted_sample_ldos = model(dIdV_exp_all)
        print("\nFitted Tip Parameters:", fitted_tip_params.numpy())
        for i in range(num_locations):
            plt.figure()
            plt.plot(E_vals_np, fitted_sample_ldos[i].numpy(), label=f'Fitted Sample LDOS (Loc {i+1})')
            plt.xlabel("Energy (eV)")
            plt.ylabel("LDOS")
            plt.title(f"Sample LDOS at Location {i+1}")
            plt.legend()
            plt.grid(True)
            plt.show()

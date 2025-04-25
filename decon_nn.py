import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import trapz

# Define the BCS-like tip LDOS function (as before)
def rho_tip(E, params):
    amplitude, delta, gamma = params
    complex_denominator = torch.sqrt(E**2 + 2j * E * gamma - delta**2)
    real_part = torch.real(E / complex_denominator)
    return amplitude * torch.sign(E) * torch.abs(real_part)

def fermi_dirac(E, eV, T, mu=0):
    kT = 8.617e-5 * T
    return 1.0 / (torch.exp((E - eV - mu) / kT) + 1.0)

def d_fermi_dirac_dV(E, eV, T, mu=0):
    kT = 8.617e-5 * T
    exponent = (E - eV - mu) / kT
    exp_val = torch.exp(exponent)
    return (e / kT) * exp_val / ((1 + exp_val)**2)

# Define the neural network
class LDOSNet(nn.Module):
    def __init__(self, num_locations, num_E):
        super(LDOSNet, self).__init__()
        self.num_locations = num_locations
        self.num_E = num_E
        # Network to predict tip parameters (shared across locations)
        self.tip_param_net = nn.Sequential(
            nn.Linear(100, 32), # Example: Input dI/dV (discretized)
            nn.ReLU(),
            nn.Linear(32, 3)    # Output: amplitude, delta, gamma
        )
        # Network to predict sample LDOS for each location
        self.sample_ldos_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(100, 64),
                nn.ReLU(),
                nn.Linear(64, num_E), # Output: rho_S at num_E points
                nn.ReLU() # Ensure non-negative LDOS
            ) for _ in range(num_locations)
        ])

    def forward(self, dIdV_spectra):
        tip_params = self.tip_param_net(dIdV_spectra[0]) # Assume first spectrum to predict shared tip params
        sample_ldos = [self.sample_ldos_nets[i](dIdV_spectra[i]) for i in range(self.num_locations)]
        return tip_params, sample_ldos

# Loss function incorporating the physics
def physics_loss(tip_params, sample_ldos, V_all, dIdV_exp_all, E_vals, T):
    loss = 0
    for i in range(len(V_all)):
        dIdV_model = torch.stack([
            torch.tensor(trapz((fermi_dirac(E_vals - e * v, e * v, T) - fermi_dirac(E_vals, 0, T)).detach().numpy() *
                              torch.autograd.grad(rho_tip(E_vals - e * v, tip_params).sum(), tip_params, create_graph=True)[0].detach().numpy() * sample_ldos[i].detach().numpy(), E_vals), requires_grad=True)
            + torch.tensor(trapz(d_fermi_dirac_dV(E_vals - e * v, e * v, T).detach().numpy() *
                              rho_tip(E_vals - e * v, tip_params).detach().numpy() * sample_ldos[i].detach().numpy(), E_vals), requires_grad=True)
            for v in V_all[i]
        ])
        loss += nn.MSELoss()(dIdV_model, torch.tensor(dIdV_exp_all[i]))
    return loss

if __name__ == '__main__':
    # --- Data Preparation (Replace with your data) ---
    T = 4.2
    E_min = -0.05
    E_max = 0.05
    num_E = 100
    E_vals_np = np.linspace(E_min, E_max, num_E)
    E_vals = torch.tensor(E_vals_np, dtype=torch.float32, requires_grad=True)
    V_points = np.linspace(-0.03, 0.03, 50)
    num_locations = 2
    e = 1.602e-19

    # Example dI/dV data (replace with your actual data)
    dIdV_exp_all = [torch.randn(len(V_points)) for _ in range(num_locations)]
    V_all = [torch.tensor(V_points, dtype=torch.float32) for _ in range(num_locations)]

    # --- Model and Training ---
    model = LDOSNet(num_locations, num_E)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 100

    for epoch in range(epochs):
        optimizer.zero_grad()
        tip_params, sample_ldos = model(dIdV_exp_all)
        loss = physics_loss(tip_params, sample_ldos, V_all, [d.numpy() for d in dIdV_exp_all], E_vals_np, T)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # --- Evaluation (Illustrative) ---
    with torch.no_grad():
        fitted_tip_params, fitted_sample_ldos = model(dIdV_exp_all)
        print("\nFitted Tip Parameters:", fitted_tip_params.numpy())
        for i in range(num_locations):
            plt.figure()
            plt.plot(E_vals_np, fitted_sample_ldos[i].numpy(), label=f'Fitted Sample LDOS (Loc {i+1})')
            plt.xlabel("Energy (eV)")
            plt.ylabel("LDOS")
            plt.title(f"Fitted Sample LDOS (Location {i+1})")
            plt.legend()
            plt.show()

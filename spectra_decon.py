import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import trapz
import matplotlib.pyplot as plt

def fermi_dirac(E, eV, T, mu=0):
    """Fermi-Dirac distribution function."""
    kT = 8.617e-5 * T  # in eV/K
    return 1.0 / (np.exp((E - eV - mu) / kT) + 1.0)

def d_fermi_dirac_dV(E, eV, T, mu=0):
    """Derivative of Fermi-Dirac distribution with respect to V."""
    kT = 8.617e-5 * T
    exponent = (E - eV - mu) / kT
    exp_val = np.exp(exponent)
    return (e / kT) * exp_val / ((1 + exp_val)**2)

def rho_tip(E, params):
    """BCS-like density of states for the tip."""
    amplitude, delta, gamma = params
    complex_denominator = np.sqrt(E**2 + 2j * E * gamma - delta**2)
    real_part = np.real(E / complex_denominator)
    return amplitude * np.sign(E) * np.abs(real_part)

def d_rho_tip_dV(E, eV, params, delta_V=1e-6):
    """Numerical derivative of rho_tip with respect to V."""
    amplitude, delta, gamma = params
    rho_plus = rho_tip(E - e * (eV + delta_V), [amplitude, delta, gamma])
    rho_minus = rho_tip(E - e * (eV - delta_V), [amplitude, delta, gamma])
    return (rho_plus - rho_minus) / (2 * delta_V) * (-e)

# Constants
e = 1.602e-19  # Elementary charge (not explicitly used if we keep proportionality)
kB = 8.617e-5   # Boltzmann constant in eV/K

def calculate_dIdV(V, rho_sample_E, E_vals, T, tip_params):
    """Calculates dI/dV for a given voltage and sample LDOS."""
    integral1 = trapz((fermi_dirac(E_vals - e * V, e * V, T) - fermi_dirac(E_vals, 0, T)) *
                      d_rho_tip_dV(E_vals, e * V, tip_params), E_vals)
    integral2 = trapz(d_fermi_dirac_dV(E_vals - e * V, e * V, T) *
                      rho_tip(E_vals - e * V, tip_params) * rho_sample_E, E_vals)
    return integral1 + integral2

def residuals(params_combined, V_all, dIdV_exp_all, E_vals, T, num_locations, num_tip_params, num_E):
    """Residuals to minimize in the least squares fit."""
    tip_params = params_combined[:num_tip_params]
    rho_sample_all = params_combined[num_tip_params:].reshape(num_locations, num_E)
    residuals_all = []

    for i in range(num_locations):
        dIdV_model = [calculate_dIdV(v, rho_sample_all[i], E_vals, T, tip_params) for v in V_all[i]]
        residuals_all.extend(np.array(dIdV_exp_all[i]) - np.array(dIdV_model))

    return np.array(residuals_all)

if __name__ == '__main__':
    # --- Simulation of Example Data (Replace with your actual data) ---
    T = 4.2  # Temperature in Kelvin
    E_min = -0.05  # Energy range in eV
    E_max = 0.05
    num_E = 100
    E_vals = np.linspace(E_min, E_max, num_E)
    V_points = np.linspace(-0.03, 0.03, 50)
    num_locations = 2

    # True tip parameters (amplitude, delta, gamma)
    true_tip_params = [1.5, 0.008, 0.001]

    # Generate some "true" sample LDOS for different locations
    true_rho_sample = []
    for i in range(num_locations):
        center = 0.005 * (i - 0.5)
        width = 0.005 + 0.002 * i
        true_rho_sample.append(1.0 + 0.8 * np.exp(-((E_vals - center)**2) / (2 * width**2)))

    # Generate synthetic dI/dV data
    dIdV_exp_all = []
    V_all = []
    for i in range(num_locations):
        dIdV_exp = [calculate_dIdV(v, true_rho_sample[i], E_vals, T, true_tip_params) + 0.005 * np.random.randn() for v in V_points] # Add some noise
        dIdV_exp_all.append(dIdV_exp)
        V_all.append(V_points)

    # --- Optimization ---
    num_tip_params = len(true_tip_params)
    initial_tip_params = [1.0, 0.005, 0.002]  # Initial guess for tip parameters
    initial_rho_sample = [np.ones(num_E) for _ in range(num_locations)] # Initial guess for sample LDOS

    initial_params_combined = np.concatenate([initial_tip_params] + initial_rho_sample)

    # Perform least squares optimization
    result = least_squares(residuals, initial_params_combined,
                           args=(V_all, dIdV_exp_all, E_vals, T, num_locations, num_tip_params, num_E),
                           verbose=2,
                           max_nfev=500)

    # Extract the fitted parameters
    fitted_tip_params = result.x[:num_tip_params]
    fitted_rho_sample = result.x[num_tip_params:].reshape(num_locations, num_E)

    print("\n--- Fitted Results ---")
    print("True Tip Parameters:", true_tip_params)
    print("Fitted Tip Parameters:", fitted_tip_params)

    # --- Plotting ---
    plt.figure(figsize=(14, 10))

    # Plot tip LDOS
    plt.subplot(num_locations + 1, 1, 1)
    plt.plot(E_vals, rho_tip(E_vals, true_tip_params), label='True Tip LDOS')
    plt.plot(E_vals, rho_tip(E_vals, fitted_tip_params), '--', label='Fitted Tip LDOS')
    plt.xlabel("Energy (eV)")
    plt.ylabel("LDOS")
    plt.title("Tip LDOS")
    plt.legend()

    # Plot sample LDOS for each location
    for i in range(num_locations):
        plt.subplot(num_locations + 1, 1, i + 2)
        plt.plot(E_vals, true_rho_sample[i], label=f'True Sample LDOS (Loc {i+1})')
        plt.plot(E_vals, fitted_rho_sample[i], '--', label=f'Fitted Sample LDOS (Loc {i+1})')
        plt.xlabel("Energy (eV)")
        plt.ylabel("LDOS")
        plt.title(f"Sample LDOS (Location {i+1})")
        plt.legend()

    plt.tight_layout()
    plt.show()

    # --- Plotting dI/dV (Comparison with Data) ---
    plt.figure(figsize=(14, 10))
    for i in range(num_locations):
        plt.subplot(num_locations, 1, i + 1)
        plt.plot(V_all[i], dIdV_exp_all[i], 'o', label=f'Exp. dI/dV (Loc {i+1})')
        dIdV_fitted = [calculate_dIdV(v, fitted_rho_sample[i], E_vals, T, fitted_tip_params) for v in V_all[i]]
        plt.plot(V_all[i], dIdV_fitted, '-', label=f'Fitted dI/dV (Loc {i+1})')
        plt.xlabel("Voltage (V)")
        plt.ylabel("dI/dV (arb. units)")
        plt.title(f"dI/dV Comparison (Location {i+1})")
        plt.legend()

    plt.tight_layout()
    plt.show()

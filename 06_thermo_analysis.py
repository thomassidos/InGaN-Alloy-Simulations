import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import cumtrapz
from typing import Tuple

# --- Constants ---
KB = 8.617333262e-5  # Boltzmann constant in eV/K
BASE_DIR = Path('final_simulations')
OUTPUT_DIR = Path('thermo_analysis')
OUTPUT_DIR.mkdir(exist_ok=True)

def analyze_composition(file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads MC output and calculates Entropy and Free Energy via integration.
    Returns (Temperature, Entropy_mix, FreeEnergy_mix).
    """
    data = np.loadtxt(file_path)
    # Sort by temperature (low to high needed for integration)
    data = data[data[:, 0].argsort()]
    
    T = data[:, 0]
    E_mix = data[:, 1] # Mixing energy
    
    # Thermodynamic Integration
    # F(T) / T - F(T0) / T0 = - Integral_{T0}^{T} (E / T^2) dT
    # Assuming at T=0 (or T_min), F_mix approx E_mix (Entropy is small)
    
    # 1. Compute Integral
    integrand = E_mix / (T**2)
    integral = cumtrapz(integrand, T, initial=0)
    
    # 2. Compute Free Energy F_mix
    # Reference point at lowest T
    T_ref = T[0]
    F_ref = E_mix[0] 
    
    F_mix = T * ((F_ref / T_ref) - integral)
    
    # 3. Compute Entropy S_mix = (E - F) / T
    S_mix = (E_mix - F_mix) / T
    
    return T, S_mix, F_mix

def plot_thermo(x_vals, y_vals, y_label, title, filename):
    """Helper for consistent plotting."""
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, 'o-')
    plt.xlabel("Temperature (K)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()

def main():
    # Loop over composition folders
    for folder in sorted(BASE_DIR.iterdir()):
        if not folder.is_dir(): continue
        try:
            conc = float(folder.name)
        except ValueError: continue
            
        sim_file = folder / 'output.txt'
        if not sim_file.exists(): continue
        
        print(f"Analyzing x = {conc}...")
        T, S_mix, F_mix = analyze_composition(sim_file)
        
        # Normalize Entropy by kB
        S_mix_kB = S_mix / KB
        
        # Plot for this composition
        plot_thermo(T, S_mix_kB, 
                   r"Entropy ($k_B$/atom)", 
                   f"Mixing Entropy (x={conc})", 
                   f"entropy_x{conc:.2f}.png")
                   
        plot_thermo(T, F_mix, 
                   r"Free Energy (eV/atom)", 
                   f"Mixing Free Energy (x={conc})", 
                   f"free_energy_x{conc:.2f}.png")

if __name__ == "__main__":
    main()

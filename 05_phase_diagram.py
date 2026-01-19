import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# --- Constants ---
# Total energies per atom for pure endpoints (eV/atom)
E_INN = -11.7571335
E_GAN = -13.9413455
DELTA_X = 0.05  # Composition step in simulations
SIM_DIR = Path('final_simulations') # Directory containing MC outputs

def load_simulation_data(base_dir: Path) -> pd.DataFrame:
    """Loads MC results from all composition folders."""
    data = []
    
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory {base_dir} not found.")

    for folder in base_dir.iterdir():
        if not folder.is_dir(): continue
        
        try:
            x = float(folder.name) # Folder names should be '0.05', '0.10', etc.
        except ValueError: continue
            
        output_file = folder / 'output.txt'
        if not output_file.exists(): continue
        
        # Read T, E_mix, Cv
        df_temp = pd.read_csv(output_file, delim_whitespace=True, 
                              header=None, names=["T", "Emix", "Cp"])
        
        for _, row in df_temp.iterrows():
            data.append([x, row["T"], row["Emix"]])
            
    # Add endpoints (pure phases have 0 mixing energy)
    df = pd.DataFrame(data, columns=["x", "T", "Emix"])
    unique_temps = df["T"].unique()
    
    zeros = pd.DataFrame({"x": [0.0]*len(unique_temps) + [1.0]*len(unique_temps),
                          "T": list(unique_temps) * 2,
                          "Emix": 0.0})
    
    return pd.concat([df, zeros], ignore_index=True)

def calculate_chemical_potential(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Delta Mu = 2 * (dEmix/dx + E_InN - E_GaN).
    Uses finite differences for dEmix/dx.
    """
    dmu_data = []
    unique_temps = sorted(df["T"].unique())
    dmu_ref = 2 * (E_INN - E_GAN) # Constant term
    
    for T in unique_temps:
        # Get Emix vs x for this temperature
        df_T = df[df["T"] == T].set_index("x")["Emix"]
        
        # Loop through compositions (excluding 0)
        for x in np.arange(0.05, 1.01, DELTA_X):
            x = round(x, 2)
            x_prev = round(x - DELTA_X, 2)
            
            if x in df_T.index and x_prev in df_T.index:
                # Finite difference derivative
                d_emix = (df_T.loc[x] - df_T.loc[x_prev]) / DELTA_X
                
                # Formula from Thesis
                dmu = 2 * d_emix + dmu_ref
                dmu_data.append([x, T, dmu])
        
        # Add edges manually for visualization consistency
        dmu_data.append([0.0, T, dmu_ref]) 
        dmu_data.append([1.0, T, dmu_ref]) # Approx

    return pd.DataFrame(dmu_data, columns=["x", "T", "dmu"])

def plot_phase_diagram(dmu_df: pd.DataFrame):
    """Interpolates and plots the Delta Mu Phase Diagram."""
    x = dmu_df["x"].values
    y = dmu_df["T"].values
    z = dmu_df["dmu"].values

    # Create grid
    xi = np.linspace(0, 1, 200)
    yi = np.linspace(y.min(), y.max(), 200)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate
    Zi = griddata((x, y), z, (Xi, Yi), method='linear')
    Zi_smooth = gaussian_filter(Zi, sigma=1.5) # Smooth contours
    
    # Plot
    sns.set_context("talk")
    plt.figure(figsize=(10, 8))
    
    # Filled contours
    cp = plt.contourf(Xi, Yi, Zi_smooth, levels=100, cmap="plasma")
    plt.colorbar(cp, label=r'$\Delta\mu_{In-Ga}$ (eV)')
    
    plt.xlabel(r'Concentration $x$ ($In_xGa_{1-x}N$)')
    plt.ylabel('Temperature (K)')
    plt.title('Phase Diagram: Chemical Potential Difference')
    plt.tight_layout()
    plt.savefig('phase_diagram.png')
    plt.show()

if __name__ == "__main__":
    raw_data = load_simulation_data(SIM_DIR)
    dmu_data = calculate_chemical_potential(raw_data)
    plot_phase_diagram(dmu_data)

import time
import numpy as np
from pathlib import Path
from ase.io import read, write
from ase.build import make_supercell
from mchammer.ensembles import CanonicalEnsemble
from mchammer.calculators import ClusterExpansionCalculator
from icet import ClusterExpansion
from icet.tools.structure_generation import occupy_structure_randomly

# --- Configuration ---
CE_MODEL_PATH = 'mixing_energy.ce'
SUPERCELL_SIZE = [10, 10, 5]
# Compositions to simulate (example: just 0.35, loop over this list externally or use job arrays on HPC)
TARGET_CONCENTRATION = 0.35 
TEMP_START = 1800
TEMP_END = 0
TEMP_STEP_HIGH = 20  # High T step
TEMP_STEP_LOW = 5    # Low T step (< 800K)
KB = 8.617333262e-5  # eV/K

def get_simulation_schedule():
    """Defines the annealing schedule."""
    # Split into two ranges as per thesis
    range_high = np.arange(TEMP_START, 800, -TEMP_STEP_HIGH)
    range_low = np.arange(800, TEMP_END, -TEMP_STEP_LOW)
    return np.concatenate((range_high, range_low))

def run_mc_fixed_steps(supercell, calc, T, steps=1000000, equil_steps=500000):
    """
    Runs MC simulation for a fixed number of steps.
    Returns the final structure and calculated observables.
    """
    mc = CanonicalEnsemble(structure=supercell, calculator=calc, temperature=T,
                           ensemble_data_write_interval=1000) # Write less often to save I/O
    
    # Run simulation
    mc.run(steps)
    
    # Analyze results (last 50% of trajectory)
    # mchammer stores data in mc.data_container
    data = mc.data_container.get('potential')
    valid_data = data[-equil_steps:] # Take last N steps
    
    N_atoms = len(supercell)
    energies_per_atom = valid_data / N_atoms
    
    mean_energy = np.mean(energies_per_atom)
    mean_sq_energy = np.mean(energies_per_atom**2)
    
    # Heat Capacity: Cv = (<E^2> - <E>^2) / (kB * T^2)
    # Note: Energies must be TOTAL energies for fluctuation formula, then normalized, 
    # but here we use per-atom variance directly scaling.
    # Correct formula using per-atom values: N * (<e^2> - <e>^2) / (kB*T^2)
    var_energy = np.var(energies_per_atom)
    heat_capacity = (N_atoms * var_energy) / (KB * T**2)
    
    return mc.structure, mean_energy, heat_capacity

def main():
    # Setup paths
    base_dir = Path(f"simulations/conc_{TARGET_CONCENTRATION:.2f}")
    base_dir.mkdir(parents=True, exist_ok=True)
    output_txt = base_dir / "output.txt"
    
    # Load CE Model
    if not Path(CE_MODEL_PATH).exists():
        raise FileNotFoundError(f"CE model {CE_MODEL_PATH} not found.")
    ce = ClusterExpansion.read(CE_MODEL_PATH)
    
    # Initialize Supercell
    # Try to load previous state or create new
    initial_struct_path = base_dir / "initial_structure.xyz"
    
    if initial_struct_path.exists():
        print(f"Resuming from {initial_struct_path}")
        supercell = read(initial_struct_path)
    else:
        print("Generating random initial supercell...")
        sc_prim = occupy_structure_randomly(ce.primitive_structure, SUPERCELL_SIZE, 
                                            {'In': TARGET_CONCENTRATION, 'Ga': 1-TARGET_CONCENTRATION})
        supercell = sc_prim
    
    calc = ClusterExpansionCalculator(supercell, ce)
    temps = get_simulation_schedule()
    
    # Check already computed temps
    computed_temps = set()
    if output_txt.exists():
        data = np.loadtxt(output_txt)
        if data.ndim == 1 and data.size > 0: computed_temps.add(int(data[0]))
        elif data.ndim > 1: computed_temps.update(data[:, 0].astype(int))
            
    print(f"Starting simulation for x={TARGET_CONCENTRATION} on Metropolis HPC node...")
    
    for T in temps:
        if int(T) in computed_temps:
            continue
            
        print(f"Simulating T = {T} K...")
        start_t = time.perf_counter()
        
        try:
            supercell, E_mean, Cv = run_mc_fixed_steps(supercell, calc, T)
            
            # Save Data
            with open(output_txt, "a") as f:
                f.write(f"{T} {E_mean:.6f} {Cv:.6f}\n")
            
            # Save Structure
            struct_file = base_dir / f"structure_{T}K.xyz"
            write(struct_file, supercell)
            
            # Update 'restart' file
            write(initial_struct_path, supercell)
            
        except Exception as e:
            print(f"Error at T={T}: {e}")
            break
            
        print(f"Finished {T}K in {time.perf_counter() - start_t:.2f}s")

if __name__ == "__main__":
    main()

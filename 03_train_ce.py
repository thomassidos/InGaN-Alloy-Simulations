import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ase.db import connect
from icet import ClusterSpace, StructureContainer, ClusterExpansion
from icet.tools import ConvexHull, enumerate_structures
from trainstation import CrossValidationEstimator

# --- Configuration ---
DB_FILE = Path('new.db')
OUTPUT_CE = Path('mixing_energy.ce')
CUTOFFS = [6.0, 3.5, 2.5]  # Optimized values from previous step
FIT_METHOD = 'ardr'        # Automatic Relevance Determination Regression

def train_and_save_model():
    """Trains the final CE model and saves it to disk."""
    print("Initializing Cluster Space...")
    with connect(DB_FILE) as db:
        primitive_structure = db.get(id=1).toatoms()
        
    cs = ClusterSpace(structure=primitive_structure,
                      cutoffs=CUTOFFS,
                      chemical_symbols=['In', 'Ga'],
                      position_tolerance=1e-03)
    
    sc = StructureContainer(cluster_space=cs)
    
    print("Loading structures from database...")
    with connect(DB_FILE) as db:
        for row in db.select():
            sc.add_structure(structure=row.toatoms(),
                             user_tag=row.tag,
                             properties={'mixing_energy': row.mixing_energy})
    
    print(f"Training using {FIT_METHOD}...")
    opt = CrossValidationEstimator(fit_data=sc.get_fit_data(key='mixing_energy'),
                                   fit_method=FIT_METHOD)
    opt.validate()
    opt.train()
    print(opt.summary)
    
    ce = ClusterExpansion(cluster_space=cs,
                          parameters=opt.parameters,
                          metadata=opt.summary)
    ce.write(OUTPUT_CE)
    print(f"Model saved to {OUTPUT_CE}")

def validate_predictions():
    """Compares DFT reference energies vs CE predictions."""
    if not OUTPUT_CE.exists():
        raise FileNotFoundError("CE model not found. Run training first.")
        
    ce = ClusterExpansion.read(OUTPUT_CE)
    data = {'conc': [], 'dft': [], 'ce': []}
    
    with connect(DB_FILE) as db:
        for row in db.select('natoms<=100'): # Validate on smaller structures
            ref = 1e3 * row.mixing_energy # Convert to meV
            pred = 1e3 * ce.predict(row.toatoms())
            
            data['conc'].append(row.concentration)
            data['dft'].append(ref)
            data['ce'].append(pred)
            
    # Plotting
    plt.figure(figsize=(6, 5))
    plt.scatter(data['conc'], data['dft'], marker='o', label='DFT Reference', alpha=0.7)
    plt.scatter(data['conc'], data['ce'], marker='x', label='CE Prediction', alpha=0.7)
    plt.xlabel('Ga Concentration')
    plt.ylabel('Mixing Energy (meV/atom)')
    plt.legend()
    plt.title('Model Validation')
    plt.tight_layout()
    plt.savefig('validation_plot.png')
    plt.show()

def compute_convex_hull():
    """Generates and plots the convex hull of stability at T=0K."""
    print("Enumerating structures for Convex Hull...")
    ce = ClusterExpansion.read(OUTPUT_CE)
    primitive = ce.primitive_structure
    
    concs, energies = [], []
    
    # Enumerate small supercells to find ground states
    for structure in enumerate_structures(structure=primitive,
                                          sizes=range(1, 7), # Up to size 6
                                          chemical_symbols=['In', 'Ga'],
                                          position_tolerance=1e-03):
        conc = structure.get_chemical_symbols().count('Ga') / len(structure)
        energy = ce.predict(structure)
        concs.append(conc)
        energies.append(energy)
        
    hull = ConvexHull(concentrations=concs, energies=energies)
    
    # Plot
    plt.figure(figsize=(6, 5))
    plt.scatter(concs, np.array(energies) * 1e3, marker='.', color='gray', alpha=0.5)
    plt.plot(hull.concentrations, np.array(hull.energies) * 1e3, 'o-', color='green', lw=2)
    plt.xlabel('Ga Concentration')
    plt.ylabel('Formation Energy (meV/atom)')
    plt.title('T=0 K Convex Hull')
    plt.tight_layout()
    plt.savefig('convex_hull.png')
    plt.show()

if __name__ == "__main__":
    train_and_save_model()
    validate_predictions()
    compute_convex_hull()

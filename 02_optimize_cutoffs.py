import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
from ase.db import connect
from icet import ClusterSpace, StructureContainer
from trainstation import CrossValidationEstimator

# --- Configuration ---
DB_FILE = Path('new.db')
PRIMITIVE_ID = 1  # ID of the primitive structure in the database
FIT_METHOD = 'lasso' # LASSO allows for feature selection
NSPLITS = 100        # Cross-validation splits

def get_fit_data(db_path: Path, primitive_structure, cutoffs: List[float]) -> tuple:
    """
    Constructs the StructureContainer and retrieves fitting data (A matrix, y vector).
    """
    cs = ClusterSpace(structure=primitive_structure,
                      cutoffs=cutoffs,
                      chemical_symbols=['In', 'Ga'],
                      position_tolerance=1e-03)
    
    sc = StructureContainer(cluster_space=cs)
    
    # Load structures from database
    with connect(db_path) as db:
        for row in db.select():
            sc.add_structure(structure=row.toatoms(),
                             user_tag=row.tag,
                             properties={'mixing_energy': row.mixing_energy})
                             
    return sc.get_fit_data(key='mixing_energy')

def train_ce(db_path: Path, primitive_structure, cutoffs: List[float]) -> Dict[str, Any]:
    """
    Trains a CE model with specific cutoffs and returns validation metrics.
    """
    A, y = get_fit_data(db_path, primitive_structure, cutoffs)
    
    cve = CrossValidationEstimator(fit_data=(A, y), 
                                   fit_method=FIT_METHOD,
                                   validation_method='shuffle-split',
                                   n_splits=NSPLITS)
    cve.validate()
    cve.train()
    
    return {
        'rmse_validation': cve.rmse_validation,
        'rmse_train': cve.rmse_train,
        'BIC': cve.model.BIC,
        'n_parameters': cve.n_parameters,
        'n_nonzero_parameters': cve.n_nonzero_parameters
    }

def plot_results(df: pd.DataFrame, x_col: str, title_suffix: str):
    """Generates diagnostic plots for RMSE and BIC."""
    fig, axes = plt.subplots(figsize=(4, 5.2), dpi=120, sharex=True, nrows=3)
    
    # RMSE Plot
    axes[0].plot(df[x_col], 1000 * df.rmse_validation, '-o', label='Validation')
    axes[0].plot(df[x_col], 1000 * df.rmse_train, '--s', label='Train')
    axes[0].set_ylabel('RMSE (meV/atom)')
    axes[0].legend()
    axes[0].set_title(f"Optimization: {title_suffix}")

    # BIC Plot
    axes[1].plot(df[x_col], df.BIC * 1e-3, '-o', color='tab:orange')
    axes[1].set_ylabel(r'BIC ($\times 10^{3}$)')

    # Parameters Plot
    axes[2].plot(df[x_col], df.n_parameters, '--s', label='Total')
    axes[2].plot(df[x_col], df.n_nonzero_parameters, '-o', label='Non-zero')
    axes[2].set_xlabel(f'{title_suffix} cutoff (Å)')
    axes[2].set_ylabel('Parameters')
    axes[2].legend()

    plt.tight_layout()
    plt.show()

def main():
    # Load primitive structure once
    with connect(DB_FILE) as db:
        primitive_structure = db.get(id=PRIMITIVE_ID).toatoms()

    # --- Step 1: Optimize Pair Cutoff (2nd order) ---
    print("Optimizing 2nd order cutoffs...")
    c2_vals = np.arange(3.0, 7.0, 0.5) # Example range
    records = []
    for c2 in c2_vals:
        stats = train_ce(DB_FILE, primitive_structure, [c2])
        records.append({'c2': c2, **stats})
    
    df2 = pd.DataFrame(records)
    plot_results(df2, 'c2', 'Pair')
    
    # Select best c2 (e.g., hardcoded based on plot analysis)
    c2_best = 6.0 
    
    # --- Step 2: Optimize Triplet Cutoff (3rd order) ---
    print(f"Optimizing 3rd order cutoffs (fixed pair={c2_best})...")
    c3_vals = np.arange(2.0, 5.0, 0.5)
    records = []
    for c3 in c3_vals:
        stats = train_ce(DB_FILE, primitive_structure, [c2_best, c3])
        records.append({'c2': c2_best, 'c3': c3, **stats})
        
    df3 = pd.DataFrame(records)
    plot_results(df3, 'c3', 'Triplet')

if __name__ == "__main__":
    main()

import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
from typing import Dict, List, Tuple
from pathlib import Path

# --- Constants for Neighbor Shells (Wurtzite specific) ---
# These need to be tuned based on your lattice constants (a ~ 3.19, c ~ 5.18)
CUTOFF_1ST_PLANE = 3.3   # Covers 1st neighbor in-plane (a)
CUTOFF_2ND_PLANE = 5.6   # Covers 2nd neighbor in-plane (sqrt(3)*a)
CUTOFF_VERTICAL = 5.3    # Covers c/2 and c neighbors

def calculate_warren_cowley_sro(atoms, shell_type='in-plane-1'):
    """
    Calculates the Warren-Cowley SRO parameter using ASE's optimized neighbor lists.
    
    Args:
        atoms (ASE.Atoms): The structure.
        shell_type (str): 'in-plane-1', 'in-plane-2', or 'vertical'.
        
    Returns:
        float: The SRO parameter alpha.
    """
    # 1. Get species counts
    chemical_symbols = np.array(atoms.get_chemical_symbols())
    n_In = np.sum(chemical_symbols == 'In')
    n_Ga = np.sum(chemical_symbols == 'Ga')
    concentration_Ga = n_Ga / len(atoms)
    
    if n_In == 0 or n_Ga == 0: return 0.0 # Pure case

    # 2. Compute Neighbor List (Optimized C-backend)
    # 'i' are indices of atoms, 'j' are indices of neighbors, 'd' are distances
    # 'D' are vector distances (dx, dy, dz)
    i_list, j_list, d_list, D_list = neighbor_list('ijdD', atoms, cutoff=CUTOFF_2ND_PLANE)
    
    # 3. Filter neighbors based on geometry (Shell definitions)
    valid_pairs = []
    
    for k in range(len(i_list)):
        idx_i = i_list[k]
        idx_j = j_list[k]
        dist = d_list[k]
        dz = abs(D_list[k][2]) # Vertical component of distance vector
        
        is_neighbor = False
        
        # Geometry Logic for Wurtzite (approximate checks)
        if shell_type == 'in-plane-1':
            # Same plane (small dz) and dist ~ a
            if dz < 0.5 and 3.0 < dist < 3.4: 
                is_neighbor = True
                
        elif shell_type == 'in-plane-2':
            # Same plane (small dz) and dist ~ sqrt(3)*a (~5.5)
            if dz < 0.5 and 5.3 < dist < 5.8:
                is_neighbor = True
                
        elif shell_type == 'vertical':
            # Next plane (dz ~ c/2 or c) - Check Thesis definition
            # Usually strict vertical is c ~ 5.18. 
            if 5.0 < dz < 5.4 and dist < 5.4:
                is_neighbor = True

        if is_neighbor:
            valid_pairs.append((idx_i, idx_j))
            
    # 4. Calculate SRO
    if not valid_pairs:
        return None
    
    # Count how many 'Ga' neighbors surround 'In' atoms
    ga_neighbors_of_in = 0
    total_neighbors_of_in = 0
    
    # Group by atom i to handle coordination number correctly
    from collections import defaultdict
    neighbors_map = defaultdict(list)
    for src, dst in valid_pairs:
        neighbors_map[src].append(dst)
        
    for idx_i, neighbors in neighbors_map.items():
        if chemical_symbols[idx_i] == 'In':
            total_neighbors_of_in += len(neighbors)
            for idx_j in neighbors:
                if chemical_symbols[idx_j] == 'Ga':
                    ga_neighbors_of_in += 1
                    
    if total_neighbors_of_in == 0: return None

    p_InGa = ga_neighbors_of_in / total_neighbors_of_in
    alpha = 1 - (p_InGa / concentration_Ga)
    
    return alpha

def main():
    # Example usage
    file_path = Path("simulations/conc_0.35/structure_700K.xyz")
    if not file_path.exists():
        print("Structure file not found.")
        return

    atoms = read(file_path)
    
    print(f"Analyzing {file_path}...")
    alpha_1 = calculate_warren_cowley_sro(atoms, 'in-plane-1')
    alpha_2 = calculate_warren_cowley_sro(atoms, 'in-plane-2')
    alpha_c = calculate_warren_cowley_sro(atoms, 'vertical')
    
    print(f"SRO alpha (1st in-plane): {alpha_1:.4f}")
    print(f"SRO alpha (2nd in-plane): {alpha_2:.4f}")
    print(f"SRO alpha (vertical):     {alpha_c:.4f}")

if __name__ == "__main__":
    main()

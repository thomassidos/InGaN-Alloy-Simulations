import numpy as np
from ase import Atoms
from ase.db import connect
from pathlib import Path
from typing import Tuple, List, Dict

# Constants
INPUT_FILE = Path('structures.in')
DB_FILE = Path('new.db')
E_TOT_GAN_PER_ATOM = -13.9413455  # Explicit value for clarity
E_TOT_INN_PER_ATOM = -11.7571335

def read_data_from_file(file_path: Path) -> Tuple[List, List, List, List]:
    """
    Parses the custom DFT output file.
    
    Args:
        file_path (Path): Path to the input file.
        
    Returns:
        Tuple containing lists of primitive vectors, coordinates, energies, and species counts.
    """
    prim_vecs, direct_cords, energies, species = [], [], [], []
    
    if not file_path.exists():
        raise FileNotFoundError(f"Input file {file_path} not found.")

    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line == 'Direct':
            # 1. Parse Species (line before 'Direct')
            try:
                spec_line = list(map(int, lines[i - 1].strip().split()))
                species.append(spec_line)
            except ValueError:
                print(f"Warning: Could not parse species at line {i-1}")
                i += 1
                continue

            # 2. Parse Primitive Vectors (4 lines above 'Direct')
            p_vec = []
            try:
                for k in range(i - 4, i - 1):
                    vec = list(map(float, lines[k].strip().split()))
                    p_vec.append(vec)
                prim_vecs.append(p_vec)
            except ValueError:
                print(f"Warning: Could not parse vectors around line {i}")

            # 3. Parse Atomic Positions
            d_cord = []
            j = i + 1
            while j < len(lines):
                parts = lines[j].strip().split()
                if len(parts) != 3:
                    break
                try:
                    coord = list(map(float, parts))
                    d_cord.append(coord)
                except ValueError:
                    break
                j += 1
            direct_cords.append(d_cord)
            
            # 4. Parse Energy (line after coordinates)
            if j < len(lines):
                try:
                    energies.append(float(lines[j].strip()))
                except ValueError:
                    print(f"Warning: Could not parse energy at line {j}")
            
            i = j # Move index forward
        else:
            i += 1
            
    return prim_vecs, direct_cords, energies, species

def main():
    print(f"Reading data from {INPUT_FILE}...")
    prim_vecs, direct_cords, energies, species = read_data_from_file(INPUT_FILE)
    
    print(f"Found {len(prim_vecs)} structures. Writing to {DB_FILE}...")
    
    with connect(DB_FILE) as db:
        for i in range(len(prim_vecs)):
            lattice_vectors = np.array(prim_vecs[i])
            # Ensure proper shape for dot product
            positions_frac = np.array(direct_cords[i])
            atomic_positions = np.dot(positions_frac, lattice_vectors)
            
            # Species mapping: species[i][0] is In, species[i][1] is Ga
            n_In = species[i][0]
            n_Ga = species[i][1]
            symbols = ['In'] * n_In + ['Ga'] * n_Ga
            
            atoms = Atoms(symbols=symbols, positions=atomic_positions, cell=lattice_vectors, pbc=True)
            
            # Calculate Mixing Energy
            n_total = n_In + n_Ga
            mixing_energy = (energies[i] - n_In * E_TOT_INN_PER_ATOM - n_Ga * E_TOT_GAN_PER_ATOM) / n_total
            concentration = n_In / n_total
            
            # Tag string for reference
            tag_string = "".join(symbols)
            
            db.write(atoms, tag=tag_string, mixing_energy=mixing_energy, 
                     concentration=concentration, lattice_parameter=1.0)
            
    print("Database creation complete.")

if __name__ == "__main__":
    main()

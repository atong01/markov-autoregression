import h5py
import mdtraj
import numpy as np
import tempfile
import os

with h5py.File('/mnt/labs/data/tong/mdCATH/data/mdcath_dataset_1wgxA00.h5', 'r') as f:
    domain = '1wgxA00'
    
    # Use pdbProteinAtoms (1089 atoms, matches coords)
    pdb_str = f[f'{domain}/pdbProteinAtoms'][()].decode('utf-8')
    
    # Write to temp file for mdtraj
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
        tmp.write(pdb_str)
        tmp_path = tmp.name
    
    try:
        topology = mdtraj.load_pdb(tmp_path).topology
    finally:
        os.unlink(tmp_path)
    
    # Load coordinates (Angstrom -> nm)
    coords = np.array(f[f'{domain}/320/0/coords']) / 10.0
    
    # Create trajectory
    traj = mdtraj.Trajectory(coords, topology)
    print(traj)
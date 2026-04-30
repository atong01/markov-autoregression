import h5py
import tempfile
import numpy as np
import argparse
import os
from multiprocessing import Pool
from tqdm import tqdm

MDCATH_PICOSECONDS_PER_FRAME = 1000.

def _open_h5_file(h5):
    if isinstance(h5, str):
        h5 = h5py.File(h5, "r")
    code = [_ for _ in h5][0]
    return h5, code


def _extract_structure_and_coordinates(h5, code, temp, replica):
    """
    Extracts the structure in PDB format and coordinates from an H5 file based on temperature and replica.

    Parameters:
    h5 : h5py.File
        An opened H5 file object containing protein structures and simulation data.
    code : str
        The identifier for the dataset in the H5 file.
    temp : int or float
        The temperature (in Kelvin).
    replica : int
        The replica number.

    Returns:
    tuple
        A tuple containing the PDB data as bytes, coordinates as a numpy array, and box as a numpy vector.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as pdbfile:
        pdb = h5[code]["pdbProteinAtoms"][()]
        pdbfile.write(pdb)
        pdbfile.flush()
        coords = h5[code][f"{temp}"][f"{replica}"]["coords"][:]
        box = h5[code][f"{temp}"][f"{replica}"]["box"][:]
    coords = coords / 10.0
    coords -= coords.mean(axis=1, keepdims=True)
    return pdbfile.name, coords, box


def convert_to_mdtraj(h5, temp, replica):
    """
    Convert data from an H5 file to an MDTraj trajectory object.

    This function extracts the first protein atom structure and coordinates
    for a given temperature and replica from an H5 file and creates an MDTraj
    trajectory object. This object can be used for further molecular dynamics
    analysis.

    Parameters:
    h5 : h5py.File
        An opened H5 file object containing protein structures and simulation data.
    temp : int or float
        The temperature (in Kelvin) at which the simulation was run. This is used
        to select the corresponding dataset within the H5 file.
    replica : int
        The replica number of the simulation to extract data from. This is used
        to select the corresponding dataset within the H5 file.

    Returns:
    md.Trajectory
        An MDTraj trajectory object containing the loaded protein structure and
        simulation coordinates.

    Example:
    -------
    import h5py
    import mdtraj as md

    # Open the H5 file
    with h5py.File('simulation_data.h5', 'r') as h5file:
        traj = convert_to_mdtraj(h5file, 300, 1)

    # Now 'traj' can be used for analysis with MDTraj
    """
    import mdtraj as md

    h5, code = _open_h5_file(h5)
    pdb_file_name, coords, box = _extract_structure_and_coordinates(h5, code, temp, replica)
    top = md.load(pdb_file_name).topology
    os.unlink(pdb_file_name)
    nframes = coords.shape[0]
    uc_lengths = np.repeat(box.diagonal()[None,:], nframes, axis=0)
    uc_angles =  np.repeat(np.array([90.,90.,90.])[None,:], nframes, axis=0)
    trj = md.Trajectory(coords.copy(), 
                        topology=top, 
                        time=np.arange(1, coords.shape[0] + 1)*MDCATH_PICOSECONDS_PER_FRAME,
                        unitcell_lengths = uc_lengths,
                        unitcell_angles = uc_angles
                        )
    return trj


def convert_to_moleculekit(h5, temp, replica):
    """
    Convert data from an H5 file to a MoleculeKit/HTMD trajectory object.

    This function extracts the first protein atom structure and coordinates
    for a given temperature and replica from an H5 file and creates an MDTraj
    trajectory object. This object can be used for further molecular dynamics
    analysis.

    Parameters:
    h5 : h5py.File
        An opened H5 file object containing protein structures and simulation data.
    temp : int or float
        The temperature (in Kelvin) at which the simulation was run. This is used
        to select the corresponding dataset within the H5 file.
    replica : int
        The replica number of the simulation to extract data from. This is used
        to select the corresponding dataset within the H5 file.

    Returns:
    moleculekit.molecule.Molecule
        A Molecule object containing the loaded protein structure and
        simulation coordinates.

    Example:
    -------
    import h5py
    import moleculekit as mk

    # Open the H5 file
    with h5py.File('simulation_data.h5', 'r') as h5file:
        traj = convert_to_moleculekit(h5file, 300, 1)

    # Now 'traj' can be used for analysis with HTMD
    """

    import moleculekit.molecule as mk

    h5, code = _open_h5_file(h5)
    pdb_file_name, coords, box = _extract_structure_and_coordinates(h5, code, temp, replica)
    trj = mk.Molecule(pdb_file_name, name=f"{code}_{temp}_{replica}")
    os.unlink(pdb_file_name)
    nframes = coords.shape[0]
    uc_lengths = np.repeat(box.diagonal()[None,:], nframes, axis=0)
    trj.coords = coords.transpose([1, 2, 0]).copy()
    trj.time = np.arange(1, coords.shape[0] + 1)
    trj.box = uc_lengths.T * 10.0

    # TODO? .step, .numframes
    return trj


def convert_to_files(
    fn, traj_dir, topo_dir, temp_list=[320, 348, 379, 413, 450], replica_list=[0, 1, 2, 3, 4]
):
    h5, code = _open_h5_file(fn)

    pdbpath = os.path.join(topo_dir, f"{code}.pdb")
    with open(pdbpath, "wb") as pdbfile:
        pdb = h5[code]["pdbProteinAtoms"][()]
        pdbfile.write(pdb)
        print(f"Wrote {pdbpath}")

    for temp in temp_list:
        for replica in replica_list:
            xtcpath = os.path.join(traj_dir, f"{code}_{temp}_{replica}.xtc")
            trj = convert_to_mdtraj(h5, temp, replica)
            trj.save_xtc(xtcpath)
            print(f"Wrote {xtcpath}")


def _convert_worker(args):
    fpath, traj_dir, topo_dir, temp_list, replica_list = args
    print(f"Processing {fpath}")
    convert_to_files(fpath, traj_dir, topo_dir, temp_list, replica_list)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a directory of H5 files to PDB and XTC files."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing H5 files to convert.",
    )
    parser.add_argument(
        "out_dir",
        type=str,
        help="Output directory. XTC files go into <out_dir>/trajectory/, PDB files into <out_dir>/topology/.",
    )
    parser.add_argument(
        "--temp_list",
        type=int,
        nargs="+",
        help="List of temperatures.",
        default=[320, 348, 379, 413, 450],
    )
    parser.add_argument(
        "--replica_list",
        type=int,
        nargs="+",
        help="List of replicas.",
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel worker processes.",
    )

    args = parser.parse_args()

    traj_dir = os.path.join(args.out_dir, "trajectory")
    topo_dir = os.path.join(args.out_dir, "topology")
    os.makedirs(traj_dir, exist_ok=True)
    os.makedirs(topo_dir, exist_ok=True)

    h5_files = sorted(
        f for f in os.listdir(args.input_dir) if f.endswith(".h5") or f.endswith(".hdf5")
    )
    if not h5_files:
        print(f"No H5 files found in {args.input_dir}")
        return

    work_items = [
        (os.path.join(args.input_dir, fname), traj_dir, topo_dir, args.temp_list, args.replica_list)
        for fname in h5_files
    ]

    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            list(tqdm(pool.imap_unordered(_convert_worker, work_items), total=len(work_items), unit="file"))
    else:
        for item in tqdm(work_items, unit="file"):
            _convert_worker(item)


if __name__ == "__main__":
    main()
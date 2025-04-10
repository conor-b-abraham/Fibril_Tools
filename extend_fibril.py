import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align, distances
import argparse
import os
import sys
from tqdm import tqdm
from datetime import datetime

# ------------------------------------------------------------------------------
# INPUT
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Extend PDB fibril')
parser.add_argument('-i', '--inputfile', required=True, help='Input PDB File (after running correct_pdb.py)')
parser.add_argument('-l', '--n_layers', type=int, required=True, help='Number of layers for output pdb')
parser.add_argument('-o', '--outputfile', required=True, help='Output PDB File')
args = parser.parse_args()
IN = args.inputfile
OUT = args.outputfile
NLAYERS = args.n_layers

if IN[-4:] != '.pdb':
    print(f'INPUT ERROR: file specified by --inputfile/-i must be a pdb, so {IN} is not a valid input file.\n')
    exit=True
elif not os.path.isfile(IN):
    print(f'INPUT ERROR: {IN} does not exist.\n')
    exit=True
else:
    exit=False
if exit:
    parser.print_help()
    sys.exit('\nPlease try again with valid input file name')
else:
    print(f'\nCONSTRUCTING A {NLAYERS} LAYER FIBRIL FROM {IN}')

# ------------------------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------------------------
def add_layer(u, n_protofils):
    n_layers = int(u.segments.n_segments/n_protofils)
    # COORDINATES
    # each array is of shape (6, N_atoms/segment, 3) where axis 0 is the segid, axis 1 is the
    # atom, and axis 2 has the coordinates x,y,z
    xyzs = [np.array([u.select_atoms(f'segid P{p}{lndx}').positions for lndx in range(1, n_layers+1)]) for p in range(1, n_protofils+1)]


    # POSITIONS of NEW LAYER
    # P_new = 1/(N_l-2)[(2N_l-3)P_last - (N_l-1)P_{last-1} + P_1 - P_2]
    newxyzs = [(1/(n_layers-2))*((2*n_layers-3)*xyzs[p][-1,:,:]-(n_layers-1)*xyzs[p][-2,:,:]+xyzs[p][0,:,:]-xyzs[p][1,:,:]) for p in range(n_protofils)]

    # CREATE NEW LAYER
    Nlayeru = mda.Merge(*[u.select_atoms(f'segid P{p}{n_layers}') for p in range(1, n_protofils+1)])
    Nlayeru.segments.segids = np.array([f'P{p}{n_layers+1}' for p in range(1, n_protofils+1)]) # rename segments

    # CREATE GHOST LAYER
    # Has coordinates matching the guessed coordinates
    Glayeru = Nlayeru.copy()
    for p in range(1, n_protofils+1):
        Glayeru.select_atoms(f'segid P{p}{n_layers+1}').positions = newxyzs[p-1]

    # ALIGN NEW LAYER TO GHOST LAYER
    align.alignto(Nlayeru, Glayeru, select="segid "+" ".join([f"P{p}{n_layers+1}" for p in range(1, n_protofils+1)]), weights="mass")

    # ADD NEW LAYER TO FULL UNIVERSE
    u = mda.Merge(u.atoms, Nlayeru.atoms)
    return u

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
print('\n----------------------------IGNORE THIS WARNING----------------------------')
u = mda.Universe(IN)
print('----------------------------------------------------------------------------\n')

# find the number of layers and protofilaments in the input file
input_segids = u.segments.segids
n_layers_IN = np.unique([i[2:] for i in input_segids]).size
n_protofils = np.unique([i[1] for i in input_segids]).size
IN_atoms_per_segment = int(u.atoms.n_atoms/u.segments.n_segments)
print('INPUT DETAILS:')
print(f'{"Number of layers:":>30} {n_layers_IN}')
print(f'{"Number of protofilaments:":>30} {n_protofils}')
print(f'{"Atoms per segment:":>30} {IN_atoms_per_segment}')
print(f'{"Residues per segment:":>30} {int(u.residues.n_residues/u.segments.n_segments)}\n')


# MAIN
n_layers_toadd = NLAYERS-n_layers_IN
for _ in range(n_layers_toadd):
    u = add_layer(u, n_protofils)
print(f'{n_layers_toadd} layers were added to the fibril')

segment_order = []
for lndx in range(1, NLAYERS+1):
    for pfndx in range(1, n_protofils+1):
        segment_order.append(u.select_atoms(f'segid P{pfndx}{lndx}'))
output_universe = mda.Merge(*segment_order)

# chain = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
for i, s in enumerate(output_universe.segments):
    s.atoms.chainIDs = [str(i)]*s.atoms.n_atoms

OUT_atoms_per_segment = output_universe.select_atoms('protein').n_atoms/(NLAYERS*n_protofils)
if IN_atoms_per_segment != OUT_atoms_per_segment:
    print(f'\nAtoms per segment in output does not match atoms per segment:')
    print(f'                                      INPUT = {IN_atoms_per_segment}')
    print(f'                                      INPUT = {OUT_atoms_per_segment}')

print('\n--------------------------IGNORE THESE WARNINGS---------------------------')
output_universe.atoms.write(OUT)
u = mda.Universe(OUT)
print('----------------------------------------------------------------------------\n')

print('\nOUTPUT DETAILS:')
if n_protofils > 1:
    print(f'{"SIDE":>5}{"LAYER":>6}{"SEGID":>6}{"CHAINID":>8}{"NATOMS":>7}{"NRES":>5}{"LAYER+1 DISTANCE":>20}{"INTERSIDE DISTANCE":>20}')
else:
    print(f'{"SIDE":>5}{"LAYER":>6}{"SEGID":>6}{"CHAINID":>8}{"NATOMS":>7}{"NRES":>5}{"LAYER+1 DISTANCE":>20}')

pfs = [u.select_atoms('segid '+' '.join([f'P{p}{l}' for l in range(1, NLAYERS+1)])) for p in range(1, n_protofils+1)]

pfsegs = [[seg for seg in pfs[p].segments] for p in range(n_protofils)]
for pi, segs in enumerate(pfsegs):
    for i, seg in enumerate(segs):
        segid = seg.segid
        chainID = np.unique(seg.atoms.chainIDs)
        if len(chainID) > 1:
            sys.exit(f"ERROR: Segment {segid} has multiple chainIDs: {' '.join(chainID.tolist())}")
        else:
            chainID = chainID[0]
        if i != len(segs)-1:
            ILdist = np.round(np.min(distances.distance_array(seg.atoms.positions, segs[i+1].atoms.positions)), 3)
        else:
            ILdist = "--"
        if n_protofils > 1:
            otherpfs = u.select_atoms('segid '+' '.join([' '.join([f'P{p}{l}' for l in range(1, NLAYERS+1)]) for p in range(1, n_protofils+1) if p != pi+1]))
            ISdist = np.round(np.min(distances.distance_array(seg.atoms.positions, otherpfs.atoms.positions)), 3)
            print(f'{pi+1:>5}{i+1:>6}{segid:>6}{chainID:>8}{seg.atoms.n_atoms:>7}{seg.residues.n_residues:>5}{ILdist:>20}{ISdist:>20}')
        else:
            print(f'{pi+1:>5}{i+1:>6}{segid:>6}{chainID:>8}{seg.atoms.n_atoms:>7}{seg.residues.n_residues:>5}{ILdist:>20}')

print(f'Extended fibril written to: {OUT}\n')

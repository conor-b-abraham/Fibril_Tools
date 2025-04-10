import MDAnalysis as mda
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from skspatial.objects import Plane
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import os
import sys
import argparse

# ------------------------------------------------------------------------------
# INPUT
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='FIND INTERNAL WATERS IN SAA FIBRIL')
INPUT_FILES = parser.add_argument_group('INPUT')
INPUT_FILES.add_argument('-t','--trajectory', required=True, help='(REQUIRED) Trajectory file containing coordinate information (e.g. XTC, TRR, DCD)')
INPUT_FILES.add_argument('-s', '--structure', default='system.pdb', help='(OPTIONAL; default=system.pdb) Structure/topology file containing segids (e.g. PDB)')
OUTPUT_FILES = parser.add_argument_group('OUTPUT')
OUTPUT_FILES.add_argument('-o', '--output_directory', default=f'internal_water', help='(OPTIONAL; default=internal_water) Directory to which to write results. Directory will be created if it does not already exist.')
OPTIONS = parser.add_argument_group('OPTIONS')
OPTIONS.add_argument('-n', '--n_protofilaments', required=True, type=int, help='(REQUIRED) The number of protofilaments in the fibril.')
OPTIONS.add_argument('-l', '--layers_to_omit', default=1, type=int, help='(OPTIONAL; default=1) Number of layers to ignore at the ends of the fibril. This will reduce artifacts due to delamination at the ends of the fibril.')
OPTIONS.add_argument('-v', '--write_vmd', default=-1, type=int, help='(OPTIONAL; default=-1) Write a vmd visualization state for the internal waters at this frame. -1 turns this option off')
args = parser.parse_args()

TRAJ = args.trajectory
STRUC = args.structure
OUTDIR = args.output_directory
N_PF = args.n_protofilaments
OMIT = args.layers_to_omit
VMD = args.write_vmd

# get rid of slash at end of output directory name if it is there
if OUTDIR[-1] == '/':
    OUTDIR = OUTDIR[:-1]

# check to make sure input files exist
inputerror = False
if not os.path.isfile(TRAJ):
    print(f'INPUT ERROR: {TRAJ} file not found.')
    inputerror = True
if not os.path.isfile(STRUC):
    print(f'INPUT ERROR: {STRUC} file not found.')
    inputerror = True
if inputerror:
    parser.print_help()
    sys.exit()

# dictionary of files that will be written
OUT = {
'SELIN1':f'{OUTDIR}/select_H2Oin1.npy',
'SELIN2':f'{OUTDIR}/select_H2Oin2.npy',
'SELOUT':f'{OUTDIR}/select_H2Oout.npy'
}

if VMD != -1:
    OUT['VMD'] = f'{OUTDIR}/view_frame{VMD}.vmd'
    OUT['VMDPDB'] = f'{OUTDIR}/view_frame{VMD}.pdb'

# Make sure output directory exists and files to be written don't overwrite other files
if os.path.isdir(OUTDIR):
    file_status = [] # 0 means file in OUT does not exist, 1 means it exists and can be backed up, 2 means backup already exists and will be replaced by new backup
    for file in OUT.values():
        if os.path.isfile(file):
            prevfile = file.replace(f'{OUTDIR}/', f'{OUTDIR}/prev_')
            if os.path.isfile(prevfile):
                print(f'WARNING: {file} and {prevfile} already exist. If you proceed {prevfile} will be permenantly deleted.')
                file_status.append(2) # backup and file exists
            else:
                file_status.append(1) # file exists but backup does not
        else:
            file_status.append(0) # no file exists in output directory
    print(' ')
    if 2 in file_status:
        if input('Would you like to kill the program to avoid overwriting backups? (y/n)') == 'y':
            sys.exit('\nUSER ABORT: Backups could not be made safely')
    else:
        for i, file in enumerate(OUT.values()):
            prevfile = file.replace(f'{OUTDIR}/', f'{OUTDIR}/prev_')
            if file_status[i] == 1:
                shutil.copy2(file, prevfile)
                print(f'MOVED: {file} --> {prevfile}')
            elif 1 in file_status:
                print(f'WARNING: {file} did not already exist, but other files are being backed up. If {prevfile} already exists, it will not match the backups from this run.')
else:
    os.makedirs(OUTDIR)

print(f'Results will be saved to {OUTDIR}/\n')

# ------------------------------------------------------------------------------
# FUNCTIONS & CLASSES
# ------------------------------------------------------------------------------
def segid_array(ag, N_PF, OMIT_LAYERS):
    '''
    Create (n_layers, n_protofilaments) array of segment IDs

    Parameters
    ----------
    ag : MDAnalysis.atomgroup
        Atom group of the fibril
    N_PF : Int
        The number of protofilaments in the fibril
    OMIT_LAYERS : Int
        The number of layers to omit from each end of the fibril

    Returns
    -------
    segids : numpy.ndarray
        The (n_layers, n_protofilaments) array of segment IDs
    '''
    all_segids = [ag.residues.segids[i] for i in sorted(np.unique(ag.residues.segids, return_index=True)[1])]
    pf_segids = []
    for pf in range(N_PF):
        pf_segids.append(all_segids[pf::N_PF])
    pf_segids = np.array(pf_segids).T[OMIT_LAYERS:-OMIT_LAYERS, :]
    if N_PF == 1: # Force to be 2d so single protofilament case will behave like multiple protofilament case
        pf_segids = np.atleast_2d(pf_segids)
    return pf_segids

def make_ags(system, segids_side):
    wholeside_sel = f'segid {" ".join(segids_side)}' # all layers on the side of the fibril
    subsystems = [] # all atoms to be considered in an iteration (all layers of fibril other than those omitted and water oxygen atoms within 10 Angstroms of layer of interest)
    mainlayers = [] # the main layer (by which the system will be fit) in an iteration
    otherlayers = [] # layers within 3 layers of the main layer
    waters = [] # all waters to consider in an iteration (within 10 angstroms of main layer)
    sidetops = [] # top layer of protofilament (not including layers that are omitted)
    sidebottoms = [] # bottom layer of protofilament (not including layers that are omitted)
    for i, layer in enumerate(segids_side):
        layer_sel = f'segid {layer} and name CA and resid 22-50'
        if i-3 < 0:
            below = " ".join(segids_side[:i])
        else:
            below = " ".join(segids_side[i-3:i])
        if i+4 >= len(segids_side):
            notbelow = " ".join(segids_side[i:])
        else:
            notbelow = " ".join(segids_side[i:i+4])
        alllayer_sel = f'segid {below} {notbelow} and name CA and resid 22-50'
        water_sel = f'resname TIP3 and name OH2 and around 10 segid {layer}'
        subsystems.append(system.select_atoms(f'({water_sel}) or ({wholeside_sel})', updating=True))
        small_otherlayers = []
        otherlayer_ag = subsystems[-1].select_atoms(f'({alllayer_sel}) and not ({layer_sel})')
        for segid in otherlayer_ag.segments.segids:
            small_otherlayers.append(otherlayer_ag.select_atoms(f'segid {segid} and resid 22-50 and name CA'))
        otherlayers.append(small_otherlayers)
        mainlayers.append(subsystems[-1].select_atoms(layer_sel))
        sidetops.append(subsystems[-1].select_atoms(f'segid {segids_side[-1]}'))
        sidebottoms.append(subsystems[-1].select_atoms(f'segid {segids_side[0]}'))
        waters.append(subsystems[-1].select_atoms('resname TIP3', updating=True))
    return subsystems, mainlayers, otherlayers, sidetops, sidebottoms, waters

def find_normal(X):
    # FIT PLANE TO COORDINATES (X)
    plane = Plane.best_fit(X)
    N = plane.normal
    return N

def find_internal_waters(subsystems, mainlayers, otherlayers, sidetops, sidebottoms, waters):
    # for each layer (protein) we need to...
    # 1) find the normal to the polygons (the CA atoms of res. 22-50 in all 24 SAA proteins)
    # 2) rotate the system s.t. the CA atoms of the SAA protein are in the plane of the normal
    # 3) make the polygon from the CA atoms of the main SAA protein and the layers 3 above and 3 below that layer.
    #    For each water molecule, check if it is inside or outside the layer that it is closest to
    # 4) collect the indices of the water Oxygen atoms that fall within the polygon at the layer thickness
    #
    # 5) repeat for all layers
    # 6) collect the unique indices of the waters that are inside the fibril (this is returned)

    inside_water_indices = []
    for sndx, subsystem in enumerate(subsystems):
        mainlayer = mainlayers[sndx]
        otherlayer = otherlayers[sndx]
        water = waters[sndx]
        sidetop = sidetops[sndx]
        sidebottom = sidebottoms[sndx]

        # save subsystem positions so they can be reset before moving to the next iteration
        original_subsystem_positions = subsystem.positions

        # 1) Find the normal to the layer of interest's alpha carbon plane of best fit
        # find the normal vector of the subsystem atoms we select to represent the channel of the fibril
        mainlayer_coords = mainlayer.positions
        Nvec = find_normal(mainlayer_coords)

        # 2) Rotate the system so that the layer of interest is oriented in the direction of the x-axis
        subsystem_coords = original_subsystem_positions.copy()
        centered_subsystem_coords = subsystem_coords - np.mean(mainlayer_coords, axis=0)

        # find the rotation matrix, R,  that will align Nvec to Nref, <1, 0, 0>
        Nref = np.array([1,0,0])
        v = np.cross(Nvec,Nref)
        c = np.dot(Nvec,Nref)
        s = np.linalg.norm(v)
        I = np.identity(3)
        k = np.matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = I + k + np.matmul(k,k) * ((1 -c)/(s**2))

        # rotate the coordinates, then set the positions in the subsystem group to these transformed coordinates
        rotated_coords = R.dot(centered_subsystem_coords.T).T
        subsystem.positions = rotated_coords

        # 3 & 4)
        # for each layer in alllayers
        # create a 2D polygon from the protein CA atoms and determine COM
        polygon_list = [Polygon(list(map(tuple, mainlayer.atoms.positions[:,1:3])))]
        segid_com_list = [mainlayer.atoms.center_of_mass()]
        for layer in otherlayer:
            polygon_list.append(Polygon(list(map(tuple, layer.atoms.positions[:,1:3]))))
            segid_com_list.append(layer.atoms.center_of_mass())

        # Check for errors
        for pindex, polygon in enumerate(polygon_list):
            if not polygon.is_valid:
                print(f"ERROR: Coordinates provided for segid {np.unique(mainlayer.segments.segids)[0]}, polygon {pindex+1} do not determine a valid geometric object. Area of created object is {polygon.area}")
                if input('Do you want to plot the polygon? (y/n): ').lower() != 'y':
                    if input('Do you want to kill the program? (y/n): ').lower() == 'y':
                        sys.exit('User Abort')
                else:
                    xpoly, ypoly = polygon.exterior.xy
                    print(len(xpoly))
                    plt.plot(xpoly, ypoly)
                    for candx, (x, y) in enumerate(zip(xpoly, ypoly)):
                        plt.scatter(x, y, marker=f"${candx+1}$", s=100, c='k')
                    plt.show()
                    if input('Do you want to kill the program? (y/n): ').lower() == 'y':
                        sys.exit('User Abort')

        # Identify internal water indices
        stcom = sidetop.center_of_mass()[0]
        sbcom = sidebottom.center_of_mass()[0]
        if stcom > sbcom:
            topcut = stcom+5
            bottomcut = sbcom-5
        else:
            topcut = sbcom+5
            bottomcut = stcom-5

        new_inside_indices = []
        for i in range(water.n_atoms):
            atom_position = water.positions[i][0] # x position of atom
            if atom_position <= topcut and atom_position >= bottomcut:
                relative_distance_list = [abs(atom_position-pcom[0]) for pcom in segid_com_list] # find the distance between the water and each segments center of mass along the x coordinate
                closest_segid_index = relative_distance_list.index(min(relative_distance_list)) # choose the layer whose center of mass is closest to the water
                if polygon_list[closest_segid_index].contains(Point(water.positions[i][1:3])): # if the water is within the chosen layer
                    if polygon_list[0].contains(Point(water.positions[i][1:3])): # if the water is also within the layer of interest
                        new_inside_indices.append(water.atoms[i].index)
        inside_water_indices += new_inside_indices

        # reset subsystem positions
        subsystem.positions = original_subsystem_positions

    # prune any potentially double-counted water indices
    inside_water_indices = np.unique(inside_water_indices)
    return inside_water_indices

class visualization_state:
    '''
    Write a VMD visualization state.
    '''
    def __init__(self, topology_file, trajectory_file, frame, outpdb):
        self.topology_file = topology_file
        self.trajectory_file = trajectory_file
        self.outpdb = outpdb
        self.frame = frame
        self.universe = mda.Universe(self.topology_file, self.trajectory_file)
        self.universe.trajectory[self.frame]
        self.state = ['# Written with mda2vmd',
                     f'mol new {self.outpdb} type pdb first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all',
                      'mol delrep 0 top\n']

    def add_representation(self, selection, style, color):
        # set color
        default_colors = {'blue':0,'red':1,'gray':2,'grey':2,'orange':3,'yellow':4,'tan':5,'silver':6,
                          'green':7,'white':8,'pink':9,'cyan':10,'purple':11,'lime':12,'mauve':13,'ochre':14,
                          'iceblue':15,'black':16,'yellow2':17,'yellow3':18,'green2':19,'green3':20,'cyan2':21,
                          'cyan3':22,'blue2':23,'blue3':24,'violet':25,'violet2':26,'magenta':27,'magenta2':28,
                          'red2':29,'red3':30,'orange2':31,'orange3':32}
        color = f'ColorID {default_colors[color.lower()]}'

        # set style
        styles = {'licorice':'Licorice 0.300000 12.000000 12.000000',
                  'cpk':'CPK 0.500000 0.300000 12.000000 12.000000',
                  'vdw':'VDW 1.000000 12.000000',
                  'newcartoon':'NewCartoon 0.300000 10.000000 4.100000 0'}
        style = styles[style.lower()]

        # create atomgroup from selection
        indices = self.universe.select_atoms(selection).atoms.indices.tolist()
        indices.sort()
        if indices == list(range(indices[0], indices[-1]+1)):
            selection_command = '{'+f'index {indices[0]} to {indices[-1]}'+'}'
        else:
            selection_command = '{'+f'index {" ".join([str(i) for i in indices])}'+'}'

        # add to state
        self.state.append(f'mol representation {style}')
        self.state.append(f'mol color {color}')
        self.state.append(f'mol selection {selection_command}')
        self.state.append('mol material Opaque')
        self.state.append('mol addrep top\n')

    def write(self, filename):
        self.state.append(f'mol rename top {self.outpdb}')
        with open(filename, 'w+') as io:
            for line in self.state:
                io.write(f'{line}\n')
        self.universe.atoms.write(self.outpdb)

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
# CREATE UNIVERSE AND ATOM GROUPS
print('INITIALIZING')
u = mda.Universe(STRUC, TRAJ)

# Correct protein resids (Matching residues in fibril segments need to have the same index)
new_resids = []
for seg in u.select_atoms("protein").segments:
    new_resids += list(range(1, seg.residues.n_residues+1))
new_resids += u.select_atoms("not protein").residues.resids.tolist()
u.residues.resids = np.array(new_resids)

# Create Fibril Atom Group and Check INPUT option --layers_to_omit
fibril = u.select_atoms('protein')
n_layers = int(fibril.segments.n_segments/N_PF)
segids = segid_array(fibril, N_PF, OMIT)
print(f"\nSEGMENTS TO BE ANALYZED:")
for pf in range(N_PF):
    pf_string = " ".join(segids[:, pf].tolist())
    print(f"Protofilament {pf+1}: {pf_string}")

if n_layers/2 <= OMIT: # You cannot choose to ignore all layers of the fibril
    print(f'INPUT ERROR: You chose to ignore {OMIT} layers on each end of the fibril and the fibril only has {n_layers} layers. You must choose a value < {n_layers/2} (i.e. n_layers/2).')
    parser.print_help()
    sys.exit()

# Create atom groups for analysis
allwater = u.select_atoms('resname TIP3')
allwater_oxygen = allwater.select_atoms('name OH2')
system = u.select_atoms('(protein and resid 22-50) or (resname TIP3 and name OH2)') # Only look at hydrophilic channels and water

subsystems1, mainlayers1, otherlayers1, sidetops1, sidebottoms1, waters1 = make_ags(system, segids[:,0])
subsystems2, mainlayers2, otherlayers2, sidetops2, sidebottoms2, waters2 = make_ags(system, segids[:,1])
print(f'Trajectory length: {u.trajectory.n_frames}\n')

# FIND THE INTERNAL WATERS
print('FINDING INTERNAL WATERS')
selection1 = np.zeros((u.trajectory.n_frames, allwater_oxygen.n_atoms), dtype=bool)
selection2 = np.zeros((u.trajectory.n_frames, allwater_oxygen.n_atoms), dtype=bool)
external = np.zeros((u.trajectory.n_frames, allwater_oxygen.n_atoms), dtype=bool)
for ts in tqdm(u.trajectory):
    # for each side of the fibril, collect the indices of the internal waters and cast them into a boolean numpy array that can be used to select them
    # from a indexable object containing all atoms in the system.
    selection1[ts.frame, :] = np.isin(allwater_oxygen.indices, find_internal_waters(subsystems1, mainlayers1, otherlayers1, sidetops1, sidebottoms1, waters1))
    selection2[ts.frame, :] = np.isin(allwater_oxygen.indices, find_internal_waters(subsystems2, mainlayers2, otherlayers2, sidetops2, sidebottoms2, waters2))
    # create a similar boolean array for external waters
    external[ts.frame, :] = np.logical_not(np.logical_or(selection1[ts.frame, :], selection2[ts.frame, :]))

print('\nSAVING RESULTS')
np.save(OUT['SELIN1'], selection1)
print(f'Internal waters selection file for side 1 written to {OUT["SELIN1"]}')
np.save(OUT['SELIN2'], selection2)
print(f'Internal waters selection file for side 2 written to {OUT["SELIN2"]}')
np.save(OUT['SELOUT'], external)
print(f'External waters selection file written to {OUT["SELOUT"]}')

print(f'''
These selection files contain numpy boolean arrays that can be used to make new atom groups
from an atomgroup containing all of the water OH2 atoms.

For example:
u = mda.Universe({STRUC}, {TRAJ})
water_oxygen = u.select_atoms("resname TIP3 and name OH2")
internal1 = np.load({OUT["SELIN1"]})
internal2 = np.load({OUT["SELIN2"]})
external = np.load({OUT["SELOUT"]})
for ts in u.trajectory:
        internal1_atomgroup = water_oxygen.atoms[internal1[ts.frame, :]] # selects water OH2 atoms inside side one at current frame
        internal2_atomgroup = water_oxygen.atoms[internal2[ts.frame, :]] # selects water OH2 atoms inside side two at current frame
        external_atomgroup = water_oxygen.atoms[external[ts.frame, :]] # selects water OH2 atoms outside fibril at current frame

''')

if VMD != -1:
    if VMD >= u.trajectory.n_frames:
        print(f'WARNING: Cannot write frame {VMD} for trajectory of length {u.trajectory.n_frames}. Only frames 0 to {u.trajectory.n_frames-1} can be written.',
              f'         The last frame of the trajectory ({u.trajectory.n_frames-1}) will be written instead.')
        VMD = int(u.trajectory.n_frames-1)
    print('MAKING VMD FILE')
    v = visualization_state(STRUC, TRAJ, VMD, OUT['VMDPDB'])
    v.add_representation('protein', 'NewCartoon', 'gray')
    v.add_representation('resname TIP3 and index '+' '.join([str(i) for i in allwater_oxygen.indices[selection1[VMD, :]]]), 'vdw', 'blue2')
    v.add_representation('resname TIP3 and index '+' '.join([str(i) for i in allwater_oxygen.indices[selection2[VMD, :]]]), 'vdw', 'red2')
    v.add_representation('resname TIP3 and name OH2', 'cpk', 'white')
    v.write(OUT['VMD'])
    print(f'DONE: vmd visualization state of frame {VMD} written to {OUT["VMD"]}',
          f'      Open in vmd with >> vmd -e {OUT["VMD"]}')

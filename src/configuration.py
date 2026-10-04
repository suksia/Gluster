from copy import copy, deepcopy
from vasp_file import *
import numpy as np
from pathlib import Path
import subprocess, shutil, time
from ase.io import read
from math import floor

ELEMENTS = {
    'W': {'mass': 183.84, 'a0': 3.165},
    'Mo': {'mass': 95.95, 'a0': 3.147},
    'V': {'mass': 50.94, 'a0': 3.027}}

class Configuration(object):
    def __init__(self, cid: int, cdir: Path, conf_dict: dict):
        self.cid = cid
        self.cdir = cdir

        self.name = self.conf_dict['name']
        if self.name is None:
            self.name = cid

        self.transform: np.ndarray = conf_dict['transform']
        self.basis = conf_dict['basis']
        self.size = conf_dict['size']
        self.comp: dict[str, int] = conf_dict['comp']
        self.seed = conf_dict['seed']
        self.lattice = None
        self.a0 = None

        self.incar: VaspIncar = conf_dict['incar']
        self.poscar: VaspPoscar = conf_dict['poscar']
        self.kpoints: VaspKpoints = conf_dict['kpoints']
        self.potcar: VaspPotcar = conf_dict['potcar']

        self.run = conf_dict['run']
        self.ncores = conf_dict['ncores']
        self.include_fit = conf_dict['include_fit']
        self.energy = None
        self.mixing_energy = None
        self.Atoms = None
        self.status = 0
        self.process = None
        self.start_time, self.process_time = None, None
        
        if self.load_energy():
            self.status = 2
        if self.run is None or self.run is False:
            pass
        elif self.run is True:
            self.status = 0

        if self.poscar is None:
            self._create_chemical_basis()
        else:
            self.comp = {sp: cnt for sp, cnt in zip(self.poscar.species, self.poscar.ion_counts)}

    def _create_chemical_basis(self):
        if self.transform is None:
            raise ValueError(f"({self.cid}) Transformation matrix must be provided")
        else:
            self._check_transformation_matrix()

        # enumerate possible basis positions
        transform_inv = np.linalg.inv(self.transform)
        num_basis_points = round(abs(np.linalg.det(self.transform)))
        
        frac_positions = []
        for en_vector in list(np.ndindex(*[num_basis_points]*3)):
            f = transform_inv @ en_vector
            f_reduced = f % 1.0
            frac_positions.append(f_reduced)

        # filter out duplicates
        basis_points = []
        for f in frac_positions:
            if not any(np.allclose(f, x) for x in basis_points):
                basis_points.append(tuple([float(np.round(fp, 5)) for fp in f]))
            if len(basis_points) == num_basis_points:
                break

        # write basis points to a file
        A_prim = 1/2*np.array([[ 1, -1,  1], 
                                [ 1,  1, -1],
                                [-1,  1,  1]])
        self.A_chem = A_prim @ self.transform

        with open(self.cdir / 'basis.out', 'w') as f:
            f.write('Chemical lattice vectors: A(chem) = A(prim) @ P, cols, a0=1\n')
            for i in range(3):
                lv = np.round(self.A_chem[:,i], 6)
                f.write(f"{lv[0]:2.6f}\t{lv[1]:2.6f}\t{lv[2]:2.6f}\n")
            f.write('\nChemical basis positions: b(chem) = Pinv @ n mod 1, frac\n')
            for bp in basis_points:
                f.write(f"{bp[0]:2.6f}\t{bp[1]:2.6f}\t{bp[2]:2.6f}\n")
            f.write('\nChemical lattice positions: r(chem) = A(chem) @ b(chem)\n')
            for bp in basis_points:
                bp = [float(v) for v in self.A_chem @ np.array(bp)]
                f.write(f"{bp[0]:2.6f}\t{bp[1]:2.6f}\t{bp[2]:2.6f}\n")
            
        # compare computed basis positions with provided ones
        self.chemical_basis = {f: None for f in basis_points}
        if self.basis:
            if len(self.basis) != len(self.chemical_basis):
                raise ValueError(f"({self.cid}) Insufficient number of basis points provided. Expected {len(self.chemical_basis)}, got {len(self.basis)}")
            for bp_user, chem_map in self.basis.items():
                bp_found = False
                for bp in basis_points:
                    if np.allclose(np.array(bp_user), np.array(bp), atol=1e-2):
                        bp_found = bp
                        break
                if bp_found is False:
                    raise ValueError(f"({self.cid}) Basis atom at {bp_user} does not match the specified transform matrix. Enumerated positions: {basis_points}")
                else:
                    # make sure chemical mapping is dictionary of probabilities
                    if type(chem_map) == str:
                        chem_map = {chem_map: 1.0}
                    elif type(chem_map) == dict:
                        if sum(chem_map.values()) != 1.0:
                            raise ValueError(f"({self.cid}) Occupancy probabilities do not sum to 1 for basis atom {bp_user}")
                    self.chemical_basis[bp_found] = chem_map
        else:
            return # basis is not fully defined, so we are done
        
        # with the chemical basis, we can decorate a supercell
        self._create_supercell()

    def _create_supercell(self):
        # need these values to define the lattice
        if self.comp is None:
            raise ValueError(f"[{self.cid}] Chemical basis was defined, but missing composition")
        if self.size is None:
            raise ValueError(f"[{self.cid}] Chemical basis was defined, but missing supercell size")

        # define lattice constant as average
        self.a0 = 0.0
        num_atoms_in_comp = sum(self.comp.values()) 
        for sp, num in self.comp.items():
            self.a0 += ELEMENTS[sp]['a0'] * num / num_atoms_in_comp
        
        self.A_chem *= self.a0
        self.A_chem_inv = np.linalg.inv(self.A_chem)

        # replicate conventional lattice
        A_super = self.a0*np.array([[1, 0, 0], 
                                    [0, 1, 0], 
                                    [0, 0, 1]])
        basis_super = self.a0*np.array([[0]*3, [0.5]*3])
        super_positions = []
        for bp in basis_super:
            for tr_vector in list(np.ndindex(*self.size)):
                super_positions.append(bp + A_super @ tr_vector)

        num_atoms_in_super = len(super_positions)
        if num_atoms_in_comp != num_atoms_in_super:
            raise ValueError(f"({self.cid}) Number of atoms in composition ({num_atoms_in_comp}) does not match number of atoms in supercell ({num_atoms_in_super})")
        
        # assign lattice points to their corresponding basis point in the chemical basis
        lattice_points_per_basis_point = {bp: [] for bp in self.chemical_basis.keys()}
        for position in super_positions:
            f = np.round(self.A_chem_inv @ position, 8) # floating point errors affect modulo
            f_reduced = f % 1.0
            bp_found = False
            for bp in self.chemical_basis.keys():
                if np.allclose(np.array(bp), f_reduced):
                    bp_found = bp
            if bp_found is False:
                raise ValueError(f"({self.cid}) Could not match supercell position r={position}, f(reduced)={f_reduced} with any chemical basis point. Please investigate")
            else:
                lattice_points_per_basis_point[bp_found].append(position)
        
        # allocate chemical species for each basis point
        species_per_basis_point = {bp: {sp: 0 for sp in self.comp.keys()} for bp in self.chemical_basis.keys()}
        for bp, chem_map in self.chemical_basis.items():
            num_lattice_points = len(lattice_points_per_basis_point[bp])
            for sp in self.comp.keys():
                if sp in chem_map.keys():
                    species_per_basis_point[bp][sp] = round(num_lattice_points*chem_map[sp])
                else:
                    species_per_basis_point[bp][sp] = 0

        # write information to file
        with open(self.cdir / 'composition.out', 'w') as f:
            for bp in self.chemical_basis.keys():
                f.write('Chemical basis position (frac)\n')
                f.write(f"{bp[0]:2.6f}\t{bp[1]:2.6f}\t{bp[2]:2.6f}\n\n")

                f.write('Supercell lattice position\n')
                for lp in lattice_points_per_basis_point[bp]:
                    f.write(f"{lp[0]:2.6f}\t{lp[1]:2.6f}\t{lp[2]:2.6f}\n")
                f.write('\n')

                f.write('Decorated species count\n')
                for sp, num in species_per_basis_point[bp].items():
                    f.write(f"{sp}: {num}\n")
                f.write('\n\n')

        # compare needed number of each species with comp
        for sp, num_sp_comp in self.comp.items():
            num_sp_needed = 0
            for sp_dict in species_per_basis_point.values():
                num_sp_needed += sp_dict[sp]
            if num_sp_needed != num_sp_comp:
                raise ValueError(f"({self.cid}) Composition incompatible with chemical basis. Need {num_sp_needed} {sp} atoms, but {num_sp_comp} in composition")

        # define lattice by randomly assigning species to lattice points
        rng = np.random.default_rng(seed=self.seed)

        self.lattice = {sp: [] for sp in self.comp.keys()}
        for bp in self.chemical_basis.keys():
            lattice_points = lattice_points_per_basis_point[bp]
            species = []
            for sp, num in species_per_basis_point[bp].items():
                species += [sp]*num
            species = rng.permutation(species)
            for i, sp in enumerate(species):
                self.lattice[sp].append([float(p) for p in lattice_points[i]])

        with open(self.cdir / 'supercell.xyz', 'w') as f:
            f.write(f'{num_atoms_in_super}\n\n')
            for sp, positions in self.lattice.items():
                for pos in positions:
                    f.write(f"{sp} {pos[0]:3.8f} {pos[1]:3.8f} {pos[2]:3.8f}\n")

        poscar_data = {'cid': self.cid, 'size': [self.a0*s for s in self.size], 'lattice': self.lattice}
        if self.poscar is None:
            self.poscar = VaspPoscar().load(from_data=poscar_data)
        
    def _check_transformation_matrix(self):
        if self.transform.shape != (3,3):
            raise ValueError(f"({self.cid}) Transformation matrix must be 3x3. Got {self.transform.shape}")

        if np.any(self.transform < 0):
            raise ValueError(f"({self.cid}) Transformation matrix must not have any negative elements")
        
        if not np.all(self.transform == np.floor(self.transform)):
            raise ValueError(f"({self.cid}) Transformation matrix must have only integer elements")
        
        rowc, colc = False, False
        if np.any(self.transform[np.tril_indices(3, k=-1)]):
            rowc = True
        if np.any(self.transform[np.triu_indices(3, k=1)]):
            colc = True    
        if all([rowc, colc]):
            return ValueError(f"({self.cid}) Transformation matrix must be either an upper triangular or lower triangular matrix")

        if rowc:
            for el1, el2 in zip([(1,0), (2,0), (2,1)], [(1,1), (2,2), (2,2)]):
                if self.transform[el1] >= self.transform[el2]:
                    raise ValueError(f"({self.cid}) Transformation matrix element {el1} must be less than {el2}")
        
        if colc:
            for el1, el2 in zip([(0,1), (0,2), (1,2)], [(0,0), (0,0), (1,1)]):
                if self.transform[el1] >= self.transform[el2]:
                    raise ValueError(f"({self.cid}) Transformation matrix element {el1} must be less than {el2}")
                
    def write_vasp_files(self, write_dir: Path):
        success = {'INCAR': False, 'POSCAR': False, 'KPOINTS': False, 'POTCAR': False}
        if not write_dir.exists():
            raise FileNotFoundError(f"Directory not found {write_dir}")
        
        if self.incar is not None:
            self.incar.write(write_dir / 'INCAR')
            success['INCAR'] = True

        if self.poscar is not None:
            self.poscar.write(write_dir / 'POSCAR')
            success['POSCAR'] = True

        if self.kpoints is not None:
            self.kpoints.write(write_dir / 'KPOINTS')
            success['KPOINTS'] = True

        if self.potcar is not None:
            self.potcar.write(write_dir / 'POTCAR')
            success['POTCAR'] = True

        return success
    
    def load_energy(self):
        """Attempts to load POSCAR data and obtain energy for cluster expansion."""
        poscar_path = self.cdir / 'POSCAR'
        
        if poscar_path.exists():
            self.Atoms = read(poscar_path, format='vasp')
            success = True
        else:
            success = False

        energy_path = self.cdir / 'energy.out'
        if energy_path.exists():
            with open(energy_path, 'r') as f:
                self.energy = float(f.readline().strip())
        else:
            success = False

        if self.include_fit and success:
            self.fit = True
        
        return success

    def wipe_cdir(self):
        for p in self.cdir.iterdir():
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)

    def run_vasp(self):
        self.outfile = open(self.cdir / 'vasp_std.out', 'w')
        vasp_cmd = ['srun', f'--ntasks={self.ncores}', '--export=ALL', 'vasp_std']
        self.process = subprocess.Popen(vasp_cmd, cwd=self.cdir, stdout=self.outfile, stderr=subprocess.STDOUT)
        self.status = 1
        self.start_time = time.perf_counter()

    def poll_process(self):
        # process can not be polled
        if self.status < 1 or self.process is None:
            return
        
        poll = self.process.poll()
        if poll == 0:
            self.status = 2
            outcar = VaspOutcar().load(from_path = self.cdir / 'OUTCAR')
            self.energy = outcar.get_energy()

            with open(self.cdir / 'energy.out', 'w') as f:
                f.write(f'{self.energy:4.8f}')

            self.process_time = time.perf_counter() - self.start_time
            self.outfile.write(f'Total wall time: {floor(self.process_time/3600)}hrs {floor((self.process_time/60)%60)}min {round((self.process_time%60))}sec')

def check_configuration_dict(conf_dict: dict):
    conf_dict = deepcopy(conf_dict)

    if 'run' not in conf_dict.keys():
        conf_dict['run'] = None

    if 'ncores' not in conf_dict.keys():
        conf_dict['ncores'] = None

    if 'include_fit' not in conf_dict.keys():
        conf_dict['include_fit'] = True

    # things needed to create a lattice
    if 'transform' not in conf_dict.keys():
        conf_dict['transform'] = None
    else:
        conf_dict['transform'] = np.array(conf_dict['transform'])

    if 'basis' not in conf_dict.keys():
        conf_dict['basis'] = None
    else:
        basis_dict = {}
        for key, val in conf_dict['basis'].items():
            bp = key.split(',')
            bp = tuple([float(v) for v in bp])
            basis_dict[bp] = val
        conf_dict['basis'] = basis_dict

    if 'size' not in conf_dict.keys():
        conf_dict['size'] = None

    if 'comp' not in conf_dict.keys():
        conf_dict['comp'] = None

    if 'seed' not in conf_dict.keys():
        conf_dict['seed'] = None
    else:
        conf_dict['seed'] = int(conf_dict['seed'])

    if 'name' not in conf_dict.keys():
        conf_dict['name'] = None

    # VASP files
    if 'incar' in conf_dict.keys():
        conf_dict['incar'] = VaspIncar().load(from_string=conf_dict['incar'])
    else:
        conf_dict['incar'] = None

    if 'poscar' in conf_dict.keys():
            conf_dict['poscar'] = VaspPoscar().load(from_string=conf_dict['poscar'])
    else:
        conf_dict['poscar'] = None

    if 'kpoints' in conf_dict.keys():
            conf_dict['kpoints'] = VaspKpoints().load(from_string=conf_dict['kpoints'])
    else:
        conf_dict['kpoints'] = None

    if 'potcar' in conf_dict.keys():
        conf_dict['potcar'] = VaspPotcar().load(dirnames=conf_dict['potcar'])
    else:
        conf_dict['potcar'] = None

    return conf_dict

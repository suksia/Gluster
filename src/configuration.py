from copy import deepcopy
from vasp_file import *
import numpy as np
from pathlib import Path
import subprocess, os, shutil, time
from ase.io import read
from math import floor
from utils import is_hnf, product

ELEMENTS = {
    'W': {'mass': 183.84, 'a0': 3.165},
    'Mo': {'mass': 95.95, 'a0': 3.147},
    'V': {'mass': 50.94, 'a0': 3.027},
}

try:
    NTASKS = int(os.environ['SLURM_NTASKS'])
except:
    NTASKS = 1

class IterRegistry(type):
    def __iter__(cls):
        return iter(cls._registry)

class Configuration(object):
    __metaclass__ = IterRegistry
    _registry: list[Configuration] = []

    def __init__(self, cid: int, cdir: Path, conf_dict: dict):
        self._registry.append(self)
        self.cid = cid
        self.cdir = cdir

        self.size = [1]*3
        if 'size' in conf_dict.keys():
            self.size = conf_dict['size']
        self.num_conv_lps = 2*product(self.size)

        # make sure transformation matrix is in Hermite normal form
        if 'transform' not in conf_dict.keys():
            raise KeyError(f"({self.cid}) No transformation matrix was provided")
        else:
            self.transform = np.array(conf_dict['transform'])
            if self.transform.shape != (3,3):
                raise ValueError(f"({self.cid}) Transformation matrix must be 3x3. Got {self.transform.shape}")

            if np.any(self.transform < 0):
                raise ValueError(f"({self.cid}) Transformation matrix must not have any negative elements")

            if np.all(self.transform == np.floor(self.transform)):
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
                for el1, el2 in zip([(0,1), (0,2), (1,2)], [(1,1), (2,2), (2,2)]):
                    if self.transform[el1] >= self.transform[el2]:
                        raise ValueError(f"({self.cid}) Transformation matrix element {el1} must be less than {el2}")

        self.ncores = 1
        if 'ncores' in conf_dict.keys():
            self.ncores = conf_dict['ncores']

        self.comp, self.comp_at = None, None
        if 'comp' in conf_dict.keys():
            total_conc = sum(conf_dict['comp'].values())
            if total_conc == 1.0:
                self.comp = conf_dict['comp']
                self.comp_at = {sp: round(c*self.num_conv_lps) for sp, c in self.comp.items()}
            elif total_conc == self.num_conv_lps:
                self.comp = {sp: c/total_conc for sp, c in self.comp.items()}
                self.comp_at = conf_dict['comp']
            else:
                raise ValueError(f"({self.cid}) Composition values must sum to 1.0 if they are percentages or {self.num_conv_lps} if they are atom counts")

        # attributes used for cluster expansion
        self.energy = None # VASP output
        self.Atoms = None # ASE Atoms object loaded from POSCAR
        
        # set status to finished if run=False and energy.out and POSCAR don't exist
        self.status = 0
        if self.load_energy():
            self.status = 2
        if 'run' in conf_dict.keys():
            if conf_dict['run']:
                self.status = 0

        # attributes for subprocess
        self.process = None
        self.start_time, self.process_time = None, None

        # atomic data
        self.positions = None
        self.ids = None
        self.species = None

        # VASP files
        self.incar = VaspIncar()
        if 'incar' in conf_dict.keys():
            self.incar.load(from_string=conf_dict['incar'])

        self.kpoints = VaspKpoints()
        if 'kpoints' in conf_dict.keys():
            self.kpoints.load(from_string=conf_dict['kpoints'])

        self.poscar = VaspPoscar()
        if 'poscar' in conf_dict.keys():
            self.poscar.load(from_string=conf_dict['poscar'])

        self.potcar = VaspPotcar()
        if 'potcar' in conf_dict.keys():
            self.potcar.load(conf_dict['potcar'])
            
        # determine basis positions
        transform_inv = np.linalg.inv(self.transform)
        num_bp = round(abs(np.linalg.det(self.transform)))
        
        frac_positions_chem = np.array([transform_inv @ t % 1.0 for t in list(np.ndindex(*[num_bp]*3))])
        basis_chem = []
        for f in frac_positions_chem:
            if not any(np.allclose(f, x) for x in basis_chem):
                basis_chem.append(tuple(f))
            if len(basis_chem) == num_bp:
                break

        # check basis positions if they were provided
        self.basis = {bp: None for bp in basis_chem}
        if 'basis' in conf_dict.keys():
            for key, val in conf_dict['basis'].items():
                bp_in = key.split(',')
                bp_in = tuple([float(v) for v in bp_in])

                bp_found = False
                for bp in self.basis.keys():
                    if np.allclose(np.array(bp_in), np.array(bp)):
                        bp_found = bp
                if not bp_found:
                    raise ValueError(f"({self.cid}) Basis atom at {bp_in} does not match the specified transform matrix. Enumerated positions: {basis_chem}")
                
                # make sure probabilities add to one
                if type(val) == 'str':
                    self.basis[bp_found] = {val: 1.0}
                elif type(val) == 'dict':
                    if sum(val.values()) != 1.0:
                        raise ValueError(f"({self.cid}) Occupancy probabilities do not sum to 1 for basis atom {bp_in}")
                    self.basis[bp_found] = val

        # composition can be determined if all basis positions exist and the configuration is ordered
        self.basis_provided, self.ordered = False, False
        comp, comp_at = {}, {}
        if all([True if val is not None else False for val in self.basis.values()]):
            self.basis_provided = True

            # configuration is not ordered if there is an occupancy probability that is not 0 or 1
            for bp_occ in self.basis.values():
                if any([True if val != 1.0 or val != 0.0 else False for val in bp_occ.values()]):
                    pass
                else:
                    self.ordered = True

            if self.ordered:
                # count atoms on basis positions
                for bp_occ in self.basis.values():
                    for sp, p in bp_occ.items():
                        if p == 1:
                            if sp not in comp_at.keys():
                                comp_at[sp] = 1
                            else:
                                comp_at[sp] += 1

                # compute composition percentages
                comp = {sp: c / sum(comp_at.values()) for sp, c in comp_at.items()}

                # scale up counts to full size
                comp_at = {sp: round(c*self.num_conv_lps) for sp, c in comp.items()}

        # check composition if it is provided and known
        if self.comp and len(comp):
            for sp, c_in in self.comp:
                if sp not in comp.keys():
                    raise ValueError(f"({self.cid}) Species {sp} was not found in basis. Computed composition: {comp}")
                else:
                    if c_in != comp[sp]:
                        raise ValueError(f"({self.cid}) Provided composition is inconsistent with basis. Computed composition: {comp}")

            for sp, c_in in self.comp_at:
                if sp not in comp_at.keys():
                    raise ValueError(f"({self.cid}) Species {sp} was not found in basis. Computed composition: {comp_at}")
                else:
                    if c_in != comp_at[sp]:
                        raise ValueError(f"({self.cid}) Provided composition is inconsistent with basis. Computed composition: {comp_at}")

        # raise error if composition was not provided, but the system is disordered and the basis is fully defined 
        elif self.comp is None and self.ordered is False and self.basis_provided is True:
            raise KeyError(f"({self.cid}) Composition must be provided if the configuration is disordered")

        # compute lattice constant
        self.a0 = 0.0
        if 'a0' in conf_dict.keys():
            self.a0 = conf_dict['a0']
        elif self.comp:
            for sp, c in self.comp.items():
                self.a0 += c*ELEMENTS[sp]['a0']
        else:
            self.a0 = 1.0 # dummy value
        
        A_prim = self.a0/2*np.array([[ 1, -1,  1], 
                                     [ 1,  1, -1], 
                                     [-1,  1,  1]])
        self.A_chem = A_prim @ self.transform

    @classmethod
    def write_basis(cls, write_path: Path):
        with open(write_path, 'w') as f:
            for conf in cls._registry:
                A_chem = 1/2*np.array([[1,-1,1], [1,1,-1],[-1,1,1]]) @ conf.transform
                f.write(f"{conf.cid}\n\n")
                for i in range(3):
                    lv = np.round(A_chem[:,i], 6)
                    f.write(f"{lv[0]:2.6f}\t{lv[1]:2.6f}\t{lv[2]:2.6f}\n")
                f.write('\n')
                for bp in conf.basis.keys():
                    f.write(f"{bp[0]:2.6f}\t{bp[1]:2.6f}\t{bp[2]:2.6f}\n")
                f.write('\n\n')
    
    def create_lattice(self):
        # enumerate all translation vectors
        transv = []
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    transv.append(np.array([i,j,k]))
        
        # define positions in conventional unit cell
        unit_pos = [np.array([0, 0, 0]), np.array([0.5, 0.5, 0.5])]
        pos = []
        for p in unit_pos:
            for t in transv:
                pos.append(p+t)

        self.positions = np.array(pos)
        self.N = len(self.positions)
        self.ids = np.arange(self.N)

        # determine number atoms to be assigned for each element
        for sp, conc in self.comp.items():
            self.comp_at[sp] = round(self.N*conc)

        # generate a set of indices corresponding to random positions
        rng = np.random.default_rng()
        rand_pos_idx = rng.permutation(self.N)

        # random decoration for M-component alloys
        if self.order is None:
            self.species = []
            for sp, nat in self.comp_at.items():
                spi = self.species_to_spi[sp]
                self.species += [spi]*nat
            self.species = rng.permutation(self.species)

        # ordering in binary alloys
        elif self.M == 2:
            raise NotImplementedError()

        elif self.M > 2:
            raise NotImplementedError()

        # reorder positions, ids, species so that species are sorted
        order = np.argsort(self.species)
        self.species = self.species[order]
        self.positions = self.positions[order]*self.a0
        self.ids = self.ids[order]

        self.poscar = VaspPoscar().load(from_data={'size': self.a0*self.size, 'positions': self.positions, 'composition': self.comp_at})
    
    def write_vasp_files(self, write_dir: Path):
        if not write_dir.exists():
            raise FileNotFoundError(f"Directory not found {write_dir}")
        self.incar.write(write_dir / 'INCAR')
        self.kpoints.write(write_dir / 'KPOINTS')
        self.poscar.write(write_dir / 'POSCAR')
        self.potcar.write(write_dir / 'POTCAR')
    
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
        poll = self.process.poll()
        if poll == 0:
            self.status = 2
            outcar = VaspOutcar().load(from_path = self.cdir / 'OUTCAR')
            self.energy = outcar.get_energy()

            with open(self.cdir / 'energy.out', 'w') as f:
                f.write(f'{self.energy:4.8f}')

            self.process_time = time.perf_counter() - self.start_time
            self.outfile.write(f'Total wall time: {floor(self.process_time/3600)}hrs {floor((self.process_time/60)%60)}min {round((self.process_time%60))}sec')
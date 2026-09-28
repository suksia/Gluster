from copy import deepcopy
from vasp_file import *
import numpy as np
from pathlib import Path
import subprocess, os, shutil, time
from ase.io import read
from math import floor

ELEMENTS = {
    'W': {'mass': 183.84, 'a0': 3.165},
    'Mo': {'mass': 95.95, 'a0': 3.147},
    'V': {'mass': 50.94, 'a0': 3.027},
}

try:
    NTASKS = int(os.environ['SLURM_NTASKS'])
except:
    NTASKS = 1

class Configuration:
    def __init__(self, cid: int, cdir: Path, conf_dict: dict):
        self.cid = cid
        self.cdir = cdir
        self.a0 = conf_dict['a0']
        self.size = conf_dict['size']
        self.comp = deepcopy(conf_dict['comp'])
        self.order = conf_dict['order']
        self.ncores = conf_dict['ncores']
        self.status = 0

        # set status to finished if run=False and energy.out and POSCAR don't exist
        if self.load_energy():
            self.status = 2

        if 'run' in conf_dict.keys():
            if conf_dict['run']:
                self.status = 0

        # check composition
        if self.comp is None:
            raise ValueError(f'Composition data missing for configuration {self.cid}. Check input file')
     
        total_conc = sum(self.comp.values())
        if total_conc == 100:
            self.comp = {c: v/total_conc for c, v in self.comp.items()}
        elif total_conc == 1:
            pass
        else:
            raise ValueError(f'Composition must add to 1.00 or 100.00 exactly. Got {self.comp} for configuration {self.cid}, adding to {total_conc:3.2f}')

        # define lattice constant as weighted average over composition
        if self.a0 is None:
            self.a0 = 0
            for sp, c in self.comp.items():
                self.a0 += c*ELEMENTS[sp]['a0']
        
        # VASP files
        self.incar = VaspIncar().load(from_string=conf_dict['incar'])
        self.kpoints = VaspKpoints().load(from_string=conf_dict['kpoints'])
        self.potcar = VaspPotcar().load(conf_dict['potcar'])
        self.poscar = None

        # miscellaneous attributes
        self.comp_at = {el: None for el in self.comp.keys()} # number of atoms per species
        self.M = len(self.comp) # number of species
        self.N = None # number of lattice points
        self.process = None
        self.start_time, self.process_time = None, None

        # dictionaries mapping element name to index (e.g., W -> 1 and 1 -> W)
        self.species_to_spi = {sp: spi for spi, sp in enumerate(self.comp.keys())}
        self.spi_to_species = {spi: sp for sp, spi in self.species_to_spi.items()}
        
        # atomic data
        self.positions = None
        self.ids = None
        self.species = None

        # attributes used for cluster expansion
        self.energy = None # VASP output
        self.Atoms = None # ASE Atoms object loaded from POSCAR

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
import subprocess, time
import numpy as np
from pathlib import Path
from ase import Atoms
from ase.io import read
from subprocess import Popen
from math import floor

from vasp_file import VaspIncar, VaspPoscar, VaspKpoints, VaspPotcar, VaspOutcar
from utils import is_hnf, replicate, ELEMENTS


class Configuration:
    def __init__(self, H: np.ndarray, basis: dict[tuple[float], str], size: int, cid: int = None, cdir: Path = None):
        if not is_hnf(H):
            raise ValueError(f"Matrix {H} is not in Hermite normal form")

        self.cid = cid
        self.cdir = cdir

        self.ordered = None
        self.basis = basis
        self.elements = set(self.basis.values())
        self.rank = len(self.elements)

        self.at_composition: dict[str, int] = None
        self.composition: dict[str, float] = None
        self.a0, self.avg_a0 = 1.0, None

        self.H = H
        self.A_prim = 0.5*np.array([[1, -1, 1], [1, 1, -1], [-1, 1, 1]])
        self.B = self.A_prim @ self.H
        self.B_inv = np.linalg.inv(self.B)

        self.A_super = np.eye(3)
        self.supercell: dict[tuple[float], str] = None
        self.size = size

        self.incar: VaspIncar = None
        self.kpoints: VaspKpoints = None
        self.poscar: VaspPoscar = None
        self.potcar: VaspPotcar = None
        self.outcar: VaspOutcar = None

        self.status = 0  # 0 = not run, 1 = running, 2 = finished
        self.process: Popen = None
        self.start_time: float = None
        self.process_time: float = None
        self.ncores: int = None
        
        self.Atoms: Atoms = None
        self.energy: float = None
        self.norm_energy: float = None
        self.mixing_energy: float = None
        
        if self.get_energy():
            self.status = 2
          
    def compute_composition(self):
        """Given the supercell dictionary, compute the atomic and fractional composition. Also compute average lattice constant."""
        if self.supercell is None:
            mapping = self.basis
        else:
            mapping = self.supercell

        self.at_composition = {}
        for sp in mapping.values():
            if sp not in self.at_composition.keys():
                self.at_composition[sp] = 1
            else:
                self.at_composition[sp] += 1

        self.composition = {}
        for sp, num in self.at_composition.items():
            self.composition[sp] = num / sum(self.at_composition.values())

        self.avg_a0 = 0.0
        for sp, conc in self.composition.items():
            self.avg_a0 += ELEMENTS[sp]['a0'] * conc

    def set_lattice_const(self, avg = True, a0 = None):
        """Replaces the lattice constant and rescales primitive and supercell lattice vectors."""
        if avg:
            if self.avg_a0 is None:
                self.compute_composition()
            a0 = self.avg_a0
        elif a0 is None:
            raise ValueError('a0 must be provided if the average lattice constant is not being used')

        self.A_prim /= self.a0
        self.A_super /= self.a0

        self.a0 = a0
        self.A_prim *= a0
        self.A_super *= a0

    def create_poscar(self):
        if self.supercell is None:
            raise ValueError(f"Supercell must be defined to create POSCAR")

        if self.composition is None:
            self.compute_composition()

        if self.a0 == 1.0:
            self.set_lattice_const(avg=True)

        lattice_vectors = []
        for lv in self.A_super:
            lattice_vectors.append((lv*self.size).tolist())

        ions = {}
        for lp, sp in self.supercell.items():
            if sp not in ions.keys():
                ions[sp] = [lp]
            else:
                ions[sp].append(lp)

        self.poscar = VaspPoscar().load(from_data={
            "comment": f"Configuration {self.cid}",
            "scaling_factor": 1.0,
            "lattice_vectors": lattice_vectors,
            "coord_system": "Direct",
            "ions": ions})

    def set_vasp_files(self, overwrite = True, incar: VaspIncar = None, kpoints: VaspKpoints = None, potcar_names: list[str] = None):
        if incar:
            if self.incar is not None and overwrite is False:
                pass
            else:
                self.incar = incar

        if kpoints:
            if self.kpoints is not None and overwrite is False:
                pass
            else:
                self.kpoints = kpoints

        if self.poscar is None:
            self.create_poscar()
        
        if self.poscar and potcar_names:
            names_in_poscar = []
            for sp in self.poscar.ions.keys():
                for nm in potcar_names:
                    if sp in nm:
                        names_in_poscar.append(nm)
                        break

            self.potcar = VaspPotcar().load(names_in_poscar)

    def write_vasp_files(self, write_dir: Path = None):
        if write_dir is None:
            write_dir = self.cdir

        if not write_dir.exists():
            raise FileNotFoundError(f"[{write_dir}] Directory not found")
        
        if self.incar is not None:
            self.incar.write(write_dir / 'INCAR')

        if self.poscar is not None:
            self.poscar.write(write_dir / 'POSCAR')

        if self.kpoints is not None:
            self.kpoints.write(write_dir / 'KPOINTS')

        if self.potcar is not None:
            self.potcar.write(write_dir / 'POTCAR')

    def get_energy(self):
        """Attempts to load POSCAR data and obtain energy for cluster expansion."""
        if self.cdir is None:
            return False
        
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
            self.outfile.close()


class OrderedConfiguration(Configuration):
    def __init__(self, H: np.ndarray, basis: dict[tuple[float], str], size: int, cid: int = None, cdir: Path = None):
        if len(basis) != int(np.rint(np.linalg.det(H))):
            raise ValueError(f"Number of points in basis ({len(basis)}) does not match determinant of {H}")

        super().__init__(H, basis, size, cid, cdir)
        self.ordered = True

    def map_to_supercell(self):
        """Replicates the conventional supercell and maps the basis onto it."""
        conv_basis_pos = [np.array([0, 0, 0]), np.array([0.5, 0.5, 0.5])]
        supercell_size = tuple([self.size]*3)
        supercell_positions = replicate(self.A_super, conv_basis_pos, supercell_size)

        supercell_map = {}
        for lp in supercell_positions:
            lp_reduced = np.round(self.B_inv @ lp, 6) % 1.0

            bp_found = False
            for bp, sp in self.basis.items():
                if np.allclose(lp_reduced, bp):
                    bp_found = bp
                    break

            if bp_found is False:
                raise ValueError(f"[{self.cid}] Could not match supercell position f={lp} with any basis point. Please investigate")
            else:
                lp /= np.array(supercell_size)
                lp = lp.tolist()
                supercell_map[tuple(lp)] = sp

        self.supercell = supercell_map


class DisorderedConfiguration(Configuration):
    def __init__(self, basis: dict[tuple[float], str], size: int, cid: int = None, cdir: Path = None):
        super().__init__(np.eye(3), basis, size, cid, cdir)

        self.supercell = basis
        self.ordered = False
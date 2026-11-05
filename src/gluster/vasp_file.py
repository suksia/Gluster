import logging
from pathlib import Path
from utils import strip_split

logger = logging.getLogger('Gluster')

POTCAR_DIR = Path('/storage/group/xvw5285/default/atomistic_pkgs/potpaw_PBE.54/')


class VaspFile:
    def __init__(self):
        self.name = self.__class__.__name__
        self.lines: list[str] = None
        self.last_read_path: Path = None
    
    def load(self, from_path: Path = None, from_string: str = None):
        if from_path:
            if not from_path.exists():
                raise FileNotFoundError(f'[{self.name}] File not found at {from_path}')

            self.last_read_path = from_path
            with open(from_path, 'r') as f:
                lines = f.readlines()
            lines = [l.strip('\n') for l in lines] # remove any \n characters
            lines = [l for l in lines if l.strip() != ''] # remove blank lines
            self.lines = lines

        elif from_string:
            self.lines = from_string.split('\n')
            self.lines = [l+'\n' for l in self.lines]
        
        return self

    def write(self, path: Path):
        if not path.parent.exists():
            raise FileNotFoundError(f'[{self.name}] Write directory not found at {path.parent}')
        
        with open(path, 'w') as f:
            f.writelines(self.lines)

        #logger.debug(f"{self.name}: Wrote lines to {path}")
        
class VaspIncar(VaspFile):
    def update_setting(self, name: str, value):
        for i, line in enumerate(self.lines):
            if name in line:
                self.lines[i] = f"{name} = {value}\n"
                return

        self.lines.append(f"{name} = {value}\n")

    def remove_setting(self, name: str):
        new_lines = []
        for line in self.lines:
            if name not in line:
                new_lines.append(line)
                
        self.lines = new_lines

class VaspPoscar(VaspFile):
    def __init__(self):
        super().__init__()
        self.comment_line: str = None
        self.scaling_factor: float = None
        self.lattice_vectors: list[list[float]] = None
        self.coord_system: str = None
        self.ions: dict[str, list[tuple[float]]] = None

    def load(self, from_path: Path = None, from_string: str = None, from_data: dict = None):
        if from_data:
            self.comment_line = from_data['comment']
            self.scaling_factor = from_data['scaling_factor']
            self.lattice_vectors = from_data['lattice_vectors']
            self.coord_system = from_data['coord_system']
            self.ions = from_data['ions']

            lines = [self.comment_line + '\n']
            lines.append(f'{self.scaling_factor:3.5f}\n')

            for lv in self.lattice_vectors:
                lines.append(f"{lv[0]:3.8f}\t{lv[1]:3.8f}\t{lv[2]:3.8f}\n")
           
            sp_line, nat_line, pos_line = "", "", []
            for sp, lps in self.ions.items():
                sp_line += f"{sp:3} "
                nat_line += f"{len(lps):<3} "
                for pos in lps:
                    pos_line.append(f"{pos[0]:3.8f}\t{pos[1]:3.8f}\t{pos[2]:3.8f}\n")

            lines.append(sp_line+'\n')
            lines.append(nat_line+'\n')
            lines.append(self.coord_system+'\n')
            lines += pos_line
            self.lines = lines

            return self
        
        elif from_path:
            super().load(from_path=from_path)
        elif from_string:
            super().load(from_string=from_string)

        self.lattice_vectors = []
        self.species, self.ion_counts, self.lattice_points = [], [], []

        pos_start_i = None
        for i, l in enumerate(self.lines):
            l = l.strip()

            if i == 0:
                self.comment_line = l

            elif i == 1:
                self.scaling_factor = float(l)

            elif i in [2, 3, 4]:
                self.lattice_vectors.append([float(v) for v in l.split()])

            elif i == 5:
                for sp in l.strip():
                    self.species.append(sp)

            elif i == 6:
                for cnt in l.strip():
                    self.ion_counts.append(cnt)

            elif i == 7:
                if l == 'Selective Dynamics':
                    self.coord_system = self.lines[8].strip()
                    pos_start_i = 9
                else:
                    self.coord_system = l
                    pos_start_i = 8

            elif i >= pos_start_i:
                self.lattice_points.append([float(p) for p in l.split()])

        return self

class VaspPotcar(VaspFile):
    def __init__(self):
        super().__init__()
        self.dirnames = None
    
    def load(self, dirnames: list[str] = None):
        self.lines = []
        self.dirnames = dirnames
        for dirname in self.dirnames:
            potcar_path = POTCAR_DIR / dirname / 'POTCAR'
            if not potcar_path.exists():
                raise FileNotFoundError(f"[{self.name}] POTCAR file not found in {potcar_path.parent}")

            with open(potcar_path, 'r') as f:
                self.lines += f.readlines()

        return self

class VaspKpoints(VaspFile):
    def update_mesh(self, mesh: list[int, int, int]):
        self.lines[3] = f"{mesh[0]} {mesh[1]} {mesh[1]}\n"

class VaspOutcar(VaspFile):
    def get_energy(self):
        last_line_w_energy = None
        for line in self.lines:
            if 'energy(sigma->0)' in line:
                last_line_w_energy = line
        energy = strip_split(last_line_w_energy)[-1]
        return float(energy)

    def get_elapsed_time(self):
        elapsed_time_line = None
        for line in self.lines:
            if 'Elapsed time' in line:
                elapsed_time_line = line
        elapsed_time = strip_split(elapsed_time_line)[-1]
        return float(elapsed_time)

    def get_num_ibz_kpoints(self):
        num_ibz_kpoints_line = None
        for line in self.lines:
            if 'irreducible k-points' in line:
                num_ibz_kpoints_line = line
        num_ibz_kpoints = strip_split(num_ibz_kpoints_line)[1]
        return int(num_ibz_kpoints)
        
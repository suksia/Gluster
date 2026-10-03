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

        logger.debug(f"{self.name}: Wrote lines to {path}")
        
class VaspIncar(VaspFile):
    pass

class VaspPoscar(VaspFile):
    def load(self, from_path: Path = None, from_string: str = None, from_data: dict = None):
        if from_path:
            super().load(from_path=from_path)
        elif from_string:
            super().load(from_string=from_string)
        elif from_data:
            lines = ['POSCAR\n']
            lines.append('1.000\n')
            lines.append(f"{from_data['size']:3.8f}\t{0:3.8f}\t{0:3.8f}\n")
            lines.append(f"{0:3.8f}\t{from_data['size']:3.8f}\t{0:3.8f}\n")
            lines.append(f"{0:3.8f}\t{0:3.8f}\t{from_data['size']:3.8f}\n")

            sp_line, nat_line, pos_line = "", "", []
            low_ri = 0
            for sp, nat in from_data['composition'].items():
                sp_line += f"{sp:3} "
                nat_line += f"{nat:<3} "
                high_ri = low_ri + nat
                for ri in range(low_ri, high_ri):
                    pos = from_data['positions'][ri] / from_data['size']
                    pos_line.append(f"{pos[0]:3.8f}\t{pos[1]:3.8f}\t{pos[2]:3.8f}\n")
                low_ri = high_ri

            lines.append(sp_line+'\n')
            lines.append(nat_line+'\n')
            lines.append('Direct'+'\n')
            lines += pos_line
            self.lines = lines
            
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
    pass

class VaspOutcar(VaspFile):
    def get_energy(self):
        last_line_w_energy = None
        for line in self.lines:
            if 'energy(sigma->0)' in line:
                last_line_w_energy = line
        energy = strip_split(last_line_w_energy)[-1]
        return float(energy)
from copy import deepcopy
import os, time, logging, itertools
from pathlib import Path
import matplotlib.pyplot as plt

from configuration import Configuration
from utils import alloy_string

try:
    NTASKS = int(os.environ['SLURM_NTASKS'])
except:
    NTASKS = 1

logger = logging.getLogger('Gluster')

class Dataset:
    def __init__(self, configs: dict[int, Configuration]):
        self.configs: dict[int, Configuration] = configs
        self.queue: dict[int, Configuration] = {}
        self.end_members: dict[str, Configuration] = {}
        
        for conf in self.configs.values():
            if conf.rank == 1:
                el = next(iter(conf.basis.values()))
                self.end_members[el] = conf

        for cid, conf in self.configs.items():
            if conf.status == 0:
                self.queue[cid] = conf

        self.max_rank = 0
        for conf in self.configs.values():
            self.max_rank = max(self.max_rank, conf.rank)

        self.num_available_cores = NTASKS

        self.end_member_energy: dict[str, float] = {}
        self.mixing_energies: dict[Configuration, float] = {}

    def run_jobs(self):
        if len(self.queue) == 0:
            logger.debug(f"Queue is empty, skipping VASP")

        while len(self.queue):
            # launch jobs
            for cid, conf in self.queue.items():
                if conf.status == 1 or conf.status == 2:
                    continue

                if conf.ncores > NTASKS:
                    raise ValueError(f"Too many cores requested for config {cid}. Need {conf.ncores} but only have a maximum of {NTASKS}")
                elif self.num_available_cores < conf.ncores:
                    continue

                conf.run_vasp()

                self.num_available_cores -= conf.ncores
                conf.status = 1
                logger.debug(f'Running VASP on config {cid} with {conf.ncores} cores (# cores ready: {self.num_available_cores})')
                
                time.sleep(1) # give slurm time to add job steps
            
            # check statuses
            finished = []
            for cid, conf in self.queue.items():
                conf.poll_process()
                if conf.status == 1:
                    continue
                elif conf.status == 2:
                    self.num_available_cores += conf.ncores
                    finished.append(cid)
                    logger.debug(f'VASP finished for config {cid} (# cores ready: {self.num_available_cores})')

            for fin_cid in finished:
                self.queue.pop(fin_cid)

    def compute_mixing_energies(self):
        # compute energy of end members
        for el, conf in self.end_members.items():
            if conf.energy is None:
                raise ValueError(f"Need energy for end-member {el} (config {conf.cid}) for calculating mixing energy, but VASP may not have run or finished")

            conf.norm_energy = conf.energy / len(conf.supercell)
            self.end_member_energy[el] = conf.norm_energy

        # compute mixing energies
        for cid, conf in self.configs.items():
            conf.compute_composition()

            if conf.rank == 1:
                continue
            elif conf.energy is None:
                continue
            else:
                conf.norm_energy = conf.energy / len(conf.supercell)

                conf.mixing_energy = conf.norm_energy
                for sp, conc in conf.composition.items():
                    conf.mixing_energy -= conc*self.end_member_energy[sp]

    def plot_binary_mixing_energy(self, write_dir: Path, elements: tuple[str]):
        x_element = elements[0]
        all_str = alloy_string(elements)

        x, y = [], []
        for conf, me in self.mixing_energies.items():
            skip = False

            if conf.rank != 2:
                skip = True

            for el in elements:
                if el not in conf.elements:
                    skip = True
                    break

            if skip:
                continue
            
            x.append(conf.composition[x_element])
            y.append(me*1000)

        fig = plt.figure()

        plt.scatter(x, y)
        plt.xlabel(f"x({x_element})")
        plt.ylabel(f"Mixing Energy [meV]")
        plt.title(f"{all_str} Mixing Energy")
        plt.grid('True')

        write_dir = write_dir / all_str
        write_dir.mkdir(exist_ok=True)

        fig.savefig(write_dir / f'{all_str} mixing energy.png', bbox_inches='tight')
        
    def plot_ternary_mixing_energy(self, write_dir: Path, elements: tuple[str]):
        pass

    def plot_mixing_energy(self, write_dir: Path):
        with open(write_dir / 'mixing energies.out', 'w') as f:
            f.write("End Member Energies [eV]\n\n")
            for end_mem, engy in self.end_member_energy.items():
                f.write(f"{end_mem} = {engy:2.5f}\n")

            f.write('\nMixing Energies [eV]\n\n')
            for conf, engy in self.mixing_energies.items():
                f.write(f"{conf.cid}\n")

                comp_line = ""
                for el, conc in conf.composition.items():
                    comp_line += f"{el}={conc:1.3f}  "
                f.write(comp_line[:-2] + '\n')

                f.write(f"E/N  {conf.norm_energy:2.8f}\n")
                f.write(f"emix  {engy:1.6f}\n")

        if len(self.end_members) == 2:
            self.plot_binary_mixing_energy(write_dir, tuple(self.end_members.keys()))

        elif len(self.end_members) == 3:
            for binary in itertools.combinations(self.end_members.keys(), 2):
                self.plot_binary_mixing_energy(write_dir, binary)

            self.plot_ternary_mixing_energy(write_dir, tuple(self.end_members.keys()))
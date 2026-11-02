from copy import deepcopy
import os, time, logging
from pathlib import Path
import matplotlib.pyplot as plt

from configuration import Configuration

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
        for conf in self.configs.items():
            self.max_rank = max(self.max_rank, conf.rank)

        self.num_available_cores = NTASKS

        self.end_member_mixing_energy: dict[Configuration, float] = {}
        self.mixing_energies = {}

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

            conf.mixing_energy = conf.energy / len(self.supercell)
            self.end_member_mixing_energy[el] = conf.mixing_energy

        # compute mixing energies
        for cid, conf in self.configs.items():
            conf.compute_composition()

            if conf.rank == 1:
                continue
            elif conf.energy is None:
                continue
            else:
                conf.mixing_energy = conf.energy / len(conf.supercell)
                for sp, conc in conf.composition.items():
                    conf.mixing_energy -= conc*self.end_members[sp].mixing_energy

            self.mixing_energies[conf] = conf.mixing_energy

    def plot_binary_mixing_energy(self, write_dir: Path, x_element: str):
        x, y = [], []
        for conf, me in self.end_member_mixing_energy.items():
            if conf.rank != 2:
                continue

            x.append(conf.composition[x[x_element]])
            y.append(self.mixing_energies)

        fig = plt.figure()
        plt.scatter(x, y)
        fig.savefig(write_dir / 'binary mixing energy.png', bbox_inches='tight')

    def plot_ternary_mixing_energy(self):
        pass
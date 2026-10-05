from configuration import Configuration
from copy import deepcopy
import os, time, logging

try:
    NTASKS = int(os.environ['SLURM_NTASKS'])
except:
    NTASKS = 1

logger = logging.getLogger('Gluster')

class Dataset:
    def __init__(self, configs: dict[int, Configuration] = None):
        self.configs: dict[int, Configuration] = {}
        self.queue: dict[int, Configuration] = {}
        self.num_available_cores = NTASKS

        if configs:
            self.update(configs)

        self.end_members: dict[str, Configuration] = {}

    def update(self, configs: dict[int, Configuration]):
        for cid, conf in configs.items():
            if cid not in self.configs.keys():
                self.configs[cid] = conf

        for cid, conf in self.configs.items():
            if conf.status == 0:
                self.queue[cid] = conf

    def run_jobs(self):
        if len(self.queue) == 0:
            logger.debug(f"Queue is empty, skipping VASP")

        while len(self.queue):
            # launch jobs
            for cid, conf in self.queue.items():
                time.sleep(0.5) # give slurm time to add steps
                if conf.status == 1 or conf.status == 2:
                    continue

                if conf.ncores > NTASKS:
                    raise ValueError(f"Too many cores requested for config {cid}. Need {conf.ncores} but only have a maximum of {NTASKS}")
                elif self.num_available_cores < conf.ncores:
                    continue
                
                # prepare directory and start job
                write_success = conf.write_vasp_files(conf.cdir)
                for fn, value in write_success.items():
                    if value is False:
                        conf.status = 2
                        logger.debug(f"Could not write {fn} for config {cid}. Skipping it")
                        continue

                conf.run_vasp()

                self.num_available_cores -= conf.ncores
                conf.status = 1
                logger.debug(f'Running VASP on config {cid} with {conf.ncores} cores (# cores ready: {self.num_available_cores})')
            
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
        # add unique species which appear across all compositions
        for cid, conf in self.configs.items():
            for sp in conf.comp.keys():
                self.end_members[sp] = None

        # make sure end member energies exist
        for cid, conf in self.configs.items():
            if conf.comp:
                sp = [sp for sp in conf.comp.keys()][0]
                if len(conf.comp) == 1:
                    if not all([conf.energy, conf.Atoms]):
                        raise ValueError(f"Need energy for end-member {next(iter(conf.comp))} (config {cid}) for calculating mixing energy, but VASP may not have run")
                    else:
                        conf.mixing_energy = conf.energy / len(conf.Atoms)
                        self.end_members[sp] = conf

        # compute mixing energies
        for cid, conf in self.configs.items():
            if len(conf.comp) == 1:
                continue
            else:
                num_atoms = len(conf.Atoms)
                conf.mixing_energy = conf.energy / num_atoms
                for sp, cnt in conf.comp.items():
                    conf.mixing_energy -= (cnt/num_atoms)*self.end_members[sp].mixing_energy
from copy import deepcopy
import os, time, logging

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

        self.num_available_cores = NTASKS

    def run_jobs(self):
        if len(self.queue) == 0:
            logger.debug(f"Queue is empty, skipping VASP")

        while len(self.queue):
            # launch jobs
            for cid, conf in self.queue.items():
                time.sleep(0.5) # give slurm time to add job steps
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
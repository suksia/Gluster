import subprocess, math, logging, os, time
from pathlib import Path

from vasp_file import VaspIncar, VaspKpoints, VaspPoscar, VaspPotcar, VaspOutcar

try:
    NTASKS = int(os.environ['SLURM_NTASKS'])
except:
    NTASKS = 1

logger = logging.getLogger('Gluster')


class Calculation:
    def __init__(self, cid: int, cdir: Path, vasp_file_lines: dict, ntasks: int = 1, kpar: int = 1, ncore: int = 1, dry_run = False):
        self.cid = cid
        
        self.incar = VaspIncar().load(from_string=vasp_file_lines['incar'])
        self.kpoints = VaspKpoints().load(from_string=vasp_file_lines['kpoints'])
        self.poscar = VaspPoscar().load(from_string=vasp_file_lines['poscar'])
        self.potcar = VaspPotcar().load(dirnames=vasp_file_lines['potcar'])

        self.ntasks = ntasks
        self.kpar = kpar
        self.ncore = ncore

        self.incar.update_setting('KPAR', self.kpar)
        self.incar.update_setting('NCORE', self.ncore)

        self.status = 0  # 0 = not run, 1 = running, 2 = finished
        self.process: subprocess.Popen = None
        self.dry_run = dry_run

        self.outcar: VaspOutcar = None
        self.elapsed_time: float = None
        self.energy: float = None

        self.dir = cdir
        self.dir.mkdir(parents=True, exist_ok=True)

        self.incar.write(self.dir / 'INCAR')
        self.poscar.write(self.dir / 'POSCAR')
        self.kpoints.write(self.dir / 'KPOINTS')
        self.potcar.write(self.dir / 'POTCAR')

    def run_vasp(self):
        self.outfile = open(self.dir / 'vasp_std.out', 'w')

        vasp_cmd = ['srun', '-N 1', f'-n {self.ntasks}', '--export=ALL', '--cpu-bind=cores', 'vasp_std']
        if self.dry_run:
            vasp_cmd.append('--dry-run')

        self.process = subprocess.Popen(vasp_cmd, cwd=self.dir, stdout=self.outfile, stderr=subprocess.STDOUT)
        self.status = 1

    def poll_process(self):
        # process can not be polled
        if self.status < 1 or self.process is None:
            return
        
        poll = self.process.poll()
        if poll == 0:
            self.status = 2
            self.outcar = VaspOutcar().load(from_path = self.dir / 'OUTCAR')

            if not self.dry_run:
                self.energy = self.outcar.get_energy()
                self.elapsed_time = self.outcar.get_elapsed_time()
                self.outfile.write(f'Elapsed time: {math.floor(self.elapsed_time/3600)}hrs {math.floor((self.elapsed_time/60)%60)}min {round((self.elapsed_time%60))}sec')

            self.outfile.close()


def run_vasp_in_parallel(queue: dict[int, Calculation]):
    num_available_cores = NTASKS

    while len(queue):
        # launch jobs
        for cid, calc in queue.items():
            if calc.status == 1 or calc.status == 2:
                continue

            if calc.ntasks > NTASKS:
                raise ValueError(f"Too many cores requested for calculation {cid}. Need {calc.ntasks} but only have a maximum of {NTASKS}")
            elif num_available_cores < calc.ntasks:
                continue

            calc.run_vasp()

            num_available_cores -= calc.ntasks
            calc.status = 1
            logger.debug(f'Running VASP for calculation {cid} with {calc.ntasks} cores (# cores ready: {num_available_cores})')
            
            time.sleep(1) # give slurm time to add job steps
        
        # check statuses
        finished = []
        for cid, calc in queue.items():
            calc.poll_process()
            if calc.status == 1:
                continue
            elif calc.status == 2:
                num_available_cores += calc.ntasks
                finished.append(cid)
                logger.debug(f'VASP finished for calculation {cid} (# cores ready: {num_available_cores})')

        for fin_cid in finished:
            queue.pop(fin_cid)
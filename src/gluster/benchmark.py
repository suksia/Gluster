# input file

import argparse, logging, sys, yaml, subprocess, time, math, os
from pathlib import Path
from copy import copy, deepcopy

from vasp_file import VaspIncar, VaspKpoints, VaspPoscar, VaspPotcar, VaspOutcar

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger('Gluster')

try:
    NTASKS = int(os.environ['SLURM_NTASKS'])
except:
    NTASKS = 1

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='Path to input file')
parser.add_argument('--dry-run', '-D', action=argparse.BooleanOptionalAction)
args = parser.parse_args()

input_fp = Path(args.input).resolve()

if not input_fp.exists():
    raise FileNotFoundError(f"[{input_fp}] File could not be found")

with open(input_fp, 'r') as f:
    input_yml: dict = yaml.safe_load(f)

with open(input_fp, 'r') as f:
    input_yml_lines = f.readlines()

logger.debug(f'Loaded input file {input_fp}')

if args.dry_run is None:
    args.dry_run = False


TOP_DIR: Path = Path(input_yml['dir']) / input_yml['name']

if TOP_DIR.parent.exists() is False:
    raise ValueError(f'Directory {TOP_DIR.parent} does not exist')

TOP_DIR.mkdir(exist_ok=True)

with open(TOP_DIR / 'last_input.yaml', 'w') as f:
    f.writelines(input_yml_lines)

logger.debug(f'Created benchmark directory at {TOP_DIR.as_posix()}')


vasp_file_lines = {
    'incar': input_yml['incar'],
    'kpoints': input_yml['kpoints'],
    'poscar': input_yml['poscar'],
    'potcar': input_yml['potcar'],
}

class Calculation:
    def __init__(self, cid: int, vasp_file_lines: dict, ntasks: int, kpar: int, ncore: int):
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

        self.outcar: VaspOutcar = None
        self.elapsed_time: float = None
        self.energy: float = None

        self.dir = TOP_DIR / f"ntasks{ntasks}" / f"kpar{kpar}_ncore{ncore}"
        self.dir.mkdir(parents=True, exist_ok=True)

        self.incar.write(self.dir / 'INCAR')
        self.poscar.write(self.dir / 'POSCAR')
        self.kpoints.write(self.dir / 'KPOINTS')
        self.potcar.write(self.dir / 'POTCAR')

    def run_vasp(self):
        self.outfile = open(self.dir / 'vasp_std.out', 'w')

        vasp_cmd = ['srun', '-N 1', f'-n {self.ntasks}', '--export=ALL', '--cpu-bind=cores', 'vasp_std']
        if args.dry_run:
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

            if not args.dry_run:
                self.energy = self.outcar.get_energy()
                self.elapsed_time = self.outcar.get_elapsed_time()
                self.outfile.write(f'Elapsed time: {math.floor(self.elapsed_time/3600)}hrs {math.floor((self.elapsed_time/60)%60)}min {round((self.elapsed_time%60))}sec')
                
            self.outfile.close()


if args.dry_run:
    calc = Calculation(0, vasp_file_lines, 1, 1, 1)
    calc.run_vasp()

    while calc.status != 2:
        calc.poll_process()
        time.sleep(1)

    logger.debug(f"Dry run calculation finished. Found {calc.outcar.get_num_ibz_kpoints()} irreducible k-points")

    exit()


calculations: dict[int, Calculation] = {}

cid = 0
for ntasks in input_yml['ntasks']:
    for kpar in range(1, ntasks+1):
        if ntasks % kpar != 0:
            continue

        a = int(ntasks/kpar)
        for ncore in range(1, a+1):
            if a % ncore != 0:
                continue

            calculations[cid] = Calculation(cid, vasp_file_lines, ntasks, kpar, ncore)
            cid += 1

queue: dict[int, Calculation] = copy(calculations)
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


with open(TOP_DIR / 'benchmark.out', 'w') as f:
    f.write(input_yml['name'] + '\n\n')
    
    f.write(" "*5 + f"{'NTASKS':<10} {'KPAR':<10f} {'NCORE':<10f} {'time [sec]':<10f}\n")

    for ntasks in input_yml['ntasks']:
        for cid, calc in calculations.items():
            if calc.ntasks != ntasks:
                continue

            f.write(f"{cid:<5f} {ntasks:<10f} {calc.kpar:<10f} {calc.ncore:<10f} {calc.elapsed_time:<8.2f}\n")
        f.write('\n')
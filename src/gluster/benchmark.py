# input file

import argparse, logging, sys, yaml, time
from pathlib import Path
from copy import copy

from calculation import Calculation, run_vasp_in_parallel

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger('Gluster')


######################## input ########################

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


################## project directory ##################

TOP_DIR: Path = Path(input_yml['dir']) / input_yml['name']

if TOP_DIR.parent.exists() is False:
    raise ValueError(f'Directory {TOP_DIR.parent} does not exist')

TOP_DIR.mkdir(exist_ok=True)

with open(TOP_DIR / 'last_input.yaml', 'w') as f:
    f.writelines(input_yml_lines)

logger.debug(f'Created benchmark directory at {TOP_DIR.as_posix()}')


####################### run vasp ######################

vasp_file_lines = {
    'incar': input_yml['incar'],
    'kpoints': input_yml['kpoints'],
    'poscar': input_yml['poscar'],
    'potcar': input_yml['potcar'],
}

if args.dry_run:
    calc = Calculation(0, TOP_DIR / 'dry', vasp_file_lines, 1, 1, 1, dry_run=True)
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

            cdir = TOP_DIR / f'{ntasks}' / f'kpar{kpar}_ncore{ncore}'
            calculations[cid] = Calculation(cid, cdir, vasp_file_lines, ntasks, kpar, ncore)
            cid += 1

queue: dict[int, Calculation] = copy(calculations)

run_vasp_in_parallel(queue)


######################## output #######################

benchmark_path = TOP_DIR / 'benchmark.out'

if not benchmark_path.exists():
    with open(benchmark_path, 'w') as f:
        f.write(input_yml['name'] + '\n\n')
        f.write(f"{'ID':<10} {'NTASKS':<10} {'KPAR':<10} {'NCORE':<10} {'time [sec]':<10}")

with open(benchmark_path, 'a') as f:
    f.write('\n\n')
    for ntasks in input_yml['ntasks']:
        for cid, calc in calculations.items():
            if calc.ntasks != ntasks:
                continue

            f.write(f"{cid:<10} {ntasks:<10} {calc.kpar:<10} {calc.ncore:<10} {calc.elapsed_time:<10.2f}\n")
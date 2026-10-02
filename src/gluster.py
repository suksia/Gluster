import argparse, yaml, time, logging, sys
from pathlib import Path
from copy import deepcopy
from configuration import Configuration, NTASKS

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('Gluster')

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='Path to input file')
parser.add_argument('--check-basis', action=argparse.BooleanOptionalAction)
args = parser.parse_args()

input_fp = Path(args.input).resolve()
assert input_fp.exists(), f'[{input_fp}] File does not exist'

with open(input_fp, 'r') as f:
    input_yml: dict = yaml.safe_load(f)
with open(input_fp, 'r') as f:
    input_yml_lines = f.readlines()

logger.debug(f'Loaded input file {input_fp}')

# create directory
top_dir = Path(input_yml['dir']) / input_yml['name']

if top_dir.parent.exists() is False:
    raise ValueError(f'Directory {top_dir.parent} does not exist')

top_dir.mkdir(exist_ok=True)

# ----------------- configurations ----------------- #

configs = deepcopy(input_yml['configurations'])

for conf_id, conf_dict in configs.items():
    conf_dir = top_dir / str(conf_id)
    conf_dir.mkdir(exist_ok=True)
    configs[conf_id] = Configuration(conf_id, conf_dir, conf_dict)

logger.debug(f'Initialized {len(configs)} configurations')

Configuration.write_basis(top_dir / 'basis.out')
logger.debug(f"Wrote computed bases to {top_dir / 'basis.out'}")

if args.check_basis:
    logger.debug(f'Detected --check-basis flag. Exiting...')
    sys.exit()

logger.debug(f'Initialized configurations in {top_dir}')

# ----------------- run VASP ----------------- #
"""
# queue configs
queue = {}
for cid, conf in configs.items():
    if conf.status == 0:
        queue[cid] = conf

# starting launch jobs from the queue
num_available_cores = NTASKS

while len(queue):
    # launch jobs
    for cid, conf in queue.items():
        if conf.status == 1 or conf.status == 2:
            continue

        if conf.ncores > NTASKS:
            raise ValueError(f"Too many cores requested for config {cid}. Need {conf.ncores} but only have a maximum of {NTASKS}")
        elif num_available_cores < conf.ncores:
            continue
        
        # prepare directory and start job
        conf.create_lattice()
        conf.wipe_cdir()
        conf.write_vasp_files(conf.cdir)
        conf.run_vasp()

        num_available_cores -= conf.ncores
        conf.status = 1
        logger.debug(f'Running VASP on config {cid} with {conf.ncores} cores (available cores: {num_available_cores})')
        time.sleep(0.5)
    
    # check statuses
    finished = []
    for cid, conf in queue.items():
        conf.poll_process()
        if conf.status == 1:
            continue
        elif conf.status == 2:
            num_available_cores += conf.ncores
            finished.append(cid)
            logger.debug(f'VASP finished for config {cid} (available cores: {num_available_cores})')

    for fin_cid in finished:
        queue.pop(fin_cid)

# create database for cluster expansion
"""
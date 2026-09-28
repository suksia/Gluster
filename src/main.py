import argparse, yaml, time
from pathlib import Path
from copy import deepcopy
from configuration import Configuration, NTASKS

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='Path to input file')
args = parser.parse_args()

input_fp = Path(args.input).resolve()
assert input_fp.exists(), f'[{input_fp}] File does not exist'

with open(input_fp, 'r') as f:
    input_yml: dict = yaml.safe_load(f)

# create directory
top_dir = Path(input_yml['dir']) / input_yml['name']

if top_dir.parent.exists() is False:
    raise ValueError(f'Directory {top_dir.parent} does not exist')

top_dir.mkdir(exist_ok=True) 

# initialize configurations
configs = deepcopy(input_yml['configurations'])

for conf_id, conf_dict in configs.items():
    # create directory
    conf_dir = top_dir / str(conf_id)
    conf_dir.mkdir(exist_ok=True)

    # initialize configuration object
    if 'a0' not in conf_dict.keys():
        conf_dict['a0'] = None
    if 'order' not in conf_dict.keys():
        conf_dict['order'] = None

    configs[conf_id] = Configuration(conf_id, conf_dir,conf_dict)

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
            raise ValueError(f"[Config {conf.cid}] Too many cores requested. Need {conf.ncores} but only have {NTASKS}")
        elif num_available_cores < conf.ncores:
            continue
        
        # prepare directory and start job
        conf.create_lattice()
        conf.wipe_cdir()
        conf.write_vasp_files(conf.cdir)
        conf.run_vasp()

        num_available_cores -= conf.ncores
        conf.status = 1
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
    
    for fin_cid in finished:
        queue.pop(fin_cid)

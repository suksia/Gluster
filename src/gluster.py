import argparse, yaml, time, logging, sys
from pathlib import Path
from copy import deepcopy
from configuration import Configuration, check_configuration_dict
from dataset import Dataset
from expansion import Expansion

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('Gluster')


######################## input ########################

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='Path to input file')
parser.add_argument('--check-basis', action=argparse.BooleanOptionalAction)
parser.add_argument('--run-vasp', action=argparse.BooleanOptionalAction)
args = parser.parse_args()

input_fp = Path(args.input).resolve()
assert input_fp.exists(), f'[{input_fp}] File does not exist'

with open(input_fp, 'r') as f:
    input_yml: dict = yaml.safe_load(f)
with open(input_fp, 'r') as f:
    input_yml_lines = f.readlines()

logger.debug(f'Loaded input file {input_fp}')


################## project directory ##################

top_dir: Path = Path(input_yml['dir']) / input_yml['name']

if top_dir.parent.exists() is False:
    raise ValueError(f'Directory {top_dir.parent} does not exist')

top_dir.mkdir(exist_ok=True)


#################### configurations ###################

configs: dict[int, Configuration] = deepcopy(input_yml['configurations'])

for conf_id, conf_dict in configs.items():
    conf_dir = top_dir / str(conf_id)
    conf_dir.mkdir(exist_ok=True)
    conf_dict = check_configuration_dict(conf_dict)
    configs[conf_id] = Configuration(conf_id, conf_dir, conf_dict)

logger.debug(f'Initialized {len(configs)} configurations in {top_dir}')

if args.check_basis:
    logger.debug(f'Detected --check-basis flag. Exiting...')
    sys.exit()


####################### run VASP ######################

dataset = Dataset(configs)
dataset.run_jobs()

if args.check_basis:
    logger.debug(f'Detected --run-vasp flag. Exiting...')
    sys.exit()


######################## fit CE #######################

if 'a0' not in input_yml.keys():
    ce_a0 = 3.15
else:
    ce_a0 = input_yml['a0']

if 'cutoffs' not in input_yml.keys():
    raise KeyError(f"Reached Cluster Space initialization, but no cutoffs provided")
else:
    cutoffs = [float(cf) for cf in input_yml['cutoffs']]

expansion = Expansion(dataset, ce_a0, cutoffs)
expansion.init_cluster_space()
expansion.init_structure_container()
expansion.fit(top_dir / 'expansion.ce')
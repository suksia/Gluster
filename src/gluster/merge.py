import argparse, logging, sys, yaml
from pathlib import Path

from dataset import Dataset
from configuration import Configuration, compare_configurations

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger('Gluster')

logging.getLogger("matplotlib").setLevel(logging.FATAL)


######################## input ########################

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='Path to input file')
args = parser.parse_args()

input_fp = Path(args.input).resolve()

if not input_fp.exists():
    raise FileNotFoundError(f"[{input_fp}] File could not be found")

with open(input_fp, 'r') as f:
    input_yml: dict = yaml.safe_load(f)

with open(input_fp, 'r') as f:
    input_yml_lines = f.readlines()

logger.debug(f'Loaded input file {input_fp}')


################## project directory ##################

TOP_DIR: Path = Path(input_yml['dir']) / input_yml['name']

if TOP_DIR.parent.exists() is False:
    raise ValueError(f'Directory {TOP_DIR.parent} does not exist')

TOP_DIR.mkdir(exist_ok=True)

logger.debug(f'Created or found project directory at {TOP_DIR.as_posix()}')


################ starting configs list ################

dataset = Dataset(input_yml['alloy'])

if 'datasets' not in input_yml.keys():
    raise KeyError("List of datasets missing in input file")

configs_path: dict[Configuration, Path] = {}
configs_list: list[Configuration] = []

for in_dataset_path in input_yml['datasets']:
    in_dataset_path = Path(in_dataset_path).resolve()

    if not in_dataset_path.exists():
        raise ValueError(f"Could not find dataset YAML file at {in_dataset_path}")

    new_configs = dataset.read_yaml(in_dataset_path)
    
    configs_list += new_configs

    for config in new_configs:
        configs_path[config] = in_dataset_path

    logger.debug(f'Read in configurations from {in_dataset_path}')


###################### duplicates #####################

unique_configs: list[Configuration] = []

logger.debug('Searching for duplicate configs')

# move configs from configs_list to unique_configs and remove duplicates
while len(configs_list):
    logger.debug(f'Configs left: {len(configs_list)}')

    config = configs_list[0]

    save_config = config

    duplicates: list[Configuration] = []
    
    for other_config in configs_list[1:]:
        dupe = compare_configurations(config, other_config)

        if dupe:
            duplicates.append(other_config)

    energies = {config: config.energy}

    if len(duplicates):
        for duplicate in duplicates:
            energies[duplicate] = duplicate.energy

            conf_in_path = configs_path[config]
            other_conf_in_path = configs_path[duplicate]

            logger.debug(f"Config {config.in_cid} in {conf_in_path} is a duplicate of {duplicate.in_cid} in {other_conf_in_path}")

        # use first config with a defined energy
        for conf, energy in energies.items():
            if energy is not None:
                save_config = conf
                break

        logger.debug(f"Using {save_config.in_cid} in {configs_path[save_config]}, now indexed as {save_config.cid}")

        new_configs_list = []

        # skip duplicates
        for i, conf in enumerate(configs_list):
            if conf not in duplicates:
                new_configs_list.append(conf)
            
        configs_list = new_configs_list

    unique_configs.append(save_config)
    configs_list.pop(0)

dataset.add_configurations(unique_configs)
dataset.write_yaml(TOP_DIR / 'merged_dataset.yaml')

logger.debug(f"Wrote {len(unique_configs)} unique configs to {TOP_DIR / 'merged_dataset.yaml'}")
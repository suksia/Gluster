import argparse, yaml, time, logging, sys
from pathlib import Path
from copy import deepcopy
from configuration import Configuration, NTASKS, check_configuration_dict
from ase import build
from icet import ClusterSpace, StructureContainer, ClusterExpansion
from trainstation import CrossValidationEstimator

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('Gluster')

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

# create directory
top_dir = Path(input_yml['dir']) / input_yml['name']

if top_dir.parent.exists() is False:
    raise ValueError(f'Directory {top_dir.parent} does not exist')

top_dir.mkdir(exist_ok=True)

# ----------------- configurations ----------------- #

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

# ----------------- run VASP ----------------- #

# queue configs
queue: dict[int, Configuration] = {}
for cid, conf in configs.items():
    if conf.status == 0:
        queue[cid] = conf

# starting launch jobs from the queue
num_available_cores = NTASKS

while len(queue):
    time.sleep(0.5)
    # launch jobs
    for cid, conf in queue.items():
        if conf.status == 1 or conf.status == 2:
            continue

        if conf.ncores > NTASKS:
            raise ValueError(f"Too many cores requested for config {cid}. Need {conf.ncores} but only have a maximum of {NTASKS}")
        elif num_available_cores < conf.ncores:
            continue
        
        # prepare directory and start job
        write_success = conf.write_vasp_files(conf.cdir)
        for fn, value in write_success.items():
            if value is False:
                conf.status = 2
                logger.debug(f"Could not write {fn} for config {cid}. Skipping it")
                continue

        conf.run_vasp()

        num_available_cores -= conf.ncores
        conf.status = 1
        logger.debug(f'Running VASP on config {cid} with {conf.ncores} cores (available cores: {num_available_cores})')
    
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

if args.check_basis:
    logger.debug(f'Detected --run-vasp flag. Exiting...')
    sys.exit()

# check which configs can be included in CE fitting, also compile a list of unique end-members
end_members = {}
for cid, conf in configs.items():
    conf.load_energy()
    if all([conf.energy, conf.Atoms, conf.include_fit]):
        conf.include_fit = True
        for sp in conf.comp.keys():
            end_members[sp] = None

# make sure end member energies exist
for cid, conf in configs.items():
    if conf.comp:
        species = [sp for sp in conf.comp.keys()]
        if len(species) == 1 and species[0] in end_members.keys():
            if conf.include_fit is False:
                raise ValueError(f"Need energies end-member {next(iter(conf.comp))} (config {cid}) for calculating mixing energy, but VASP may not have run")
            else:
                conf.mixing_energy = conf.energy / len(conf.Atoms)
                end_members[species[0]] = conf.mixing_energy

# compute mixing energies and add them to the rest of the database
for cid, conf in configs.items():
    if conf.include_fit:
        if len(conf.comp) == 1:
            continue
        num_atoms = len(conf.Atoms)
        conf.mixing_energy = conf.energy / num_atoms
        for sp, cnt in conf.comp:
            conf.mixing_energy -= (cnt/num_atoms)*end_members[sp]

# cluster space initialization
if 'a0' not in input_yml.keys():
    ce_a0 = 3.15
else:
    ce_a0 = input_yml['a0']

if 'cutoffs' not in input_yml.keys():
    raise KeyError(f"Reached ClusterSpace initialization, but no cutoffs provided")
else:
    cutoffs = [float(cf) for cf in input_yml['cutoffs']]

bcc_prim = build.bulk('W', 'bcc', ce_a0)

cs = ClusterSpace(structure=bcc_prim,
                  cutoffs=cutoffs,
                  chemical_symbols=[em for em in end_members.keys()])

# structure container
sc = StructureContainer(cluster_space=cs)
for conf in configs.values():
    if conf.include_fit:
        sc.add_structure(structure=conf.Atoms,
                        user_tag=conf.name,
                        properties={'mixing_energy': conf.mixing_energy})

# fit cluster expansion
opt = CrossValidationEstimator(
    fit_data=sc.get_fit_data(key='mixing_energy'), fit_method='ardr')
opt.validate()
opt.train()

ce = ClusterExpansion(cluster_space=cs, parameters=opt.parameters, metadata=opt.summary)
ce.write(top_dir / 'cluster_expansion.out')
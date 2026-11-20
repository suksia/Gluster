import argparse, yaml, logging, sys
from pathlib import Path
import numpy as np

from utils import ELEMENTS
from enumeration import BCC
from sampling import Sampler
from vasp_file import VaspIncar
from dataset import Dataset
from output import plot_mesh, plot_mixing_energy, write_mixing_energy, write_xyz
from expansion import Expansion, scan_cutoffs

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger('Gluster')


######################## input ########################

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='Path to input file')
parser.add_argument('--enumerate', '-E', action=argparse.BooleanOptionalAction)
parser.add_argument('--dft', '-D', action=argparse.BooleanOptionalAction)
parser.add_argument('--train', '-T', action=argparse.BooleanOptionalAction)
parser.add_argument('--hull', '-H', action=argparse.BooleanOptionalAction)
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

with open(TOP_DIR / 'last_input.yaml', 'w') as f:
    f.writelines(input_yml_lines)

logger.debug(f'Created project directory at {TOP_DIR.as_posix()}')


#################### enumeration ###################

if 'alloy' not in input_yml.keys():
    raise KeyError(f"Alloy end-members are missing. They must be a list of elements")

end_members: list[str] = input_yml['alloy']
dft_dataset = Dataset(end_members)

end_members_a0 = 0.0
for em in end_members:
    end_members_a0 += ELEMENTS[em]['a0'] / len(end_members)

if args.enumerate:
    logger.debug(f'Enumeration flag set')

    enum_dir = TOP_DIR / 'enumeration'
    enum_dir.mkdir(exist_ok=True)

    if 'enumerate' not in input_yml.keys():
        raise KeyError(f"Enumeration section missing in input file")
    
    enum: dict = input_yml['enumerate']

    if 'seed' not in enum.keys():
        seed = None
    else:
        seed = enum['seed']

    sampler = Sampler(seed)
    lattice = BCC(enum['L'])

    unaries = dft_dataset.enumerate_unaries()
    dft_dataset.add_configurations(unaries)
    
    all_compositions = dft_dataset.enumerate_compositions(enum['L'])
    dft_dataset.compositions = all_compositions

    logger.debug(f"Enumerated all compositions. Found {len(all_compositions)} possibilities")

    dft_dataset.sublattices = dft_dataset.enumerate_sublattices(lattice)
    num_sublattices = sum([len(sublat_list) for sublat_list in dft_dataset.sublattices.values()])

    logger.debug(f"Enumerated all sublattices. Found {num_sublattices} possibilities")

    logger.debug(f"Enumerating all configurations. This may take a while...") 

    enum_configs_dict, enum_configs_list = dft_dataset.enumerate_configurations()

    logger.debug(f"Enumerated all configurations. Found {len(enum_configs_list)} possibilities")

    all_configs_dataset = Dataset(end_members)
    all_configs_dataset.add_configurations(enum_configs_list)

    all_configs_dataset.sort_configurations()
    all_configs_dataset.write_yaml(enum_dir / 'enumerated_configurations.yaml')

    logger.debug(f"Wrote all enumerated configurations to {enum_dir / 'enumerated_configurations.yaml'}")

    ordered_configs = dft_dataset.sample_ordered_configurations(sampler, enum['N_ordered'], enum_configs_dict, enum_configs_list)
    disordered_configs = dft_dataset.sample_disordered_configurations(sampler, enum['N_disordered'], enum['disordered_size'])

    logger.debug(f"Sampled {len(ordered_configs) + len(disordered_configs)} total configurations")

    dft_dataset.add_configurations(ordered_configs)
    dft_dataset.add_configurations(disordered_configs)

    dft_dataset.sort_configurations()
    dft_dataset.write_yaml(enum_dir / 'dft_dataset.yaml')

    logger.debug(f"Wrote sampled configurations to {enum_dir / 'dft_dataset.yaml'}")

    plot_mesh(enum_dir / 'enumerated_compositions.png', end_members, compositions=all_compositions)
    plot_mesh(enum_dir / 'ordered_configurations.png', end_members, configurations=ordered_configs)
    plot_mesh(enum_dir / 'disordered_configurations.png', end_members, configurations=disordered_configs)

    logger.debug("Plotted composition space and sampled configurations")

    exit()  


####################### run VASP ######################

if args.dft:
    logger.debug('DFT flag set')

    dft_dir = TOP_DIR / 'dft'
    dft_dir.mkdir(exist_ok=True)

    if 'vasp' not in input_yml.keys():
        raise KeyError(f"VASP section missing in input file")

    vasp: dict = input_yml['vasp']

    if 'dataset' not in vasp.keys():
        dataset_path = TOP_DIR / 'enumeration' / 'dft_dataset.yaml'
    else:
        dataset_path = Path(vasp['dataset'])

    if not dataset_path.exists():
        raise ValueError(f"[{dataset_path}] File not found")

    loaded_configs = dft_dataset.read_yaml(dataset_path)
    dft_dataset.add_configurations(loaded_configs)
    dft_dataset.sort_configurations()

    logger.debug(f"Loaded {len(loaded_configs)} configurations from {dataset_path}")

    for config in dft_dataset.configurations:
        if config.ordered:
            setting_cls = 'ordered'
        else:
            setting_cls = 'disordered'

        config.create_poscar()

        if 'potcar_dir' in vasp['potcar'].keys():
            potcar_dir = Path(vasp['potcar']['dirpath'])
        else:
            potcar_dir = Path('/storage/group/xvw5285/default/atomistic_pkgs/potpaw_PBE.54/')

        config.create_potcar(potcar_dir, vasp['potcar']['names'])

        config.ncores = vasp[setting_cls]['ncores']

        incar = VaspIncar().load(from_string=vasp['incar'])
        incar.update_setting("KPAR", vasp[setting_cls]['KPAR'])
        incar.update_setting("NCORE", vasp[setting_cls]['NCORE'])

        if "NBANDS" in vasp[setting_cls].keys():
            config.get_nbands(*vasp[setting_cls]['NBANDS'])
            incar.update_setting("NBANDS", config.nbands)
        
        config.set_vasp_files(incar=incar)
        config.set_dir(dft_dir / str(config.cid))
        config.write_vasp_files()

        if config.ordered:
            if 'xyz_size' in vasp['ordered'].keys():
                xyz_size = vasp['ordered']['xyz_size']
            else:
                xyz_size = 10

            config.supercell = config.map_to_supercell(xyz_size)
            write_xyz(dft_dir / str(config.cid) / 'supercell.xyz', config.supercell)

    dft_dataset.queue_jobs()
    dft_dataset.run_jobs(dft_dir / 'dft_dataset.yaml')

    dft_dataset.compute_mixing_energies()

    plot_mixing_energy(TOP_DIR / 'dft_mixing_energies.png', end_members, dft_dataset)
    write_mixing_energy(TOP_DIR / 'dft_mixing_energies.out', end_members, dft_dataset)

    logger.debug(f'Computed, wrote, and plotting mixing energies')


######################## fit CE #######################

if args.train:
    logger.debug('Cluster expansion flag set')
    
    expan_dir = TOP_DIR / 'expansion'
    expan_dir.mkdir(exist_ok=True)

    if 'expansion' not in input_yml.keys():
        raise KeyError(f"Expansion section missing in input file")

    expan: dict = input_yml['expansion']

    if 'dataset' not in expan.keys():
        dataset_path = TOP_DIR / 'dft' / 'dft_dataset.yaml'
    else:
        dataset_path = Path(expan['dataset'])
    
    if not dataset_path.exists():
        raise ValueError(f"[{dataset_path}] File not found")
    
    loaded_configs = dft_dataset.read_yaml(dataset_path)
    dft_dataset.add_configurations(loaded_configs)
    dft_dataset.sort_configurations()

    logger.debug(f"Loaded {len(loaded_configs)} configurations from {dataset_path}")

    if 'cutoffs' not in expan.keys():
        cutoffs = []
    else:
        cutoffs = expan['cutoffs']

    num_tot_configs = len(dft_dataset.configurations)
    dft_dataset.remove_configurations_w_no_energy()
    num_configs_no_energy = num_tot_configs - len(dft_dataset.configurations)

    if num_configs_no_energy:
        logger.debug(f"Removed {num_configs_no_energy} configurations from loaded DFT dataset due to missing energy")

    dft_dataset.compute_mixing_energies()

    expansion = Expansion(dataset=dft_dataset, cutoffs=cutoffs, a0=end_members_a0)

    if 'scan' in expan.keys():
        cutoff_vals = np.linspace(*expan['scan']).tolist()
        scan_order = len(cutoffs)+2

        scan_cutoffs(scan_order, cutoff_vals, expansion, expan_dir)

        logger.debug(f"Scanned cluster diameter cutoffs for order {scan_order}")

    if len(cutoffs):
        expansion.init_cluster_space()
        expansion.init_structure_container()
        expansion.fit()
        expansion.write(expan_dir)

        logger.debug(f"Fit cluster expansion with cutoffs {cutoffs}")

        expansion.predict_mixing_energies(dft_dataset)

        plot_mixing_energy(TOP_DIR / 'CE_mixing_energies.png', end_members, dft_dataset)
        write_mixing_energy(TOP_DIR / 'CE_mixing_energies.out', end_members, dft_dataset)

        logger.debug("Predicted mixing energies using cluster expansion")


####################### analysis ######################

if args.hull:
    logger.debug('Convex hull flag set')

    hull_dir = TOP_DIR / 'hull'
    hull_dir.mkdir(exist_ok=True)

    if 'hull' not in input_yml.keys():
        raise KeyError(f"Convex hull section missing in input file")

    hull: dict = input_yml['hull']
    
    lattice = BCC(hull['L'])

    hull_dataset = Dataset(end_members)

    all_compositions = hull_dataset.enumerate_compositions(hull['L'])
    hull_dataset.compositions = all_compositions

    logger.debug(f"Enumerated all compositions. Found {len(all_compositions)} possibilities")

    hull_dataset.sublattices = hull_dataset.enumerate_sublattices(lattice)
    num_sublattices = sum([len(sublat_list) for sublat_list in hull_dataset.sublattices.values()])

    logger.debug(f"Enumerated all sublattices. Found {num_sublattices} possibilities")

    logger.debug(f"Enumerating all configurations. This may take a while...") 

    enum_configs_dict, enum_configs_list = hull_dataset.enumerate_configurations()

    logger.debug(f"Enumerated all configurations. Found {len(enum_configs_list)} possibilities")

    hull_dataset.add_configurations(enum_configs_list)
    hull_dataset.sort_configurations()
    hull_dataset.write_yaml(hull_dir / 'enumerated_configurations.yaml')

    logger.debug(f"Wrote all enumerated configurations to {hull_dir / 'enumerated_configurations.yaml'}")

    if 'expansion' not in hull.keys():
        ce_path = TOP_DIR / 'expansion' / 'cluster_expansion.ce'
    else:
        ce_path = Path(hull['expansion'])
    
    if not ce_path.exists():
        raise ValueError(f"[{ce_path}] File not found")
    
    expansion = Expansion(a0=end_members_a0)
    expansion.rescale_Atoms(hull_dataset)
    expansion.read_expansion(ce_path)

    logger.debug(f"Loaded cluster expansion from {ce_path}")

    expansion.predict_mixing_energies(hull_dataset)

    plot_mixing_energy(hull_dir / 'convex_hull.png', end_members, hull_dataset)
    write_mixing_energy(hull_dir / 'convex_hull.out', end_members, hull_dataset)

    logger.debug(f"Predicted mixing energies using cluster expansion")
import argparse, yaml, logging, sys
from pathlib import Path
import numpy as np

from utils import ELEMENTS
from enumeration import BCC
from sampling import Sampler
from vasp_file import VaspIncar, VaspKpoints
from dataset import Dataset
from output import plot_sampling_parameters, plot_mesh_parameters, plot_mesh, plot_mixing_energy, write_mixing_energy
from expansion import Expansion, scan_cutoffs

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger('Gluster')


######################## input ########################

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='Path to input file')
parser.add_argument('--enumerate', '-E', action=argparse.BooleanOptionalAction)
parser.add_argument('--dft', '-D', action=argparse.BooleanOptionalAction)
parser.add_argument('--train', '-T', action=argparse.BooleanOptionalAction)
parser.add_argument('--analysis', '-A', action=argparse.BooleanOptionalAction)
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

if args.enumerate:
    logger.debug(f'Enumeration flag set. Generating configurations for a new input file')

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

    dft_dataset.enumerate_unaries()
    
    all_compositions = dft_dataset.enumerate_compositions(enum['L'])
    dft_dataset.compositions = all_compositions

    logger.debug(f"Enumerated all compositions. Found {len(all_compositions)} possibilities")

    refined_compositions = dft_dataset.refine_compositions(*enum['mesh'])
    dft_dataset.compositions = refined_compositions

    logger.debug(f"Refined compositions. Reduced down to {len(refined_compositions)} possibilities")

    dft_dataset.sublattices = dft_dataset.enumerate_sublattices(lattice)
    num_sublattices = sum([len(sublat_list) for sublat_list in dft_dataset.sublattices.values()])

    logger.debug(f"Enumerated all sublattices. Found {num_sublattices} possibilities")

    ordered_configs = dft_dataset.sample_ordered_configurations(sampler, enum['N_ordered'], enum['supercell_size'], enum['sampling']['composition'], enum['sampling']['basis_size'])
    disordered_configs = dft_dataset.sample_disordered_configurations(sampler, enum['N_disordered'], enum['supercell_size'])

    logger.debug(f"Sampled {len(ordered_configs) + len(disordered_configs)} total configurations")

    dft_dataset.add_configurations(ordered_configs)
    dft_dataset.add_configurations(disordered_configs)

    dft_dataset.write_yaml(enum_dir / 'dft_dataset.yaml')

    logger.debug(f"Wrote sampled configurations to {enum_dir / 'dft_dataset.yaml'}")

    plot_sampling_parameters(enum_dir / 'sampling_parameters.png', enum['sampling']['composition'], enum['sampling']['basis_size'])
    plot_mesh_parameters(enum_dir / 'mesh_parameters.png', *enum['mesh'])
    
    plot_mesh(enum_dir / 'enumerated_compositions.png', end_members, compositions=all_compositions)
    plot_mesh(enum_dir / 'refined_compositions.png', end_members, compositions=refined_compositions)
    plot_mesh(enum_dir / 'ordered_compositions.png', end_members, configurations=ordered_configs)
    plot_mesh(enum_dir / 'disordered_compositions.png', end_members, configurations=disordered_configs)

    logger.debug(f"Plotted sampling parameters and meshes")

    exit()  


#################### configurations ###################

if 'configs' not in input_yml.keys():
    configs_path = TOP_DIR / 'enumeration' / 'dft_dataset.yaml'
else:
    configs_path = input_yml['configs']

if not configs_path.exists():
    raise ValueError(f"[{configs_path}] File not found")

loaded_configs = dft_dataset.read_yaml(configs_path)
dft_dataset.add_configurations(loaded_configs)

logger.debug(f"Loaded {len(loaded_configs)} configurations from {configs_path}")


####################### run VASP ######################

if args.dft:
    logger.debug('DFT flag set. Running VASP on loaded configurations')
    
    dft_dir = TOP_DIR / 'dft'
    dft_dir.mkdir(exist_ok=True)

    if 'vasp' not in input_yml.keys():
        raise KeyError(f"VASP section missing in input file")

    vasp: dict = input_yml['vasp']

    # organize vasp settings by rank
    vasp_files_by_rank = {'global': None, 'unary': None, 'binary': None, 'ternary': None}
    
    def init_dft_files(user_input: dict):
        vasp_file_dict = {'incar': None, 'kpoints': None, 'potcar': None}

        if 'incar' in user_input.keys():
            vasp_file_dict['incar'] = VaspIncar().load(from_string=user_input['incar'])

        if 'kpoints' in user_input.keys():
            vasp_file_dict['kpoints'] = VaspKpoints().load(from_string=user_input['kpoints'])

        if 'potcar' in user_input.keys():
            vasp_file_dict['potcar'] = user_input['potcar']
    
        return vasp_file_dict
    
    vasp_files_by_rank['global'] = init_dft_files(vasp)

    for rank in ['unary', 'binary', 'ternary']:
        if rank in vasp.keys():
            vasp_files_by_rank[rank] = init_dft_files(vasp[rank])

    ncores_by_rank = {}
    for rank in ['unary', 'binary', 'ternary']:
        if rank in vasp.keys():
            ncores_by_rank[rank] = vasp[rank]['ncores']
        
    for config in dft_dataset.configurations:
        if config.rank == 1:
            rank_name = 'unary'
        elif config.rank == 2:
            rank_name = 'binary'
        elif config.rank == 3:
            rank_name = 'ternary'
    
        # use rank-level settings first
        vfs = vasp_files_by_rank[rank_name]
        config.set_vasp_files(incar=vfs['incar'], kpoints=vfs['kpoints'], potcar_names=vfs['potcar'])

        # use global-level settings second
        vfs = vasp_files_by_rank['global']
        config.set_vasp_files(overwrite=False, incar=vfs['incar'], kpoints=vfs['kpoints'], potcar_names=vfs['potcar'])

        config.cdir = dft_dir / str(config.cid)
        config.cdir.mkdir(exist_ok=True)
        config.write_vasp_files()

        config.ncores = ncores_by_rank[rank_name]

    dft_dataset.queue_jobs()
    dft_dataset.run_jobs()

dft_dataset.compute_mixing_energies()

plot_mixing_energy(TOP_DIR / 'DFT_mixing_energies.png', end_members, dft_dataset)
write_mixing_energy(TOP_DIR / 'DFT_mixing_energies.out', end_members, dft_dataset)

logger.debug(f'Computed, wrote, and plotting mixing energies')


######################## fit CE #######################

if args.train:
    logger.debug('Cluster expansion flag set. Fitting to DFT data')
    
    expan_dir = TOP_DIR / 'expansion'
    expan_dir.mkdir(exist_ok=True)

    expan: dict = input_yml['expansion']

    if 'cutoffs' not in expan.keys():
        cutoffs = []
    else:
        cutoffs = expan['cutoffs']

    if 'a0' not in expan.keys():
        a0 = 0
        for em in dft_dataset.end_members:
            a0 += ELEMENTS[em]['a0'] / len(dft_dataset.end_members)
    else:
        a0 = expan['a0']
    
    expansion = Expansion(dft_dataset, cutoffs, a0)

    if 'scan' in expan.keys():
        scan: dict = expan['scan']
        cutoff_vals = np.linspace(*scan['range'])

        scan_cutoffs(scan['order'], cutoff_vals.tolist(), expansion, expan_dir)

    if len(cutoffs):
        expansion.init_cluster_space()
        expansion.init_structure_container()
        expansion.fit()
        expansion.write(expan_dir)

        for conf in expansion.dataset.configurations:
            conf.expansion_energy = expansion.cluster_expansion.predict(conf.Atoms)

        plot_mixing_energy(TOP_DIR / 'DFT_mixing_energies.png', end_members, dft_dataset)
        write_mixing_energy(TOP_DIR / 'DFT_mixing_energies.out', end_members, dft_dataset)


####################### analysis ######################

if args.analysis:
    raise NotImplementedError('Analysis has not been implemented')
    analy: dict = input_yml['analysis']

    if 'convex_hull' in analy.keys():
        conv_hull: dict = analy['convex_hull']

        lattice = BCC(conv_hull['max_basis_size'])
        ch_dataset = Dataset(end_members, lattice)

        ch_dataset.compositions = ch_dataset.enumerate_compositions()
        ch_dataset.sublattices = ch_dataset.enumerate_sublattices()
        ch_dataset.configurations = ch_dataset.enumerate_configurations()
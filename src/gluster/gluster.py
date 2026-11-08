import argparse, yaml, logging, sys, itertools
from pathlib import Path
import numpy as np

from enumeration import BCC
from configuration import Configuration, OrderedConfiguration, DisorderedConfiguration
from mesh import Mesh, UnaryMesh, BinaryMesh, TernaryMesh
from sampling import Sampler
from vasp_file import VaspIncar, VaspKpoints
from dataset import Dataset
from expansion import Expansion, scan_cutoffs
from utils import ELEMENTS

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger('Gluster')


######################## input ########################

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='Path to input file')
parser.add_argument('--enumerate', '-E', action=argparse.BooleanOptionalAction)
parser.add_argument('--dft', '-D', action=argparse.BooleanOptionalAction)
parser.add_argument('--train', '-T', action=argparse.BooleanOptionalAction)
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

configs: dict[int, Configuration] = {}

if args.enumerate:
    logger.debug(f'Enumeration flag set. Generating configurations for a new input file')

    if 'enumerate' not in input_yml.keys():
        raise KeyError(f"Enumeration section missing in input file")

    if 'configurations' in input_yml.keys():
        raise KeyError(f"Configurations section already included in input file")
    
    enum: dict = input_yml['enumerate']
    meshes: dict[tuple[str], Mesh] = {}

    if 'unary' not in enum.keys():
        raise ValueError(f"Unaries are missing. They must be a list of elements")

    for el in enum['unary']:
        meshes[tuple(el)] = UnaryMesh(el)
        logger.debug(f'Mesh generated for {el}')

    if 'seed' not in enum.keys():
        seed = None
    else:
        seed = enum['seed']

    sampler = Sampler(seed)
    
    if 'binary' in enum.keys():
        binary_lattice = BCC(enum['binary']['L'])
        binary_lattice.enumerate_HNF()
        logger.debug('Initialized binary lattice and enumerated its HNF matrices')

        for binary in itertools.combinations(enum['unary'], 2):
            mesh = BinaryMesh(binary, binary_lattice, sampler, enum['binary'])
            meshes[binary] = mesh
            logger.debug(f'Mesh generated for {binary[0]}-{binary[1]}')

            mesh_dir = TOP_DIR / mesh.alloy_string
            mesh_dir.mkdir(exist_ok=True)

            mesh.plot(mesh_dir)
            mesh.plot_mesh_parameters(mesh_dir)
            mesh.plot_sampling_parameters(mesh_dir)

    if 'ternary' in enum.keys():
        ternary_lattice = BCC(enum['ternary']['L'])
        ternary_lattice.enumerate_HNF()
        logger.debug('Initialized ternary lattice and enumerated its HNF matrices')

        for ternary in itertools.combinations(enum['unary'], 3):
            mesh = TernaryMesh(ternary, ternary_lattice, sampler, enum['ternary'])
            meshes[ternary] = mesh
            logger.debug(f'Mesh generated for {ternary[0]}-{ternary[1]}-{ternary[2]}')

            mesh_dir = TOP_DIR / mesh.alloy_string
            mesh_dir.mkdir(exist_ok=True)

            mesh.plot(mesh_dir)
            mesh.plot_mesh_parameters(mesh_dir)
            mesh.plot_sampling_parameters(mesh_dir)

    new_configs_lines = ['\n\nconfigurations:\n']
    cid = 0

    for mesh in meshes.values():
        for config in mesh.configurations:
            new_configs_lines.append(" "*2 + f"{cid}:\n")

            H = []
            for row in config.H:
                H.append([int(val) for val in row])

            if config.ordered:
                new_configs_lines.append(" "*4 + "ordered: True\n")
                new_configs_lines.append(" "*4 + f"H: {H}\n")
            else:
                new_configs_lines.append(" "*4 + "ordered: False\n")

            new_configs_lines.append(" "*4 + f"size: {config.size}\n")
            new_configs_lines.append(" "*4 + "basis:\n")
            pos_line = " "*6 + f"positions: ["
            sp_line = " "*6 + f"species: ["

            for pos, sp in config.basis.items():
                pos_line += f"{list(pos)}, "
                sp_line += f"{sp}, "
     
            pos_line = pos_line[:-2] + ']\n' 
            sp_line = sp_line[:-2] + ']\n'

            new_configs_lines.append(pos_line)
            new_configs_lines.append(sp_line)

            cid += 1

    with open(TOP_DIR / 'new_input.yaml', 'w') as f:
        f.writelines(input_yml_lines + new_configs_lines)

    exit()


#################### configurations ###################

if 'configurations' not in input_yml.keys():
    raise KeyError(f"Configurations section missing in input file")

conf: dict[int, dict] = input_yml['configurations']
configs: dict[int, OrderedConfiguration | DisorderedConfiguration] = {}

for cid, conf_dict in conf.items():
    conf_dir = TOP_DIR / 'configurations' / str(cid)
    conf_dir.mkdir(exist_ok=True, parents=True)

    positions = [tuple(pos) for pos in conf_dict['basis']['positions']]
    species = conf_dict['basis']['species']
    basis = dict(zip(positions, species))
    size = conf_dict['size']

    if 'incar' in conf_dict.keys():
        incar = VaspIncar().load(from_string=conf_dict['incar'])
    else:
        incar = None

    if 'kpoints' in conf_dict.keys():
        kpoints = VaspKpoints().load(from_string=conf_dict['incar'])
    else:
        kpoints = None

    if 'potcar' in conf_dict.keys():
        potcar_names = conf_dict['potcar']
    else:
        potcar_names = None

    if conf_dict['ordered']:
        H = np.array(conf_dict['H'])
        configs[cid] = OrderedConfiguration(H, basis, size, cid=cid, cdir=conf_dir)
        configs[cid].map_to_supercell()
    else:
        configs[cid] = DisorderedConfiguration(basis, size, cid=cid, cdir=conf_dir)

    # use calculation-level vasp settings first
    configs[cid].set_vasp_files(incar=incar, kpoints=kpoints, potcar_names=potcar_names)
    configs[cid].write_vasp_files()

dataset = Dataset(configs)

logger.debug(f'Initialized {len(configs)} configurations in {TOP_DIR}')


####################### run VASP ######################

if args.dft:
    if 'vasp' not in input_yml.keys():
        raise KeyError(f"Vasp section missing in input file")

    vasp: dict[str, str | list[str]] = input_yml['vasp']

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
    
    for cid, config in configs.items():
        if config.rank == 1:
            rank_name = 'unary'
        elif config.rank == 2:
            rank_name = 'binary'
        elif config.rank == 3:
            rank_name = 'ternary'

        # use rank-level settings second
        vfs = vasp_files_by_rank[rank_name]
        config.set_vasp_files(overwrite=False, incar=vfs['incar'], kpoints=vfs['kpoints'], potcar_names=vfs['potcar'])

        # use global-level settings last
        vfs = vasp_files_by_rank['global']
        config.set_vasp_files(overwrite=False, incar=vfs['incar'], kpoints=vfs['kpoints'], potcar_names=vfs['potcar'])

        config.write_vasp_files()

        config.ncores = ncores_by_rank[rank_name]

    dataset.run_jobs()

dataset.compute_mixing_energies()
dataset.plot_mixing_energy(write_dir=TOP_DIR)
logger.debug(f'Computed, wrote, and plotting mixing energies')


######################## fit CE #######################

if args.train:
    expan: dict = input_yml['expansion']

    if 'cutoffs' not in expan.keys():
        cutoffs = []
    else:
        cutoffs = expan['cutoffs']

    if 'a0' not in expan.keys():
        a0 = 0
        for em in dataset.end_members.keys():
            a0 += ELEMENTS[em]['a0'] / len(dataset.end_members)
    else:
        a0 = expan['a0']
    
    expansion = Expansion(dataset, cutoffs, a0)

    if 'scan' in expan.keys():
        scan: dict = expan['scan']
        cutoff_vals = np.linspace(*scan['range'])

        scan_cutoffs(scan['order'], cutoff_vals.tolist(), expansion, TOP_DIR)

    if len(cutoffs):
        expansion.init_cluster_space()
        expansion.init_structure_container()
        expansion.fit(write_path = TOP_DIR / 'expansion.ce')
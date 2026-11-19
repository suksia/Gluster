import os, time, logging, yaml
from pathlib import Path
import numpy as np

from configuration import Configuration, OrderedConfiguration, DisorderedConfiguration
from utils import replicate
from enumeration import Lattice, Sublattice
from sampling import Sampler

try:
    NTASKS = int(os.environ['SLURM_NTASKS'])
except:
    NTASKS = 1

logger = logging.getLogger('Gluster')


class Dataset:
    """Collection of configurations."""
    def __init__(self, end_members: tuple[str]):
        self.end_members = end_members
        self.rank = len(end_members)

        self.equiatomic_composition = tuple([1/self.rank])*self.rank
        self.compositions: dict[tuple[float], list[int]] = None

        self.sublattices: dict[int, list[Sublattice]] = None
        self.configurations: list[Configuration] = None

        self.queue: dict[int, Configuration] = {}
        self.num_available_cores = NTASKS
        
        self.end_member_energies: dict[str, float] = None
        self.mixing_energies: dict[Configuration, float] = None

    def add_configurations(self, configs: list[Configuration]):
        if self.configurations is None:
            self.configurations = configs
        else:
            self.configurations += configs

    def sort_configurations(self):
        if self.configurations is None:
            raise RuntimeError("No configurations are defined")
        
        # sort by config ID
        self.configurations.sort(key = lambda config: config.cid)

    def remove_configurations_w_no_energy(self):
        if self.configurations is None:
            raise RuntimeError("No configurations are defined")
        
        configs = []

        for conf in self.configurations:
            if conf.energy:
                configs.append(conf)

        self.configurations = configs 
    
    def enumerate_unaries(self):
        return [OrderedConfiguration(np.eye(3), {(0,0,0): em}) for em in self.end_members]

    def enumerate_compositions(self, max_basis_size: int):
        """Iterates over basis sizes and compiles a list of allowed compositions."""
        compositions: dict[tuple[float], list[int]] = {}

        for l in range(self.rank, max_basis_size+1):
            for l in range(self.rank, max_basis_size+1):
                # binary
                if self.rank == 2:
                    for c1 in range(1, l):
                        # c1 must be between 1/l and (l-1)/l
                        comp = (c1/l, 1-c1/l)

                        if comp not in compositions.keys():
                            compositions[comp] = [l]
                        elif l not in compositions[comp]:
                            compositions[comp].append(l)

                # ternary
                elif self.rank == 3:
                    for c1 in range(1, l-1):
                        # c1 must be between 2/l and (l-2)/l
                        for c2 in range(1, l-c1):
                            # c2 must be between 1/l and (l-c1-1)/l
                            comp = (c1/l, c2/l, (l-c1-c2)/l)

                            if comp not in compositions.keys():
                                compositions[comp] = [l]
                            elif l not in compositions[comp]:
                                compositions[comp].append(l)

                elif self.rank > 3:
                    raise NotImplementedError("Only binary and ternary alloys have been implemented")
                
        return compositions

    def enumerate_sublattices(self, lattice: Lattice):
        """Enumerates all possible sublattices of the parent lattice."""
        lattice.enumerate_HNF()
        return lattice.sublattices

    def enumerate_configurations(self):
        """Iterates over compositions and compiles a list of decorations pulling from each allowed sublattice."""
        if self.compositions is None:
            raise RuntimeError("No compositions are defined")

        if self.sublattices is None:
            raise RuntimeError("No sublattices are defined")

        configurations: dict[tuple[float], dict[int, dict[Sublattice, list[OrderedConfiguration]]]] = {}

        for comp, allowed_ls in self.compositions.items():
            basis_size_dict = {}
            for l in allowed_ls:
                basis_size_dict[l] = {sublat: sublat.enumerate_bases(dict(zip(self.end_members, comp))) for sublat in self.sublattices[l]}
            configurations[comp] = basis_size_dict

        configurations_list: list[OrderedConfiguration] = []

        for basis_size_dict in configurations.values():
            for sublat_dict in basis_size_dict.values():
                for configs in sublat_dict.values():
                    configurations_list += configs

        return configurations, configurations_list

    def sample_ordered_configurations(self, sampler: Sampler, num: int, enum_configs_dict: dict, enum_configs_list: list):
        """Sample ordered configurations with an exponential decay for composition distance and basis size."""      
        if len(enum_configs_list) <= num:
            logger.debug(f"WARNING: too many unique ordered configurations requested (found {len(enum_configs_list)}, but needed at least {num})")
            return enum_configs_list

        comps = list(self.compositions.keys())
        ordered_configs = set()

        num_iters = 0
        
        while len(ordered_configs) < num:
            # sample a composition
            sampled_comp = comps[sampler.sample_uni_distr(len(comps))]

            # sample a basis size (l value)
            possible_ls = self.compositions[sampled_comp]
            if len(possible_ls) == 1:
                sampled_l = possible_ls[0]
            else:
                sampled_l = possible_ls[sampler.sample_uni_distr(len(possible_ls))]

            # sample a sublattice
            possible_sublats = self.sublattices[sampled_l]
            if len(possible_sublats) == 1:
                sampled_sublat = possible_sublats[0]
            else:
                sampled_sublat = possible_sublats[sampler.sample_uni_distr(len(possible_sublats))]

            # sample a configuration
            possible_configs: list[OrderedConfiguration] = enum_configs_dict[sampled_comp][sampled_l][sampled_sublat]
            if len(possible_configs) == 0:
                continue
            elif len(possible_configs) == 1:
                sampled_config = possible_configs[0]
            else:
                sampled_config = possible_configs[sampler.sample_uni_distr(len(possible_configs))]

            ordered_configs.add(sampled_config)

            num_iters += 1
            if num_iters > 100000:
                raise RuntimeError("Maximum iteration count reached. Something went wrong when sampling ordered configurations. Please investigate")

        return list(ordered_configs)

    def sample_disordered_configurations(self, sampler: Sampler, num: int, supercell_size: int):
        """Sample disordered configurations uniformly across the composition space."""
        disordered_decorations = []
        disordered_configs: list[DisorderedConfiguration] = []

        # define conventional supercell
        A_conv = np.eye(3)
        f_conv_basis = [np.array([0, 0, 0]), np.array([0.5, 0.5, 0.5])]
        supercell_size = tuple([supercell_size]*3)

        supercell_positions = replicate(A_conv, f_conv_basis, supercell_size)
        num_conv_positions = len(supercell_positions)

        supercell_positions_reduced = []
        for r in supercell_positions:
            f = np.array(r) / supercell_size
            supercell_positions_reduced.append(tuple(f.tolist()))

        supercell_positions = supercell_positions_reduced

        for di in range(num):
            # uniformly sample the composition space
            sampled_comp = []

            c1 = sampler.sample_uni_distr(num_conv_positions-self.rank)+1
            if self.rank == 2:
                c2 = num_conv_positions - c1
                sampled_comp = [c1, c2]

            elif self.rank == 3:
                c2 = sampler.sample_uni_distr(num_conv_positions-self.rank-1)+1
                c3 = num_conv_positions - c1 - c2
                sampled_comp = [c1, c2, c3]

            # define a random decoration following the sampled composition
            decoration: list[str] = []
            for c, el in zip(sampled_comp, self.end_members):
                decoration += [el]*c
            
            # ensure decoration has not already been used (should be rare but not impossible)
            duplicate_decoration = True
            while duplicate_decoration:
                sampler.py_rng.shuffle(decoration)

                if tuple(decoration) not in disordered_decorations:
                    duplicate_decoration = False

            disordered_decorations.append(tuple(decoration))

            # define configuration instance
            config = DisorderedConfiguration(dict(zip(supercell_positions, decoration)), supercell_size[0])
            disordered_configs.append(config)

        return disordered_configs
    
    def read_yaml(self, read_path: Path):
        """Reads a YAML file containing configuration data to add to the dataset. New configurations are reindexed."""
        if not read_path.exists():
            raise FileNotFoundError(f"[{read_path}] File could not be found")

        with open(read_path, 'r') as f:
            input_yaml: dict = yaml.safe_load(f)

        new_configs: list[Configuration] = []

        for cid, conf_data in input_yaml.items():
            positions = [tuple(pos) for pos in conf_data['basis']['positions']]
            species = conf_data['basis']['species']
            basis = dict(zip(positions, species))
            
            if conf_data['ordered']:
                H = np.array(conf_data['H'])
                config = OrderedConfiguration(H, basis, cid=cid)
            else:
                config = DisorderedConfiguration(basis, conf_data['size'], cid=cid)

            if 'energy' in conf_data.keys():
                config.energy = conf_data['energy']
                config.status = 2

            new_configs.append(config)

        return new_configs

    def write_yaml(self, write_path: Path):
        """Writes a YAML file containing configuration data for each configuration in the dataset."""
        if not write_path.parent.exists():
            raise FileNotFoundError(f"[{write_path.parent}] Directory could not be found")

        lines = []

        for config in self.configurations:
            lines.append(f"{config.cid}:\n")
            if config.ordered:
                H = []
                for row in config.H:
                    H.append([int(val) for val in row])

                lines.append(" "*2 + "ordered: True\n")
                lines.append(" "*2 + f"H: {H}\n")

            else:
                lines.append(" "*2 + "ordered: False\n")
                lines.append(" "*2 + f"size: {config.size}\n")

            lines.append(" "*2 + "basis:\n")
            pos_line = " "*4 + f"positions: ["
            sp_line = " "*4 + f"species: ["

            for pos, sp in config.basis.items():
                pos_line += f"{list(pos)}, "
                sp_line += f"{sp}, "

            pos_line = pos_line[:-2] + ']\n' 
            sp_line = sp_line[:-2] + ']\n'

            lines.append(pos_line)
            lines.append(sp_line)

            if config.energy is not None:
                lines.append(" "*2 + f"energy: {config.energy}\n")

        with open(write_path, 'w') as f:
            f.writelines(lines)

    def queue_jobs(self):
        if self.configurations is None:
            raise RuntimeError("No compositions are defined")

        for conf in self.configurations:
            if conf.status == 0:
                self.queue[conf.cid] = conf

    def run_jobs(self, write_path: Path):
        if len(self.queue) == 0:
            logger.debug(f"Queue is empty, skipping VASP")

        while len(self.queue):
            # launch jobs
            for cid, conf in self.queue.items():
                if conf.status == 1 or conf.status == 2:
                    continue

                if conf.ncores > NTASKS:
                    raise ValueError(f"Too many cores requested for config {cid}. Need {conf.ncores} but only have a maximum of {NTASKS}")
                elif self.num_available_cores < conf.ncores:
                    continue

                conf.run_vasp()

                self.num_available_cores -= conf.ncores
                conf.status = 1
                logger.debug(f'Running VASP on config {cid} with {conf.ncores} cores (# cores ready: {self.num_available_cores})')
                
                time.sleep(1) # give slurm time to add job steps
            
            # check statuses
            finished = []
            for cid, conf in self.queue.items():
                conf.poll_process()
                if conf.status == 1:
                    continue
                elif conf.status == 2:
                    self.num_available_cores += conf.ncores
                    self.write_yaml(write_path)
                    finished.append(cid)
                    logger.debug(f'VASP finished for config {cid} (# cores ready: {self.num_available_cores})')

            for fin_cid in finished:
                self.queue.pop(fin_cid)

    def compute_mixing_energies(self):
        # compute energy of end members
        self.end_member_energies = {}

        for em in self.end_members:
            for conf in self.configurations:
                if conf.rank != 1:
                    continue

                em_energy = None

                if em == conf.elements[0]:
                    em_energy = conf.energy
                else:
                    continue

            if em_energy is None:
                raise ValueError(f"Need energy for end-member {em} for calculating mixing energy, but could not find an energy for it in the dataset")
                
            self.end_member_energies[em] = em_energy

        # compute mixing energies
        self.mixing_energies = {}
        
        for conf in self.configurations:
            if conf.energy is None:
                continue

            conf.mixing_energy = conf.energy
            for sp, conc in conf.composition.items():
                conf.mixing_energy -= conc*self.end_member_energies[sp]

            self.mixing_energies[conf] = conf.mixing_energy
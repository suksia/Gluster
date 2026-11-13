import os, time, logging, itertools, yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

from configuration import Configuration, OrderedConfiguration, DisorderedConfiguration
from utils import alloy_string, composition_distance, replicate
from enumeration import Lattice, Sublattice, BCC
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
        self.mixing_energies: dict[Configuration, float]

    def add_configurations(self, configs: list[Configuration]):
        if self.configurations is None:
            self.configurations = configs
        else:
            self.configurations += configs
    
    def enumerate_unaries(self):
        unaries = []

        for em in self.end_members:
            unary = OrderedConfiguration(np.eye(3), {(0,0,0): em}, 1)
            
            # not mapping to supercell
            unary.supercell = deepcopy(unary.basis)
            unary.A_super = unary.A_prim.copy()

            unaries.append(unary)

        self.add_configurations(unaries)

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

    def refine_compositions(self, min_mu: float, max_mu: float, rho: float, gamma: float):
        """Remove compositions that are very close to ones with a smaller basis size, but increase the density near the equiatomic."""
        retained_compositions: dict[tuple[float], list[int]] = {}
                
        for comp, ls in self.compositions.items():
            if len(retained_compositions) == 0:
                retained_compositions[comp] = ls
                continue

            dxc = composition_distance(comp, self.equiatomic_composition)

            mu = min_mu + rho*(dxc)**gamma
            mu = min(mu, max_mu)

            retain = True
            for retained_comp in retained_compositions.keys():
                dx = composition_distance(comp, retained_comp)

                if dx == 0:
                    continue
                
                if dx <= mu:
                    retain = False

            if retain:
                retained_compositions[comp] = ls
        
        return retained_compositions

    def sample_ordered_configurations(self, sampler: Sampler, num: int, comp_decay: float, basis_decay: float, max_num_iters=10000):
        """Sample ordered configurations with an exponential decay for composition distance and basis size."""
        enum_configs, enum_configs_list = self.enumerate_configurations()
        
        if len(enum_configs_list) <= num:
            logger.debug(f"WARNING: too many unique ordered configurations requested (found {len(enum_configs_list)}, but needed at least {num})")
            return enum_configs_list

        else:
            # arrange compositions by distance to center and sort smallest to largest
            dxc_to_comps: dict[float, list[tuple[float]]] = {}

            for comp in self.compositions.keys():
                dxc = composition_distance(comp, self.equiatomic_composition)
            
                if dxc not in dxc_to_comps.keys():
                    dxc_to_comps[dxc] = [comp]
                else:
                    dxc_to_comps[dxc].append(comp)
            
            dxc_to_comps = dict(sorted(dxc_to_comps.items()))
            dxc_keys = [k for k in dxc_to_comps.keys()]
        
            num_iters = 0
            ordered_configs = set()
            
            while len(ordered_configs) < num:
                # sample a distance, then a composition
                sampled_dxc = dxc_keys[sampler.sample_exp_distr(len(dxc_keys), comp_decay)]
                sampled_comp = dxc_to_comps[sampled_dxc][sampler.sample_uni_distr(len(dxc_to_comps[sampled_dxc]))]

                # sample a basis size (l value)
                possible_ls = self.compositions[sampled_comp]
                if len(possible_ls) == 1:
                    sampled_l = possible_ls[0]
                else:
                    sampled_l = possible_ls[sampler.sample_exp_distr(len(possible_ls), basis_decay)]

                # sample a sublattice
                possible_sublats = self.sublattices[sampled_l]
                if len(possible_sublats) == 1:
                    sampled_sublat = possible_sublats[0]
                else:
                    sampled_sublat = possible_sublats[sampler.sample_uni_distr(len(possible_sublats))]

                # sample a configuration
                possible_configs = enum_configs[sampled_comp][sampled_l][sampled_sublat]
                if len(possible_configs) == 0:
                    continue
                elif len(possible_configs) == 1:
                    sampled_config = possible_configs[0]
                else:
                    sampled_config = possible_configs[sampler.sample_uni_distr(len(possible_configs))]

                ordered_configs.add(sampled_config)

                num_iters += 1
                if num_iters > max_num_iters:
                    raise RuntimeError(f"WARNING: Maximum iteration count reached. Keeping {len(ordered_configs)} ordered configurations")
                
            return list(ordered_configs)

    def sample_disordered_configurations(self, sampler: Sampler, num: int, supercell_size: int):
        """Sample disordered configurations uniformly across the composition space."""
        disordered_decorations = []
        disordered_configs: list[DisorderedConfiguration] = []

        # define conventional supercell
        A = np.eye(3)
        conv_basis_pos = [np.array([0, 0, 0]), np.array([0.5, 0.5, 0.5])]
        supercell_size = tuple([supercell_size]*3)

        supercell_positions = replicate(A, conv_basis_pos, supercell_size)
        num_conv_positions = len(supercell_positions)

        supercell_positions_reduced = []
        for pos in supercell_positions:
            pos = np.array(pos)
            pos /= supercell_size
            supercell_positions_reduced.append(tuple(pos.tolist()))

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
            config = DisorderedConfiguration(dict(zip(supercell_positions, decoration)), supercell_size)
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
                config = OrderedConfiguration(H, basis, conf_data['size'], cid=cid)
                config.map_to_supercell()
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

    def run_jobs(self):
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
                    finished.append(cid)
                    logger.debug(f'VASP finished for config {cid} (# cores ready: {self.num_available_cores})')

            for fin_cid in finished:
                self.queue.pop(fin_cid)

    def compute_mixing_energies(self):
        # compute energy of end members
        for conf in self.configurations:
            if conf.rank != 1:
                continue

            if conf.energy is None:
                el = conf.elements[0]
                raise ValueError(f"Need energy for end-member {el} (config {conf.cid}) for calculating mixing energy, but VASP may not have run or finished")

            self.end_member_energies[el] = conf.energy

        # compute mixing energies
        for conf in self.configurations:
            if conf.energy is None:
                continue
            
            conf.norm_energy = conf.energy / len(conf.supercell)

            conf.mixing_energy = conf.norm_energy
            for sp, conc in conf.composition.items():
                conf.mixing_energy -= conc*self.end_member_energies[sp]

            self.mixing_energies[conf] = conf.mixing_energy
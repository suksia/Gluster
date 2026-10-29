from pathlib import Path
import matplotlib.pyplot as plt
from math import pi, cos, sin, atan2
import numpy as np
from copy import copy, deepcopy

from utils import ELEMENTS, composition_distance, replicate
from configuration import Configuration, OrderedConfiguration, DisorderedConfiguration
from enumeration import Lattice, Sublattice
from sampling import Sampler

        
class Mesh:
    """A container for configurations sampled by meshing the configuration space."""
    rank = 0

    def __init__(self, elements: tuple[str], lattice: Lattice, sampler: Sampler, enum_parameters: dict):
        self.elements = elements
        self.lattice = lattice
        self.sampler = sampler
        
        self.alloy_string = ''
        for el in self.elements:
            self.alloy_string += el +'-'
        self.alloy_string = self.alloy_string[:-1]

        self.L = enum_parameters['L']

        if type(enum_parameters['mesh']) != list:
            mesh_params = [enum_parameters['mesh'], enum_parameters['mesh'], 0, 0]
        else:
            mesh_params = enum_parameters['mesh']

        self.min_mu, self.max_mu, self.rho, self.delta = mesh_params

        self.comp_decay = enum_parameters['sampling']['composition']
        self.basis_decay = enum_parameters['sampling']['basis_size']

        self.N_ordered = enum_parameters['N_ordered']
        self.N_disordered = enum_parameters['N_disordered']

        self.supercell_size = enum_parameters['supercell_size']

        self.eq_comp = tuple([1/type(self).rank])*type(self).rank

        self.all_compositions = self._enumerate_compositions()
        self.compositions = self._refine_mesh()

        self.all_configurations, self.tot_num_configs = self._enumerate_configurations()

        if self.tot_num_configs < self.N_ordered:
            raise ValueError(f"Requested {self.N_ordered} ordered configurations, but only {self.tot_num_configs} unique possibilities were found")

        self.configurations = self._sample_configurations()

    def _enumerate_compositions(self) -> dict[tuple[float], list[int]]:
        pass

    def _refine_mesh(self):
        retained_comps: dict[tuple[float], list[int]] = {}
        
        for comp, ls in self.all_compositions.items():
            if len(retained_comps) == 0:
                retained_comps[comp] = ls
                continue

            dxc = composition_distance(comp, self.eq_comp)

            mu = self.min_mu + self.rho*(dxc)**self.delta
            mu = min(mu, self.max_mu)

            retain = True
            for retained_comp in retained_comps.keys():
                dx = composition_distance(comp, retained_comp)

                if dx == 0:
                    continue
                
                if dx <= mu:
                    retain = False

            if retain:
                retained_comps[comp] = ls
        
        return retained_comps

    def _enumerate_configurations(self):
        if self.lattice.sublattices is None:
            self.lattice.enumerate_HNF()

        configs: dict[tuple[float], dict[int, dict[Sublattice, list[OrderedConfiguration]]]] = {}
        num_configs = 0

        for comp, ls in self.compositions.items():
            comp_w_els = dict(zip(self.elements, comp))
            comp_dict = {l: {} for l in ls} # basis size -> sublattices

            for l in ls:
                sublat_dict = {} # sublattice -> bases

                for sublat in self.lattice.sublattices[l]:
                    sublat.enumerate_bases(comp_w_els)

                    if len(sublat.decorations):
                        sublat_configs = [] # bases / configurations
                        for decoration in sublat.decorations:
                            basis = dict(zip(sublat.frac_positions, decoration))
                            sublat_configs.append(OrderedConfiguration(sublat.H, basis, self.supercell_size))

                        sublat_dict[sublat] = sublat_configs # skip sublattice if no decorations
                        num_configs += len(sublat_configs)

                comp_dict[l] = copy(sublat_dict)

            configs[comp] = copy(comp_dict)

        return configs, num_configs

    def _sample_configurations(self):
        ordered_configs = set()

        # arrange compositions by distance to center and sort smallest to largest
        dxc_to_comps: dict[float, list[tuple[float]]] = {}
        for comp in self.compositions.keys():
            dxc = composition_distance(comp, self.eq_comp)

            if dxc not in dxc_to_comps.keys():
                dxc_to_comps[dxc] = [comp]
            else:
                dxc_to_comps[dxc].append(comp)

        dxc_to_comps = dict(sorted(dxc_to_comps.items()))
        dxc_keys = [k for k in dxc_to_comps.keys()]

        while len(ordered_configs) < self.N_ordered:
            # sample a distance, then a composition
            sampled_dxc = dxc_keys[self.sampler.sample_exp_distr(len(dxc_keys), self.comp_decay)]
            sampled_comp = dxc_to_comps[sampled_dxc][self.sampler.sample_uni_distr(len(dxc_to_comps[sampled_dxc]))]

            # sample a basis size (l value)
            possible_ls = self.compositions[sampled_comp]
            if len(possible_ls) == 1:
                sampled_l = possible_ls[0]
            else:
                sampled_l = possible_ls[self.sampler.sample_exp_distr(len(possible_ls), self.basis_decay)]

            # sample a sublattice
            possible_sublats = [k for k in self.all_configurations[sampled_comp][sampled_l].keys()]
            if len(possible_sublats) == 1:
                sampled_sublat = possible_sublats[0]
            else:
                sampled_sublat = possible_sublats[self.sampler.sample_uni_distr(len(possible_sublats))]
            
            # sample a configuration
            possible_configs = self.all_configurations[sampled_comp][sampled_l][sampled_sublat]
            if len(possible_configs) == 0:
                raise ValueError(f"No configurations available")
            elif len(possible_configs) == 1:
                sampled_config = possible_configs[0]
            else:
                sampled_config = possible_configs[self.sampler.sample_uni_distr(len(possible_configs))]

            ordered_configs.add(sampled_config)

        ordered_configs: list[OrderedConfiguration] = list(ordered_configs)

        disordered_decorations = []
        disordered_configs: list[DisorderedConfiguration] = []

        # define conventional supercell
        A = np.eye(3)
        conv_basis_pos = [np.array([0, 0, 0]), np.array([0.5, 0.5, 0.5])]
        supercell_size = tuple([self.supercell_size]*3)

        supercell_positions = replicate(A, conv_basis_pos, supercell_size)
        num_conv_positions = len(supercell_positions)

        for di in range(self.N_disordered):
            # sample a distance, then a composition
            sampled_dxc = dxc_keys[self.sampler.sample_exp_distr(len(dxc_keys), self.comp_decay)]
            sampled_comp = dxc_to_comps[sampled_dxc][self.sampler.sample_uni_distr(len(dxc_to_comps[sampled_dxc]))]

            # define a random decoration following the sampled composition
            decoration: list[str] = []
            for el, conc in zip(self.elements, sampled_comp):
                decoration += [el]*round(conc*num_conv_positions)

            # adjust for rounding errors
            if len(decoration) == num_conv_positions:
                pass
            elif len(decoration) == num_conv_positions+1:
                decoration = decoration[1:]
            elif len(decoration) == num_conv_positions-1:
                decoration = [self.elements[0]] + decoration
            else:
                raise RuntimeError(f"Decoration for composition {sampled_comp} was {len(decoration)} long, but needed to be {num_conv_positions} long")

            # ensure decoration has not already been used (should be rare but not impossible)
            duplicate_decoration = True
            while duplicate_decoration:
                self.sampler.py_rng.shuffle(decoration)

                if tuple(decoration) not in disordered_decorations:
                    duplicate_decoration = False

            disordered_decorations.append(tuple(decoration))

            # define configuration instance
            comp_dict = dict(zip(self.elements, sampled_comp))
            basis = {(0,0,0): comp_dict}

            config = DisorderedConfiguration(dict(zip(supercell_positions, decoration)))
            disordered_configs.append(config)
        
        return ordered_configs + disordered_configs
    

class UnaryMesh(Mesh):
    """A single configuration containing just one element."""
    rank = 1

    def __init__(self, element: str):
        if element not in ELEMENTS.keys():
            raise ValueError(f"Unknown element {element}")
        
        self.element = element
        self.configuration = OrderedConfiguration(np.eye(3), {(0,0,0): self.element}, 1)

        # not mapping to supercell
        self.configuration.supercell = deepcopy(self.configuration.basis)
        self.configuration.A_super = self.configuration.A_prim.copy()

        self.configurations = [self.configuration]


class BinaryMesh(Mesh):
    """Contains all enumerated binary alloy compositions and a set of sampled configurations."""
    rank = 2

    def _enumerate_compositions(self):
        comps: dict[tuple[float], list[int]] = {}

        for l in range(BinaryMesh.rank, self.L+1):
            for c1 in range(1, l):
                # c1 must be between 1/l and (l-1)/l
                comp = (c1/l, 1-c1/l)

                if comp not in comps.keys():
                    comps[comp] = [l]
                else:
                    comps[comp].append(l)

        return comps
    

class TernaryMesh(Mesh):
    """Contains all enumerated ternary alloy compositions and a set of sampled configurations."""
    rank = 3

    def _enumerate_compositions(self):
        comps: dict[tuple[float], list[int]] = {}

        for l in range(TernaryMesh.rank, self.L+1):
            for c1 in range(1, l-1):
                # c1 must be between 2/l and (l-2)/l
                for c2 in range(1, l-c1):
                    # c2 must be between 1/l and (l-c1-1)/l
                    comp = (c1/l, c2/l, (l-c1-c2)/l)

                    if comp not in comps.keys():
                        comps[comp] = [l]
                    else:
                        comps[comp].append(l)

        return comps


def plot_sampled_configs(meshes: dict[str, Mesh]):
    return
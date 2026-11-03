from pathlib import Path
import matplotlib.pyplot as plt
from math import pi, cos, sin, atan2
import numpy as np
from copy import copy, deepcopy
import logging

from utils import ELEMENTS, composition_distance, replicate, alloy_string
from configuration import OrderedConfiguration, DisorderedConfiguration
from enumeration import Lattice, Sublattice
from sampling import Sampler

logging.getLogger("matplotlib").setLevel(logging.FATAL)

MAX_ENUM_SAMPLE_ITER_COUNT = 10000

        
class Mesh:
    """A container for configurations sampled by meshing the configuration space."""
    rank = 0

    def __init__(self, elements: tuple[str], lattice: Lattice, sampler: Sampler, enum_parameters: dict):
        self.elements = elements
        self.lattice = lattice
        self.sampler = sampler
        
        self.alloy_string = alloy_string(self.elements)

        self.L = enum_parameters['L']

        if type(enum_parameters['mesh']) != list:
            mesh_params = [enum_parameters['mesh'], enum_parameters['mesh'], 0, 0]
        else:
            mesh_params = enum_parameters['mesh']

        self.min_mu, self.max_mu, self.rho, self.gamma = mesh_params

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
        raise NotImplementedError("Only subclass methods are defined")

    def _refine_mesh(self):
        retained_comps: dict[tuple[float], list[int]] = {}
        
        for comp, ls in self.all_compositions.items():
            if len(retained_comps) == 0:
                retained_comps[comp] = ls
                continue

            dxc = composition_distance(comp, self.eq_comp)

            mu = self.min_mu + self.rho*(dxc)**self.gamma
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

        num_sample_iters = 0

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

            num_sample_iters += 1
            if num_sample_iters > MAX_ENUM_SAMPLE_ITER_COUNT:
                raise RuntimeError(f"Maximum iteration count reached ({MAX_ENUM_SAMPLE_ITER_COUNT}) during sampling loop for enumerated structures")

        ordered_configs: list[OrderedConfiguration] = list(ordered_configs)

        disordered_decorations = []
        disordered_configs: list[DisorderedConfiguration] = []

        # define conventional supercell
        A = np.eye(3)
        conv_basis_pos = [np.array([0, 0, 0]), np.array([0.5, 0.5, 0.5])]
        supercell_size = tuple([self.supercell_size]*3)

        supercell_positions = replicate(A, conv_basis_pos, supercell_size)
        num_conv_positions = len(supercell_positions)

        supercell_positions_reduced = []
        for pos in supercell_positions:
            pos = np.array(pos)
            pos /= self.supercell_size
            supercell_positions_reduced.append(tuple(pos.tolist()))

        supercell_positions = supercell_positions_reduced

        for di in range(self.N_disordered):
            # uniformly sample the composition space
            sampled_comp = []

            c1 = self.sampler.sample_uni_distr(num_conv_positions-self.rank)+1
            if type(self).rank == 2:
                c2 = num_conv_positions - c1
                sampled_comp = [c1, c2]

            elif type(self).rank == 3:
                c2 = self.sampler.sample_uni_distr(num_conv_positions-self.rank-1)+1
                c3 = num_conv_positions - c1 - c2
                sampled_comp = [c1, c2, c3]

            # define a random decoration following the sampled composition
            decoration: list[str] = []
            for c, el in zip(sampled_comp, self.elements):
                decoration += [el]*c
            
            # ensure decoration has not already been used (should be rare but not impossible)
            duplicate_decoration = True
            while duplicate_decoration:
                self.sampler.py_rng.shuffle(decoration)

                if tuple(decoration) not in disordered_decorations:
                    duplicate_decoration = False

            disordered_decorations.append(tuple(decoration))

            # define configuration instance
            config = DisorderedConfiguration(dict(zip(supercell_positions, decoration)), self.supercell_size)
            disordered_configs.append(config)
        
        return ordered_configs + disordered_configs

    def plot_mesh_parameters(self, write_dir: Path):
        x = np.linspace(0, 1.5, 50)

        y = self.min_mu + self.rho*(x)**self.gamma
        y = np.clip(y, a_min=self.min_mu, a_max=self.max_mu)

        fig = plt.figure()
        plt.plot(x, y)
        plt.xlabel("Distance to Equiatomic")
        plt.ylabel(r"$\mu$")
        plt.title(rf"{self.alloy_string} Mesh parameters ($\mu_\text{{min}}={self.min_mu}$, $\mu_\text{{max}}={self.max_mu}$, $\rho={self.rho}$, $\gamma={self.gamma}$)")

        fig.savefig(write_dir / f"{self.alloy_string} mesh params.png", bbox_inches='tight')

    def plot_sampling_parameters(self, write_dir: Path):
        k = 10
        i = np.arange(0, k)

        a_comp = self.comp_decay
        a_basis = self.basis_decay

        w_comp = a_comp*np.exp(-a_comp*i/k)
        w_basis = a_basis*np.exp(-a_basis*i/k)

        fig, axes = plt.subplots(2, 1, figsize=(4,6), sharex=True)
        axes: list[plt.Axes] = axes

        axes[0].bar(i, w_comp)
        axes[0].set_title(rf"Composition sampling ($\lambda={a_comp}$)")
        axes[0].set_ylabel("Probability")

        axes[1].bar(i, w_basis)
        axes[1].set_title(rf"Basis size sampling ($\lambda={a_basis}$)")
        axes[1].set_xlabel("Sampled Value")
        axes[1].set_ylabel("Probability")

        fig.savefig(write_dir / f"{self.alloy_string} sampling params.png", bbox_inches='tight')


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

    def plot(self, write_dir: Path):
        def plot_binary(comps: dict[tuple[float], int], write_path: Path):
            x, multiplicity = [], []
            for comp, mult in comps.items():
                x.append(comp[0])
                multiplicity.append(mult)

            fig, ax = plt.subplots(figsize=(8, 2))

            map = ax.scatter(x, [0]*len(x), c=multiplicity, cmap='plasma', zorder=3)

            ax.hlines(0, 0, 1, color='black', lw=1)
            ax.plot([0,1], [0,0], 'ko')
            
            ax.set_xticks([0.1*i for i in range(0, 11)])
            ax.set_xlim([-0.05,1.05])
            ax.tick_params(axis='x', length=8)
            ax.set_xlabel(f'$x({self.elements[0]})$')
            ax.set_yticks([])

            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_position(("data", 0))
            ax.spines['bottom'].set_bounds(0,1)

            cbar = fig.colorbar(map, orientation='horizontal', fraction=0.1, aspect=20, pad=0.2)   
            cbar.set_ticks(np.linspace(min(multiplicity), max(multiplicity), 4, dtype=int))
            cbar.set_label("Multiplicity")

            ax.set_title(f'{self.alloy_string} Mesh')

            if not write_path.parent.exists():
                raise ValueError(f"{write_path.parent} does not exist")
            fig.savefig(write_path, bbox_inches='tight')

        # plot enumerated compositions
        all_enum_comps = {comp: len(ls) for comp, ls in self.all_compositions.items()}
        plot_binary(all_enum_comps, write_dir / f"{self.alloy_string} all enumerated.png")

        ref_enum_comps = {comp: len(ls) for comp, ls in self.compositions.items()}
        plot_binary(ref_enum_comps, write_dir / f"{self.alloy_string} refined enumerated.png")

        # plot compositions of all sampled configurations
        ordered_comps, disordered_comps = {}, {}
        for config in self.configurations:
            config.compute_composition()

            comp = []
            for el in self.elements:
                comp.append(config.composition[el])
            comp = tuple(comp)

            if config.ordered:
                comp_dict = ordered_comps
            else:
                comp_dict = disordered_comps

            if comp in comp_dict.keys():
                comp_dict[comp] += 1
            else:
                comp_dict[comp] = 1

        plot_binary(ordered_comps, write_dir / f"{self.alloy_string} ordered.png")
        plot_binary(disordered_comps, write_dir / f"{self.alloy_string} disordered.png")
        

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

    def plot(self, write_dir: Path):
        def plot_ternary(comps: dict[tuple[float], int], write_path: Path):
            thetas, rs, multiplicity = [], [], []
            for comp, mult in comps.items():
                x = comp[0]*cos(pi) + comp[1]*cos(pi/3) + comp[2]*cos(5*pi/3)
                y = comp[0]*sin(pi) + comp[1]*sin(pi/3) + comp[2]*sin(5*pi/3)

                thetas.append(atan2(y, x))
                rs.append((x**2 + y**2)**(1/2))
                multiplicity.append(mult)

            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, layout='constrained', figsize=(6,6))

            angles = [pi, pi/3, 5*pi/3]
            colors = ['blue', 'darkgreen', 'darkred']
            offsets = [(0.02, 0.13),  (0, 0.05),  (-0.04, 0.09)]

            for i, el in enumerate(self.elements):
                ax.plot([angles[i], angles[i]], [0, 1], linewidth=2.5, color=colors[i])
                ax.text(angles[i]+offsets[i][0], 1+offsets[i][1], el, color=colors[i], size=15)
                ax.annotate('', xy=(angles[i], 1), xytext=(angles[i], 0.9), arrowprops=dict(arrowstyle='->', edgecolor=colors[i], lw=2.5))

            map = ax.scatter(thetas, rs, c=multiplicity, cmap='plasma', s=40, zorder=2)

            cbar = fig.colorbar(map, orientation='horizontal', fraction=0.033, aspect=20, pad=0.1)
            cbar.set_ticks(np.linspace(min(multiplicity), max(multiplicity), 4, dtype=int))
            cbar.ax.tick_params(labelsize=13)
            cbar.set_label("Multiplicity", fontsize=14)

            ax.set_rmax(1)
            ax.set_xticks([])
            ax.tick_params(labelbottom=False, labelleft=False)
            ax.set_title(f'{self.alloy_string} Mesh', pad=35, fontsize=16)

            if not write_path.parent.exists():
                raise ValueError(f"{write_path.parent} does not exist")
            fig.savefig(write_path, bbox_inches='tight')

        # plot enumerated compositions
        all_enum_comps = {comp: len(ls) for comp, ls in self.all_compositions.items()}
        plot_ternary(all_enum_comps, write_dir / f"{self.alloy_string} all enumerated.png")

        ref_enum_comps = {comp: len(ls) for comp, ls in self.compositions.items()}
        plot_ternary(ref_enum_comps, write_dir / f"{self.alloy_string} refined enumerated.png")

        # plot compositions of all sampled configurations
        ordered_comps, disordered_comps = {}, {}
        for config in self.configurations:
            config.compute_composition()

            comp = []
            for el in self.elements:
                comp.append(config.composition[el])
            comp = tuple(comp)

            if config.ordered:
                comp_dict = ordered_comps
            else:
                comp_dict = disordered_comps

            if comp in comp_dict.keys():
                comp_dict[comp] += 1
            else:
                comp_dict[comp] = 1

        plot_ternary(ordered_comps, write_dir / f"{self.alloy_string} ordered.png")
        plot_ternary(disordered_comps, write_dir / f"{self.alloy_string} disordered.png")
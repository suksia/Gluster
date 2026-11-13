import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from math import cos, sin, atan2, pi
import logging

from configuration import Configuration
from utils import alloy_string

logging.getLogger("matplotlib").setLevel(logging.FATAL)


def plot_sampling_parameters(write_path: Path, comp_decay: float, basis_decay: float):
    k = 10
    i = np.arange(0, k)

    w_comp = comp_decay*np.exp(-comp_decay*i/k)
    w_basis = basis_decay*np.exp(-basis_decay*i/k)

    fig, axes = plt.subplots(2, 1, figsize=(4,6), sharex=True)
    axes: list[plt.Axes] = axes

    axes[0].bar(i, w_comp)
    axes[0].set_title(rf"Composition sampling ($\lambda={comp_decay}$)")
    axes[0].set_ylabel("Probability")

    axes[1].bar(i, w_basis)
    axes[1].set_title(rf"Basis size sampling ($\lambda={basis_decay}$)")
    axes[1].set_xlabel("Sampled Value")
    axes[1].set_ylabel("Probability")

    fig.savefig(write_path, bbox_inches='tight')

def plot_mesh_parameters(write_path: Path, min_mu: float, max_mu: float, rho: float, gamma: float):
    x = np.linspace(0, 1.5, 50)

    y = min_mu + rho*(x)**gamma
    y = np.clip(y, a_min=min_mu, a_max=max_mu)

    fig = plt.figure()
    plt.plot(x, y)
    plt.xlabel("Distance to Equiatomic")
    plt.ylabel(r"$\mu$")
    plt.title(rf"Mesh parameters $\mu_\text{{min}}={min_mu}$, $\mu_\text{{max}}={max_mu}$, $\rho={rho}$, $\gamma={gamma}$)")

    fig.savefig(write_path, bbox_inches='tight')


def plot_binary_mesh(write_path: Path, elements: list[str], compositions: dict[tuple[float], int]):
    x, multiplicity = [], []
    for comp, mult in compositions.items():
        x.append(comp[0])
        multiplicity.append(mult)

    fig, ax = plt.subplots(figsize=(8, 2))

    map = ax.scatter(x, [0]*len(x), c=multiplicity, cmap='plasma', zorder=3)

    ax.hlines(0, 0, 1, color='black', lw=1)
    ax.plot([0,1], [0,0], 'ko')
    
    ax.set_xticks([0.1*i for i in range(0, 11)])
    ax.set_xlim([-0.05,1.05])
    ax.tick_params(axis='x', length=8)
    ax.set_xlabel(f'$x({elements[0]})$')
    ax.set_yticks([])

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position(("data", 0))
    ax.spines['bottom'].set_bounds(0,1)

    cbar = fig.colorbar(map, orientation='horizontal', fraction=0.1, aspect=20, pad=0.2)   
    cbar.set_ticks(np.linspace(min(multiplicity), max(multiplicity), 4, dtype=int))
    cbar.set_label("Multiplicity")

    al_str = alloy_string(elements)
    ax.set_title(f'{al_str} Mesh')

    if not write_path.parent.exists():
        raise FileNotFoundError(f"[{write_path.parent}] Directory does not exist")
    fig.savefig(write_path, bbox_inches='tight')

def plot_ternary_mesh(write_path: Path, elements: list[str], compositions: dict[tuple[float], int]):
    thetas, rs, multiplicity = [], [], []
    for comp, mult in compositions.items():
        x = comp[0]*cos(pi) + comp[1]*cos(pi/3) + comp[2]*cos(5*pi/3)
        y = comp[0]*sin(pi) + comp[1]*sin(pi/3) + comp[2]*sin(5*pi/3)

        thetas.append(atan2(y, x))
        rs.append((x**2 + y**2)**(1/2))
        multiplicity.append(mult)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, layout='constrained', figsize=(6,6))

    angles = [pi, pi/3, 5*pi/3]
    colors = ['blue', 'darkgreen', 'darkred']
    offsets = [(0.02, 0.13),  (0, 0.05),  (-0.04, 0.09)]

    for i, el in enumerate(elements):
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

    al_str = alloy_string(elements)
    ax.set_title(f'{al_str} Mesh', pad=35, fontsize=16)

    if not write_path.parent.exists():
        raise FileNotFoundError(f"[{write_path.parent}] Directory does not exist")
    fig.savefig(write_path, bbox_inches='tight')

def plot_mesh(write_path: Path, elements: list[str], compositions: dict[tuple[float], list[int]] = None, configurations: list[Configuration] = None):
    if compositions is not None:
        compositions = {comp: len(ls) for comp, ls in compositions.items()}

    elif configurations is not None:
        compositions = {}

        for config in configurations:
            comp = tuple([config.composition[el] for el in elements])

            if comp not in compositions.keys():
                compositions[comp] = 1
            else:
                compositions[comp] += 1

    if len(elements) == 2:
        plot_binary_mesh(write_path, elements, compositions)

    elif len(elements) == 3:
        plot_ternary_mesh(write_path, elements, compositions)


def plot_binary_mixing_energy(write_path: Path, elements: tuple[str], mixing_energies: dict[Configuration, float]):
    x_element = elements[0]
    all_str = alloy_string(elements)

    x, y, y_pred = [], [], []
    for conf, me in mixing_energies.items():
        skip = False

        if conf.rank > 2:
            skip = True

        for el in elements:
            if el not in conf.elements:
                skip = True
                break

        if skip:
            continue
        
        x.append(conf.composition[x_element])
        y.append(me*1000)

        if conf.expansion_energy:
            y_pred.append(conf.expansion_energy*1000)

    fig, ax = plt.subplots()

    ax.scatter(x, y, label='DFT')
    if len(y_pred):
        ax.scatter(x, y_pred, marker='x', label='CE')
    ax.set_xlabel(f"x({x_element})")
    ax.set_ylabel(f"Mixing Energy [meV]")
    ax.set_title(f"{all_str} Mixing Energy")
    ax.grid('True')
    ax.legend()

    fig.savefig(write_path, bbox_inches='tight')

def plot_ternary_mixing_energy(write_path: Path, elements: tuple[str], mixing_energies: dict[Configuration, float]):
    raise NotImplementedError('Only binary mixing enegy plotting has been implemented')

def plot_mixing_energy(write_path: Path, elements: tuple[str], mixing_energies: dict[Configuration, float]):
    if len(elements) == 2:
        plot_binary_mixing_energy(write_path, elements, mixing_energies)

    elif len(elements) == 3:
        plot_ternary_mixing_energy(write_path, elements, mixing_energies)

"""
def write_mixing_energy(write_path: Path, elements: tuple[str], mixing_energies: dict[Configuration, float]):
    with open(write_path, 'w') as f:
        f.write("End Member Energies [eV]\n\n")
        for end_mem, engy in end_member_energy.items():
            f.write(f"{end_mem} = {engy:2.5f}\n")

        f.write('\nMixing Energies [eV]\n\n')
        for conf, engy in self.mixing_energies.items():
            f.write(f"{conf.cid}\n")

            comp_line = ""
            for el, conc in conf.composition.items():
                comp_line += f"{el}={conc:1.3f}  "
            f.write(comp_line[:-2] + '\n')

            f.write(f"E/N  {conf.norm_energy:2.8f}\n")
            f.write(f"emix  {engy:1.6f}\n")
            if conf.expansion_energy:
                f.write(f"pred {conf.expansion_energy:1.6f}\n")
            else:
                f.write(f"pred \n")
            f.write('\n')
"""
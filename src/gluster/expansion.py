from ase import build
from icet import ClusterSpace, StructureContainer, ClusterExpansion
from trainstation import CrossValidationEstimator
from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes

from dataset import Dataset

class Expansion:
    def __init__(self, dataset: Dataset, cutoffs: list[float], a0: float):
        self.dataset = dataset

        self.cutoffs = cutoffs
        self.a0 = a0

        self.cluster_space: ClusterSpace = None
        self.structure_container: StructureContainer = None
        self.cluster_expansion: ClusterExpansion = None
        self.optimizer: CrossValidationEstimator = None

        self.rmse_train, self.rmse_val = None, None # root mean-squared error
        self.bic = None  # bayesian information criterion
        self.num_params, self.num_nonzero_params = None, None

    def init_cluster_space(self):
        self.prim = build.bulk('W', 'bcc', self.a0)
        self.cluster_space = ClusterSpace(
            structure=self.prim, 
            cutoffs=self.cutoffs, 
            chemical_symbols=[em for em in self.dataset.end_members.keys()])

    def init_structure_container(self):
        if self.cluster_space is None:
            raise RuntimeError(f"Tried to initialize structure container, but cluster space does not exist")
        self.structure_container = StructureContainer(cluster_space=self.cluster_space)

        for conf in self.dataset.configs.values():
            self.structure_container.add_structure(
                structure=conf.Atoms, 
                user_tag=conf.cid, 
                properties={'mixing_energy': conf.mixing_energy})

    def fit(self, write_path: Path = None):
        if self.structure_container is None:
            raise RuntimeError(f"Tried to fit cluster expansion, but structure container does not exist")
        
        # fit cluster expansion
        self.optimizer = CrossValidationEstimator(
            fit_data= self.structure_container.get_fit_data(key='mixing_energy'), 
            fit_method='ardr')
        self.optimizer.validate()
        self.optimizer.train()

        self.rmse_train = self.optimizer.rmse_train
        self.rmse_val = self.optimizer.rmse_validation
        self.bic = self.optimizer.BIC
        self.num_params = self.optimizer.n_parameters
        self.num_nonzero_params = self.optimizer.n_nonzero_parameters

        self.cluster_expansion = ClusterExpansion(
            cluster_space=self.cluster_space, 
            parameters=self.optimizer.parameters, 
            metadata=self.optimizer.summary)

        if write_path:
            self.cluster_expansion.write(write_path)


def scan_cutoffs(order: int, cutoff_values: list[float], expansion: Expansion, write_dir: Path):
    if len(expansion.cutoffs) != order-2:
        raise ValueError(f"Need {order-2} cutoffs for lower order clusters in order to scan the cutoff for order {order}")

    scan_data = {'cutoff': [], 'rmse_train': [], 'rmse_val': [], 'bic': [], 'num_params': [], 'num_nonzero_params': []}

    for co in cutoff_values:
        current_expansion = deepcopy(expansion)
        current_expansion.cutoffs.append(co)

        current_expansion.init_cluster_space()
        current_expansion.init_structure_container()
        current_expansion.fit()

        scan_data['cutoff'].append(co)
        scan_data['rmse_train'].append(current_expansion.rmse_train)
        scan_data['rmse_val'].append(current_expansion.rmse_val)
        scan_data['bic'].append(current_expansion.bic)
        scan_data['num_params'].append(current_expansion.num_params)
        scan_data['num_nonzero_params'].append(current_expansion.num_nonzero_params)

    with open(write_dir / f'scan_{order}.out', 'w') as f:
        f.write(f"Cutoffs = {expansion.cutoffs}\n")
        f.write(f"Order = {order}\n\n")

        f.write(f"{'Cutoff':<20} {'RMSE Train':<20} {'RMSE Val':<20} {'BIC':<20} {'# Params':<20} {'# Nonzero Params':<20}\n\n")
        for i, co in enumerate(cutoff_values):
            f.write(f"{co:<20.2f} {scan_data['rmse_train'][i]:<20.2f} {scan_data['rmse_val'][i]:<20.2f} {scan_data['bic'][i]:<20.2f} {scan_data['num_params'][i]:<20} {scan_data['num_nonzero_params']:<20}\n")

    fig, axes = plt.subplots(3, 1, figsize=(4, 6), sharex=True)
    axes: list[Axes] = axes

    ax = axes[0]
    ax.plot(cutoff_values, scan_data['rmse_train'], '-o', label='Train')
    ax.plot(cutoff_values, scan_data['rmse_validation'], '-s', label='Validation')
    ax.set_ylabel('RMSE')
    ax.legend()

    ax = axes[1]
    ax.plot(cutoff_values, scan_data['bic'], '-o')
    ax.set_ylabel('BIC')

    ax = axes[2]
    ax.plot(cutoff_values, scan_data['num_params'], '-o', label='Total')
    ax.plot(cutoff_values, scan_data['num_nonzero_params'], '-o', label='Nonzero')
    ax.set_xlabel(f'Order {order} cutoff (Å)')
    ax.set_ylabel('Number of parameters')
    ax.legend()

    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    fig.align_ylabels(axes)

    fig.savefig(write_dir / f'scan_{order}.png', bbox_inches='tight')